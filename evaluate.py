import os
import sys
import torch
import numpy as np

import stim
import sinter
import multiprocessing
import pathlib
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cascade"))

from training.train import get_stim_to_grid_mapping
from cascade.model.surface_cascade import SurfaceCascade
from cascade.geometry.surface_code import (
    checkerboard_ancilla_mask,
    synthetic_data_qubit_mask,
    synthetic_logical_masks,
)
from cascade.model.embedding import syndrome_indices_from_detections
from circuits.surface_code import generate_surface_code_circuit
from plotting.plot_threshold import draw_and_save_plot

class CascadeSinterDecoder(sinter.Decoder):
    """
    Custom Sinter Decoder that uses our trained PyTorch Cascade model.
    """
    def __init__(self, distance: int, model_path: str):
        self.distance = distance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing CascadeSinterDecoder on {self.device}")
        
        # 1. Recreate mappings (assuming p_rate=0.01 for coordinate extraction, coordinates are topological anyway)
        self.mapping, (self.T, self.H, self.W) = get_stim_to_grid_mapping(distance, 0.01)
        self.mapping = self.mapping.to(self.device)
        
        ancilla_mask = checkerboard_ancilla_mask(distance)
        data_qubit_mask = synthetic_data_qubit_mask(distance, ancilla_mask)
        logical_masks = synthetic_logical_masks(distance, data_qubit_mask)[0:1] # Match train.py fix
        
        # 2. Load Model
        self.model = SurfaceCascade(
            distance=distance,
            rounds=self.T,
            hidden_dim=32,
            depth=distance,
            data_qubit_mask=data_qubit_mask,
            logical_masks=logical_masks,
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval() # Set to evaluation mode
        
        self.ancilla_mask = ancilla_mask.to(self.device)

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        """
        Sinter calls this method for every chunk of simulation data.
        We read the bit-packed syndromes, run our PyTorch model, and save the predictions.
        """
        # 1. Read input syndromes from stim bit-packed file
        # This returns a boolean numpy array of shape (num_shots, num_dets)
        dets_np = stim.read_shot_data_file(
            path=str(dets_b8_in_path),
            format="b8",
            num_measurements=num_dets,
            bit_packed=False
        )
        
        # 2. Convert to torch tensor and move to GPU
        inputs = torch.from_numpy(dets_np).to(self.device)
        
        # 3. Process in batches to avoid OOM if num_shots is large
        batch_size = 1000
        predictions_list = []
        
        with torch.no_grad():
            for i in range(0, num_shots, batch_size):
                batch_inputs = inputs[i:i+batch_size]
                b_sz = batch_inputs.size(0)
                
                # Reshape Magic (same as train.py)
                detections = torch.zeros(b_sz, self.T, self.H, self.W, device=self.device, dtype=torch.bool)
                t_idx, y_idx, x_idx = self.mapping[:, 0], self.mapping[:, 1], self.mapping[:, 2]
                detections[:, t_idx, y_idx, x_idx] = batch_inputs > 0
                
                syndrome_idx = syndrome_indices_from_detections(detections, self.ancilla_mask)
                
                # Forward pass
                logits = self.model(syndrome_idx) # (Batch, 1)
                
                # BCEWithLogitsLoss uses raw logits. A logit > 0 means prob > 0.5 (True)
                preds = (logits > 0.0).cpu().numpy().astype(bool)
                predictions_list.append(preds)
                
        # Combine all batches
        all_predictions = np.concatenate(predictions_list, axis=0)
        
        # 4. Write output to bit-packed file for sinter
        stim.write_shot_data_file(
            data=all_predictions,
            path=str(obs_predictions_b8_out_path),
            format="b8",
            num_measurements=0,
            num_detectors=0,
            num_observables=num_obs
        )


def evaluate_model():
    target_distance = 7
    model_path = f"checkpoints/cascade_d{target_distance}.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please train the model first.")
        return

    # 1. Create our custom decoder instance
    my_decoder = CascadeSinterDecoder(distance=target_distance, model_path=model_path)
    
    # 2. Define simulation tasks
    # For a fair evaluation, we test on unseen data at various error rates
    '''
    Different p rates 
    
    p_rates = [
        1e-1,   # 0.1 
        5e-2, 
        2e-2, 
        1e-2, 
        5e-3, 
        2e-3
        ]
    '''
    p_rates = [
        1e-2,   # 0.01
        5e-3,   # 0.005
        2e-3,   # 0.002
        1e-3,   # 0.001
        5e-4,   # 0.0005
        2e-4,   # 0.0002
        1e-4,   # 0.0001
        
    ]
    
    tasks = []
    for p in p_rates:
        circuit = generate_surface_code_circuit(distance=target_distance, physical_error_rate=p)
        tasks.append(
            sinter.Task(
                circuit=circuit,
                json_metadata={'d': target_distance, 'p': p}
            )
        )

    print(f"Starting Sinter evaluation using PyTorch Cascade Decoder...")
    
    # 3. Run Sinter Collect with our custom decoder
    stats = sinter.collect(
        num_workers=1, # Neural network batching is done internally, so 1 worker is safer for GPU memory
        tasks=tasks,
        decoders=['my_cascade'],
        custom_decoders={'my_cascade': my_decoder},
        max_shots=100_000,
        max_errors=1000,
        print_progress=True
    )
    
    
    # 4. Draw and save plot
    os.makedirs("plotting/output", exist_ok=True)
    draw_and_save_plot(stats, save_path=f"plotting/output/cascade_evaluation_d{target_distance}.png")

if __name__ == "__main__":
    evaluate_model()
