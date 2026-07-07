import os
import stim
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import circuits module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from circuits.surface_code import generate_surface_code_circuit

def generate_and_save_data(distance: int, p_rate: float, shots: int, save_dir: str):
    """
    Runs stim simulation for a given distance and physical error rate (p_rate),
    and saves the inputs (Syndrome) and ground truth labels (Observable) to a file.
    """
    print(f"[{distance}] Starting data generation for distance {distance}, p={p_rate}... (Shots: {shots})")
    
    # 1. Generate circuit
    circuit = generate_surface_code_circuit(distance=distance, physical_error_rate=p_rate)
    
    # 2. Compile high-speed sampler
    sampler = circuit.compile_detector_sampler()
    
    # 3. Sample data (separate_observables=True is the key!)
    # detectors: Model inputs (X)
    # observables: Ground truth labels the model needs to predict (Y)
    detectors, observables = sampler.sample(shots=shots, separate_observables=True)
    
    # Convert boolean arrays to float32 (for PyTorch training)
    X = detectors.astype(np.float32)
    Y = observables.astype(np.float32)
    
    # 4. Save to file (using compressed .npz format)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"dataset_d{distance}_p{p_rate}.npz")
    np.savez_compressed(save_path, X=X, Y=Y)
    
    print(f"[{distance}] Save complete! -> {save_path}")
    print(f"      - X (Syndrome) shape: {X.shape}")
    print(f"      - Y (Observable) shape: {Y.shape}\n")

if __name__ == "__main__":
    # Directory to save datasets
    DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
    
    # Target distances to generate
    """
    =================== settings ===================
    d = 3, 5, 7, 9, 11
    p = 0.01 (error rate)
    """
    target_distances = [3, 5, 7, 9, 11]
    
    # Physical error rate for training (usually around the threshold ~1%, 
    # or a mix of multiple rates. Here we use a single value 0.01 as an example)
    training_p = 0.01 
    
    # Large dataset size for powerful GPU training (e.g., RTX 6000)
    num_shots = 1_000_000 
    
    for d in target_distances:
        generate_and_save_data(
            distance=d, 
            p_rate=training_p, 
            shots=num_shots, 
            save_dir=DATASET_DIR
        )
