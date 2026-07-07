import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path to allow importing the Cascade model and circuits
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from cascade.model.surface_cascade import SurfaceCascade
    from cascade.geometry.surface_code import (
        checkerboard_ancilla_mask,
        synthetic_data_qubit_mask,
        synthetic_logical_masks,
    )
    from cascade.model.embedding import syndrome_indices_from_detections
    from circuits.surface_code import generate_surface_code_circuit
except ImportError as e:
    print(f"Warning: Could not import required modules. Please check the path. ({e})")
    SurfaceCascade = None


def get_stim_to_grid_mapping(distance: int, p_rate: float):
    """
    Extracts detector coordinates from the stim circuit and creates a mapping
    from the flat 1D detector array to the 3D (T, H, W) grid expected by Cascade.
    """
    circuit = generate_surface_code_circuit(distance, p_rate)
    coords = circuit.get_detector_coordinates()
    
    x_vals = sorted(list(set(c[0] for c in coords.values())))
    y_vals = sorted(list(set(c[1] for c in coords.values())))
    t_vals = sorted(list(set(c[2] for c in coords.values())))
    
    x_map = {v: i for i, v in enumerate(x_vals)}
    y_map = {v: i for i, v in enumerate(y_vals)}
    t_map = {v: i for i, v in enumerate(t_vals)}
    
    T, H, W = len(t_map), len(y_map), len(x_map)
    
    num_detectors = len(coords)
    # Map: index -> (t, y, x)
    mapping = torch.zeros(num_detectors, 3, dtype=torch.long)
    for d_idx, (x, y, t) in coords.items():
        mapping[d_idx, 0] = t_map[t]
        mapping[d_idx, 1] = y_map[y]
        mapping[d_idx, 2] = x_map[x]
        
    return mapping, (T, H, W)


class SyndromeDataset(Dataset):
    def __init__(self, npz_path: str):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")
        
        data = np.load(npz_path)
        self.X = torch.from_numpy(data['X'])
        self.Y = torch.from_numpy(data['Y'])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_model(distance: int, p_rate: float, epochs: int = 50, batch_size: int = 1024, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
    npz_path = os.path.join(dataset_dir, f"dataset_d{distance}_p{p_rate}.npz")
    
    try:
        dataset = SyndromeDataset(npz_path)
    except FileNotFoundError as e:
        print(e)
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded dataset for distance {distance}. Total samples: {len(dataset)}")

    if SurfaceCascade is None:
        print("Cannot initialize model without SurfaceCascade import. Exiting.")
        return

    # 1. Prepare Geometry and Mapping
    mapping, (T, H, W) = get_stim_to_grid_mapping(distance, p_rate)
    mapping = mapping.to(device)
    
    ancilla_mask = checkerboard_ancilla_mask(distance).to(device)
    data_qubit_mask = synthetic_data_qubit_mask(distance, ancilla_mask).to(device)
    logical_masks = synthetic_logical_masks(distance, data_qubit_mask).to(device)

    # 2. Initialize Model
    model = SurfaceCascade(
        distance=distance,
        rounds=T,          # Model expects rounds to match the temporal dimension of our grid
        hidden_dim=32,     # Adjust model capacity as needed
        depth=distance,    # Following the L ~ d heuristic from the paper
        data_qubit_mask=data_qubit_mask,
        logical_masks=logical_masks,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # --- The Reshape Magic ---
            # 1. Create empty 3D grid: (Batch, T, H, W)
            batch_sz = inputs.size(0)
            detections = torch.zeros(batch_sz, T, H, W, device=device, dtype=torch.bool)
            
            # 2. Scatter the 1D stim flat array into the 3D grid
            # inputs is (Batch, num_detectors). mapping is (num_detectors, 3)
            # We map each detector's value to its (t, y, x) position.
            t_idx = mapping[:, 0]
            y_idx = mapping[:, 1]
            x_idx = mapping[:, 2]
            detections[:, t_idx, y_idx, x_idx] = inputs > 0
            
            # 3. Convert boolean detections to the {0,1,2} embedding vocab expected by Cascade
            syndrome_idx = syndrome_indices_from_detections(detections, ancilla_mask)
            
            optimizer.zero_grad()
            
            # Forward pass expects syndrome_idx
            outputs = model(syndrome_idx)
            
            # Targets from stim might be multiple observables, we need to match model output
            # Cascade readout returns (Batch, num_logicals)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.6f}")

    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"cascade_d{distance}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Model saved to {save_path}")

if __name__ == "__main__":
    TARGET_DISTANCE = 5
    TARGET_P = 0.01
    
    # Using smaller batch for initial testing, bump back to 1024 for full GPU power
    train_model(distance=TARGET_DISTANCE, p_rate=TARGET_P, epochs=50, batch_size=512)
