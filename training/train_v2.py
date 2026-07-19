import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

# Add parent directory to path to allow importing the circuits
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the Cascade directory directly to path to allow importing 'cascade'
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Cascade"))

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


def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(torch.float32)
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    continue
                
                # Flatten >2D params (like Conv3d weights) to 2D
                g_2d = g.view(g.size(0), -1)
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g_2d)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g_2d)
                
                if nesterov:
                    g_2d = g_2d.add(buf, alpha=momentum)
                else:
                    g_2d = buf
                
                g_out = zeropower_via_newtonschulz5(g_2d, steps=ns_steps)
                
                # Scale by learning rate and aspect ratio
                scale = lr * max(1, g_2d.size(0)/g_2d.size(1))**0.5
                p.data.add_(g_out.view_as(p), alpha=-scale)


def train_model(distance: int, p_start: float, p_target: float, total_steps: int = 80000, batch_size: int = 3328, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if SurfaceCascade is None:
        print("Cannot initialize model without SurfaceCascade import. Exiting.")
        return

    # 1. Prepare Geometry and Mapping
    mapping, (T, H, W) = get_stim_to_grid_mapping(distance, p_target)
    mapping = mapping.to(device)
    
    # Prepare sampler for on-the-fly data generation
    circuit = generate_surface_code_circuit(distance, p_start)
    sampler = circuit.compile_detector_sampler()
    
    # Create masks on CPU first to avoid device mismatch inside geometry functions
    ancilla_mask = checkerboard_ancilla_mask(distance)
    data_qubit_mask = synthetic_data_qubit_mask(distance, ancilla_mask)
    logical_masks = synthetic_logical_masks(distance, data_qubit_mask)
    
    # --- IMPORTANT FIX ---
    logical_masks = logical_masks[0:1]

    # 2. Initialize Model
    model = SurfaceCascade(
        distance=distance,
        rounds=T,
        hidden_dim=32,
        depth=distance,
        data_qubit_mask=data_qubit_mask,
        logical_masks=logical_masks,
    ).to(device)

    ancilla_mask = ancilla_mask.to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    # 2.5 Initialize Hybrid Optimizers
    muon_params = []
    adamw_params = []
    for p in model.parameters():
        if p.requires_grad:
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
                
    optimizer_muon = Muon(muon_params, lr=lr * 10)
    optimizer_adamw = optim.AdamW(adamw_params, lr=lr)

    # 2.6 Initialize Learning Rate Schedulers (Cosine Annealing)
    # Muon optimizer lr decays to lr*1 (10x drop), AdamW decays to lr*0.1 (10x drop)
    scheduler_muon = optim.lr_scheduler.CosineAnnealingLR(optimizer_muon, T_max=total_steps, eta_min=lr)
    scheduler_adamw = optim.lr_scheduler.CosineAnnealingLR(optimizer_adamw, T_max=total_steps, eta_min=lr * 0.1)

    # 3. Training Loop
    print("Starting training...")
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    arr_loss = []
    
    # --- 3-Phase Curriculum Schedule ---
    warmup_steps = int(total_steps * 0.02)   # 2% Warmup (fixed at p_start)
    anneal_steps = int(total_steps * 0.08)   # 8% Annealing (p_start -> p_target)
    current_p = p_start
    
    for step in tqdm(range(total_steps)):
        # Curriculum Learning 3-Phase Update
        if step < warmup_steps:
            new_p = p_start
        elif step < warmup_steps + anneal_steps:
            progress = (step - warmup_steps) / anneal_steps
            new_p = p_start + (p_target - p_start) * progress
        else:
            new_p = p_target
            
        if new_p != current_p or step == 0:
            current_p = new_p
            circuit = generate_surface_code_circuit(distance, current_p)
            sampler = circuit.compile_detector_sampler()
                
        # Generate data on-the-fly
        detectors, observables = sampler.sample(shots=batch_size, separate_observables=True)
        inputs = torch.from_numpy(detectors.astype(np.float32)).to(device)
        targets = torch.from_numpy(observables.astype(np.float32)).to(device)
        
        # --- The Reshape Magic ---
        batch_sz = inputs.size(0)
        detections = torch.zeros(batch_sz, T, H, W, device=device, dtype=torch.bool)
        
        t_idx = mapping[:, 0]
        y_idx = mapping[:, 1]
        x_idx = mapping[:, 2]
        detections[:, t_idx, y_idx, x_idx] = inputs > 0
        
        syndrome_idx = syndrome_indices_from_detections(detections, ancilla_mask)
        
        optimizer_muon.zero_grad()
        optimizer_adamw.zero_grad()
        
        outputs = model(syndrome_idx)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer_muon.step()
        optimizer_adamw.step()
        
        # Step the learning rate schedulers
        scheduler_muon.step()
        scheduler_adamw.step()
        
        running_loss += loss.item()
        
        # Accuracy
        with torch.no_grad():
            predictions = (outputs > 0).float()
            corrects = (predictions == targets).sum().item()
            batch_acc = corrects / targets.numel()
            running_acc += batch_acc 
            
        if (step + 1) % 100 == 0:
            avg_loss = running_loss / 100
            avg_acc = running_acc / 100
            current_lr_adam = scheduler_adamw.get_last_lr()[0]
            tqdm.write(f"Step [{step+1}/{total_steps}] - p: {current_p:.5f} - LR: {current_lr_adam:.5e} - Loss: {avg_loss:.6f} - Acc: {avg_acc:.4f}")
            running_loss = 0.0
            running_acc = 0.0
            arr_loss.append(avg_loss)   
        
    # ================== Saving trained model ==================
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"cascade_d{distance}_v2.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Model saved to {save_path}")
    
    # ================== Plotting loss & Accuracy ==================
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_steps = [i * 100 for i in range(len(arr_loss))]
    ax.plot(x_steps, arr_loss, label=f"d= {distance}", marker="o", markersize=3)
    
    ax.set_title(f"Loss vs steps (3-Phase Curriculum)")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("BCE Loss logits")
    ax.grid(True, which="both", linestyle='--', alpha=0.7)
    
    ax.legend()
    plot_path = os.path.join(plot_dir, f"loss_plot_d{distance}_v2.png")
    plt.savefig(plot_path)
    print(f"plot saved to {plot_path}")
    
    # plt.show() # Commented out to prevent blocking in non-interactive environments

if __name__ == "__main__":
    TARGET_DISTANCE = 7
    P_START = 0.001
    P_TARGET = 0.01
    
    # Curriculum learning with batch size 3328
    train_model(distance=TARGET_DISTANCE, p_start=P_START, p_target=P_TARGET, total_steps=80000, batch_size=3328)
