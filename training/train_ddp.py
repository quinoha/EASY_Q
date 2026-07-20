import os
# --- Specify GPUs to use ---
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

# Add parent directory to path to allow importing the circuits
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_stim_to_grid_mapping(distance: int, p_rate: float):
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
    
    mapping = torch.zeros(num_detectors, 3, dtype=torch.long)
    for d_idx, (x, y, t) in coords.items():
        mapping[d_idx, 0] = t_map[t]
        mapping[d_idx, 1] = y_map[y]
        mapping[d_idx, 2] = x_map[x]
        
    return mapping, (T, H, W)


def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
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
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
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
                
                if not torch.isfinite(g).all():
                    continue
                
                if group['weight_decay'] > 0.0:
                    p.data.mul_(1.0 - lr * group['weight_decay'])
                
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
                scale = lr * max(1, g_2d.size(0)/g_2d.size(1))**0.5
                p.data.add_(g_out.view_as(p), alpha=-scale)


class EMA:
    def __init__(self, model, decay=0.9998):
        self.decay = decay
        self.model = model
        self.shadow = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone().detach()

    @torch.no_grad()
    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def copy_to(self, model):
        state_dict = model.state_dict()
        for name, p in self.shadow.items():
            state_dict[name].copy_(p)
        model.load_state_dict(state_dict)


def train_worker(rank: int, world_size: int, distance: int, p_start: float, p_target: float, total_steps: int, batch_size: int, lr: float):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print(f"Starting DDP training with {world_size} GPUs!")

    # Calculate local batch size (each GPU generates its own independent shots)
    local_batch_size = batch_size // world_size

    # 1. Prepare Geometry and Mapping
    mapping, (T, H, W) = get_stim_to_grid_mapping(distance, p_target)
    mapping = mapping.to(device)
    
    circuit = generate_surface_code_circuit(distance, p_start)
    sampler = circuit.compile_detector_sampler()
    
    ancilla_mask = checkerboard_ancilla_mask(distance)
    data_qubit_mask = synthetic_data_qubit_mask(distance, ancilla_mask)
    logical_masks = synthetic_logical_masks(distance, data_qubit_mask)
    logical_masks = logical_masks[1:2]

    # 2. Initialize Model
    hidden_dim = 64
    model = SurfaceCascade(
        distance=distance,
        rounds=T,
        hidden_dim=hidden_dim,
        depth=distance,
        data_qubit_mask=data_qubit_mask,
        logical_masks=logical_masks,
    ).to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    # Initialize EMA after DDP so it tracks the DDP parameters correctly
    ema = EMA(model, decay=0.9998)
    
    ancilla_mask = ancilla_mask.to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize Hybrid Optimizers
    muon_params = []
    adamw_params = []
    for p in model.parameters():
        if p.requires_grad:
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
                
    base_dim = 32
    width_ratio = base_dim / hidden_dim
    optimizer_muon = Muon(muon_params, lr=lr * 10, weight_decay=3e-3)
    optimizer_adamw = optim.AdamW(adamw_params, lr=lr * width_ratio, weight_decay=3e-3)

    # 3. Training Loop
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    arr_loss = []
    
    warmup_steps = int(total_steps * 0.02)
    anneal_steps = int(total_steps * 0.08)
    current_p = p_start
    
    iterator = range(total_steps)
    if rank == 0:
        iterator = tqdm(iterator)
        
    for step in iterator:
        if step <= warmup_steps:
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
                
        # Generate independent data on each GPU
        # (Since it relies on system entropy, it will naturally differ per process)
        detectors, observables = sampler.sample(shots=local_batch_size, separate_observables=True)
        inputs = torch.from_numpy(detectors.astype(np.float32)).to(device)
        targets = torch.from_numpy(observables.astype(np.float32)).to(device)
        
        batch_sz = inputs.size(0)
        detections = torch.zeros(batch_sz, T, H, W, device=device, dtype=torch.bool)
        
        t_idx = mapping[:, 0]
        y_idx = mapping[:, 1]
        x_idx = mapping[:, 2]
        detections[:, t_idx, y_idx, x_idx] = inputs > 0
        
        syndrome_idx = syndrome_indices_from_detections(detections, ancilla_mask)
        syndrome_idx = syndrome_idx.contiguous()
        syndrome_idx = torch.clamp(syndrome_idx, 0, 2)
        
        optimizer_muon.zero_grad()
        optimizer_adamw.zero_grad()
        
        outputs = model(syndrome_idx)
        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer_muon.step()
        optimizer_adamw.step()
        
        ema.update()
        
        # Accumulate metrics
        loss_tensor = loss.detach()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        running_loss += loss_tensor.item()
        
        with torch.no_grad():
            predictions = (outputs > 0).float()
            corrects = (predictions == targets).sum()
            dist.all_reduce(corrects, op=dist.ReduceOp.SUM)
            global_batch_acc = corrects.item() / (targets.numel() * world_size)
            running_acc += global_batch_acc 
            
        if (step + 1) % 100 == 0:
            avg_loss = running_loss / 100
            avg_acc = running_acc / 100
            if rank == 0:
                tqdm.write(f"Step [{step+1}/{total_steps}] - p: {current_p:.5f} - Loss: {avg_loss:.6f} - Acc: {avg_acc:.4f}")
                arr_loss.append(avg_loss)
            running_loss = 0.0
            running_acc = 0.0
            
    # Save Model and Plot (Only on master process)
    if rank == 0:
        # --- EMA BatchNorm Sync ---
        print("Syncing EMA weights and BatchNorm statistics...")
        ema.copy_to(model)
        model.train()
        with torch.no_grad():
            for _ in range(50):
                detectors, _ = sampler.sample(shots=local_batch_size, separate_observables=True)
                inputs = torch.from_numpy(detectors.astype(np.float32)).to(device)
                detections_sync = torch.zeros(inputs.size(0), T, H, W, device=device, dtype=torch.bool)
                detections_sync[:, t_idx, y_idx, x_idx] = inputs > 0
                synd_sync = syndrome_indices_from_detections(detections_sync, ancilla_mask)
                synd_sync = synd_sync.contiguous()
                synd_sync = torch.clamp(synd_sync, 0, 2)
                _ = model(synd_sync)
        
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, f"cascade_d{distance}_H{hidden_dim}_ddp_ema.pth")
        
        torch.save(model.module.state_dict(), save_path)
        print(f"Training complete! EMA DDP Model saved to {save_path}")
        
        plot_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        x_steps = [i * 100 for i in range(len(arr_loss))]
        ax.plot(x_steps, arr_loss, label=f"d={distance}", marker="o", markersize=3)
        ax.set_title(f"Loss vs steps (DDP)")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("BCE Loss logits")
        ax.grid(True, which="both", linestyle='--', alpha=0.7)
        ax.legend()
        
        plot_path = os.path.join(plot_dir, f"loss_plot_d{distance}_H{hidden_dim}_ddp.png")
        plt.savefig(plot_path)
        print(f"plot saved to {plot_path}")
        
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("No GPUs detected!")
        sys.exit(1)
        
    TARGET_DISTANCE = 11
    P_START = 0.001
    P_TARGET = 0.01
    TOTAL_STEPS = 20000
    GLOBAL_BATCH_SIZE = 3328
    
    print(f"Launching DDP across {world_size} GPUs...")
    mp.spawn(
        train_worker,
        args=(world_size, TARGET_DISTANCE, P_START, P_TARGET, TOTAL_STEPS, GLOBAL_BATCH_SIZE, 1e-3),
        nprocs=world_size,
        join=True
    )
