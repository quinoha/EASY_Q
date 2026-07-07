import stim
import json

d = 5
circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, after_clifford_depolarization=0.01)
coords = circuit.get_detector_coordinates()

# Get some statistics on coordinates
print(f"Num detectors: {len(coords)}")
x_vals = set()
y_vals = set()
t_vals = set()

for d_idx, (x, y, t) in coords.items():
    x_vals.add(x)
    y_vals.add(y)
    t_vals.add(t)

print(f"X range: {sorted(list(x_vals))}")
print(f"Y range: {sorted(list(y_vals))}")
print(f"T range: {sorted(list(t_vals))}")
