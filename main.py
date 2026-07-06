import os
from circuits.surface_code import generate_surface_code_circuit
from simulation.runner import run_simulations
from plotting.plot_threshold import draw_and_save_plot
import sinter

def main():
    # 1. Define simulation parameters
    distances = [3, 5, 7]
    p_rates = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3]
    
    # 2. Run simulation to get statistics
    stats = run_simulations(
        circuit_generator=generate_surface_code_circuit,
        distances=distances,
        physical_error_rates=p_rates,
        max_shots=1000, # Reduced max_shots for faster testing. Increase it for more accurate graphs.
        max_errors=100
    )
    
    # Optionally save raw stats data here
    # with open("data/raw_stats.csv", "w") as f:
    #     for stat in stats:
    #         # save logic here if needed
    #         pass
    
    # 3. Draw and save plot
    os.makedirs("plotting/output", exist_ok=True)
    draw_and_save_plot(stats, save_path="plotting/output/threshold.png")

if __name__ == "__main__":
    main()
