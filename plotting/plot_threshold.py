import sinter
import matplotlib.pyplot as plt
from typing import List

def draw_and_save_plot(stats: List[sinter.TaskStats], save_path: str = None):
    """
    Draw a threshold graph (Logical Error Rate vs Physical Error Rate) and optionally save it.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata['p'],
        group_func=lambda stat: f"d={stat.json_metadata['d']} ({stat.decoder})",
        plot_args_func=lambda index, group_key, group_stats: {
            'marker': 'o' if 'cascade' in group_key else 's',
            'linestyle': '-' if 'cascade' in group_key else '--'
        }
    )

    ax.set_title("Logical Error Rate vs Physical Error Rate")
    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical Error Rate")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    ax.legend()
    
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
