import sinter
from typing import Callable, List
import multiprocessing

def run_simulations(
    circuit_generator: Callable,
    distances: List[int],
    physical_error_rates: List[float],
    max_shots: int = 10_000,
    max_errors: int = 100
) -> List[sinter.TaskStats]:
    """
    Run quantum error correction simulations using sinter.
    """
    tasks = []
    for d in distances:
        for p in physical_error_rates:
            circuit = circuit_generator(distance=d, physical_error_rate=p)
            tasks.append(
                sinter.Task(
                    circuit=circuit,
                    json_metadata={'d': d, 'p': p}
                )
            )

    print(f"Starting {len(tasks)} simulation tasks using pymatching...")
    num_workers = max(1, multiprocessing.cpu_count() // 2)
    
    stats = sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True
    )
    return stats
