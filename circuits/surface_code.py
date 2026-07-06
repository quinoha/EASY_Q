import stim

def generate_surface_code_circuit(distance: int, physical_error_rate: float) -> stim.Circuit:
    """
    Generates a rotated surface code memory circuit with given distance and physical error rate.
    Uses stim's built-in circuit generator.
    """
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=physical_error_rate,
        after_reset_flip_probability=physical_error_rate,
        before_measure_flip_probability=physical_error_rate,
        before_round_data_depolarization=physical_error_rate
    )
