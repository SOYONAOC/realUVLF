"""Compatibility imports for the random-q experiment; implementation lives in auroralf."""

from auroralf.experiments.random_q import (
    Config,
    burst_light,
    draw_logq,
    first_crossing,
    initialize_worker,
    one_mass,
)

__all__ = ["Config", "burst_light", "draw_logq", "first_crossing", "initialize_worker", "one_mass"]
