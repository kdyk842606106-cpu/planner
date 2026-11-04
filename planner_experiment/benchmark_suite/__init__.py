"""Reproducible three-engine experiments (separate from the historical CLI)."""

SUITE_VERSION = "three-engine/v1"
SCALES = (8, 16, 32, 64, 128, 256)
GROUPS = tuple(f"C{i}" for i in range(7))
GENERATOR_SEEDS = (101, 202, 303)
RUN_SEEDS = (11, 23, 37)
BUDGETS = (1.0, 3.0, 10.0)
ENGINES = ("LEGACY", "ASTAR", "GA")
