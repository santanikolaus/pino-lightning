"""Adaptation client: compose config, load operator, build_splits target-Re data, probe."""
from .. import setup
from .adapt import (build, build_splits, describe, load_config, main, make_save_fn,
                    run_name, weights_dir, weights_plan)

__all__ = ["setup", "build", "build_splits", "describe", "load_config", "main",
           "make_save_fn", "run_name", "weights_dir", "weights_plan"]
