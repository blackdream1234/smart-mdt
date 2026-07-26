"""Deterministic research evaluation framework for Smart-MDT."""

from .config import EvaluationConfig
from .io import BenchmarkData, EvaluationDataError, load_benchmark_folder

__all__ = [
    "BenchmarkData",
    "EvaluationConfig",
    "EvaluationDataError",
    "load_benchmark_folder",
]

__version__ = "1.0.0"
