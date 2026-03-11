"""Пакет для сравнения доминирующих частот FFT и SSA."""

from .analysis import analyze_files, dominant_ssa_frequency, read_signal

__all__ = ["analyze_files", "dominant_ssa_frequency", "read_signal"]
