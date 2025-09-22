"""
Metrics module for computing and displaying pore structure metrics.

This module provides functions to compute various pore structure metrics
using poregen.features for simulated data and target data.
"""

# Import functions with error handling for missing dependencies
try:
    from .metrics_display import compute_and_display_metrics, load_simulation_data
    __all__ = ['compute_and_display_metrics', 'load_simulation_data']
except ImportError as e:
    # If dependencies are missing, we'll handle it gracefully
    __all__ = []
    _import_error = e
