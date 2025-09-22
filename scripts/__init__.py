"""
Scripts module for multi-point statistics project.

This module contains utility functions and submodules for data processing,
analysis, and visualization.
"""

# Import metrics module to make it available
try:
    from . import metrics
    __all__ = ['metrics']
except ImportError as e:
    # If metrics module can't be imported due to missing dependencies,
    # we'll handle it gracefully
    __all__ = []
    _import_error = e
