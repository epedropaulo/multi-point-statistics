"""
Patches archive for multi-point-statistics project.

This module contains patches for various libraries and functions that have
deprecated methods or compatibility issues.

Available patches:
- pyvista_patches: Fixes for deprecated PyVista methods in mpslib plotting
- numpy_patches: Fixes for deprecated NumPy methods in mpslib
"""

from .pyvista_patches import (
    patch_mpslib_plotting,
    patch_mpslib_plot_function,
    apply_all_pyvista_patches,
    numpy_to_pvgrid_fixed,
    plot_3d_reals_fixed
)

from .numpy_patches import (
    patch_mpslib_numpy,
    patch_numpy_nan_globally,
    apply_all_numpy_patches
)

__all__ = [
    'patch_mpslib_plotting',
    'patch_mpslib_plot_function',
    'apply_all_pyvista_patches',
    'numpy_to_pvgrid_fixed', 
    'plot_3d_reals_fixed',
    'patch_mpslib_numpy',
    'patch_numpy_nan_globally',
    'apply_all_numpy_patches'
]
