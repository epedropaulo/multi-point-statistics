# Patches Archive

This directory contains patches for various libraries and functions that have deprecated methods or compatibility issues.

## Available Patches

### PyVista Patches (`pyvista_patches.py`)

Fixes for deprecated PyVista methods used in mpslib plotting functions.

**Issues Fixed:**
- Replaces deprecated `pyvista.UniformGrid` with `pyvista.ImageData`
- Updates `mpslib.plot.numpy_to_pvgrid` function
- Updates `mpslib.plot.plot_3d_reals` function
- Updates `mpslib.plot.plot` function (the main plotting function used in notebooks)

### NumPy Patches (`numpy_patches.py`)

Fixes for deprecated NumPy methods used in mpslib functions.

**Issues Fixed:**
- Replaces deprecated `np.NaN` with `np.nan` (NumPy 2.0 compatibility)
- Patches `mpslib.mpslib` module initialization
- Provides global numpy compatibility fixes

**Usage:**

```python
# Import and apply all patches
from scripts.patches import apply_all_pyvista_patches, apply_all_numpy_patches

# Apply all PyVista patches
apply_all_pyvista_patches()

# Apply all NumPy patches
apply_all_numpy_patches()

# Or apply individual patches
from scripts.patches import (
    patch_mpslib_plotting, 
    patch_mpslib_plot_function,
    patch_mpslib_numpy,
    patch_numpy_nan_globally
)

# Apply specific patches
patch_mpslib_plotting()  # Fixes numpy_to_pvgrid and plot_3d_reals
patch_mpslib_plot_function()  # Fixes the main plot.plot() function
patch_mpslib_numpy()  # Fixes np.NaN issues in mpslib.mpslib
patch_numpy_nan_globally()  # Global numpy compatibility fix
```

**Example in Notebook:**

```python
# At the beginning of your notebook, before importing mpslib
from scripts.patches import apply_all_pyvista_patches, apply_all_numpy_patches

# Apply all patches
apply_all_pyvista_patches()  # Fixes PyVista plotting issues
apply_all_numpy_patches()    # Fixes NumPy compatibility issues

# Now you can safely use mpslib functions
import mpslib as mps
import numpy as np

# This will now work without deprecation warnings
O = mps.mpslib()  # No more np.NaN errors
mps.plot.plot(your_data, slice=0)  # No more PyVista deprecation warnings
```

## Migration from utils.py

The patches that were previously in `scripts/utils.py` have been moved to this dedicated patches archive for better organization. The functions are still available through the utils module for backward compatibility:

```python
# This still works (imports from patches archive)
from scripts.utils import patch_mpslib_plotting, numpy_to_pvgrid_fixed

# But it's recommended to use the patches archive directly
from scripts.patches import apply_all_pyvista_patches
```

## Adding New Patches

To add new patches:

1. Create a new module in this directory (e.g., `new_library_patches.py`)
2. Add the imports to `__init__.py`
3. Update this README with documentation
4. Follow the same pattern as the existing patches

## Dependencies

- `numpy`
- `pyvista` (for PyVista patches)
- `mpslib` (for mpslib patches)
