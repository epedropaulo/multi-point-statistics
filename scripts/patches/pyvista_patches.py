"""
PyVista compatibility patches for mpslib plotting functions.

This module provides fixes for deprecated PyVista methods used in mpslib,
specifically replacing the deprecated UniformGrid with ImageData.
"""

import numpy as np


def numpy_to_pvgrid_fixed(Data, origin=(0, 0, 0), spacing=(1, 1, 1)):
    """
    Convert 3D numpy array to pyvista ImageData (replaces deprecated UniformGrid)
    
    Args:
        Data: 3D numpy array
        origin: origin point (x, y, z)
        spacing: spacing between grid points (dx, dy, dz)
        
    Returns:
        pyvista.ImageData object
    """
    try:
        import pyvista as pv
        
        # Create the spatial reference using ImageData instead of UniformGrid
        grid = pv.ImageData()
        
        # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
        grid.dimensions = np.array(Data.shape) + 1
        
        # Edit the spatial reference
        grid.origin = origin  # The bottom left corner of the data set
        grid.spacing = spacing  # These are the cell sizes along each axis
        
        # Add the data values to the cell data
        grid.cell_data['values'] = Data.flatten(order='F')  # Flatten the array!
        
        return grid
        
    except ImportError:
        print("PyVista is not installed. Install it with: pip install pyvista")
        return None


def plot_3d_reals_fixed(O, nshow=4, slice=0):
    """
    Plot realizations in O.sim in 3D using pyvista (fixed version)
    
    Parameters
    ----------
    O : mpslib object
        
    nshow : int (def=4)
        show a maximum of 'nshow' realizations
    slice : int (def=0)
        0 = show full 3D volume, 1 = show slices
    """
    import numpy as np
    import pyvista as pv
    
    if not(hasattr(O, 'sim')):
        print('No data to plot (no "sim" attribute)')
        return -1
    if (O.sim is None):
        print('No data to plot ("sim" attribute is "None")')
        return -1
    
    nr = O.par['n_real']
    nshow = np.min((nshow, nr))
    
    nxy = np.ceil(np.sqrt(nshow)).astype('int')
    
    plotter = pv.Plotter(shape=(nxy, nxy))

    i = -1
    for ix in range(nxy):
        for iy in range(nxy):
            i = i + 1
            if i >= nshow:
                break
                
            plotter.subplot(iy, ix)

            Data = O.sim[i]
            # Use the fixed function instead of UniformGrid
            grid = numpy_to_pvgrid_fixed(Data, origin=O.par['origin'], spacing=O.par['grid_cell_size'])

            if grid is None:
                print(f"❌ Failed to create grid for realization {i}")
                continue

            if (slice == 0):
                # Show full 3D volume instead of slices
                plotter.add_mesh(grid, opacity=0.7, show_edges=False)
            else:
                # Show slices if specifically requested
                plotter.add_mesh(grid.slice_orthogonal())
                
            plotter.add_text('#%d' % (i + 1))
            
            # Configure grid and axes
            plotter.show_grid()
            plotter.show_axes()
            
            # Set background for better visibility
            plotter.set_background('white')

    plotter.show()


def patch_mpslib_plotting():
    """
    Patch MPSlib plotting functions to use the fixed PyVista ImageData instead of deprecated UniformGrid.
    This function should be called before using any MPSlib plotting functions.
    
    This patch fixes the following issues:
    - Replaces deprecated pyvista.UniformGrid with pyvista.ImageData
    - Updates mpslib.plot.numpy_to_pvgrid function
    - Updates mpslib.plot.plot_3d_reals function
    """
    try:
        import mpslib.plot as mps_plot
        
        # Replace the deprecated numpy_to_pvgrid function
        mps_plot.numpy_to_pvgrid = numpy_to_pvgrid_fixed
        
        # Replace the plot_3d_reals function
        mps_plot.plot_3d_reals = plot_3d_reals_fixed
        
        print("✅ MPSlib plotting functions patched to use PyVista ImageData instead of deprecated UniformGrid")
        
    except ImportError:
        print("❌ Could not patch MPSlib plotting - mpslib.plot module not found")
    except Exception as e:
        print(f"❌ Error patching MPSlib plotting: {e}")


def patch_mpslib_plot_function():
    """
    Patch the main mps.plot.plot() function to use the fixed PyVista methods.
    
    This is a more comprehensive patch that also fixes the main plot function
    that is commonly used in notebooks.
    """
    try:
        import mpslib.plot as mps_plot
        
        # Store the original plot function
        def plot_fixed(data, slice=0, header=None, **kwargs):
            """
            Fixed version of mps.plot.plot() that uses ImageData instead of UniformGrid
            """
            import pyvista as pv
            
            # Ensure data is 3D
            if len(data.shape) == 2:
                data = data.reshape(data.shape[0], data.shape[1], 1)
            
            # Create grid using fixed function
            grid = numpy_to_pvgrid_fixed(data)
            
            if grid is None:
                print("❌ Failed to create grid - PyVista not available")
                return
            
            # Create plotter
            plotter = pv.Plotter()
            
            # Add mesh based on slice parameter
            if slice == 0:
                # Show full 3D volume instead of slices
                plotter.add_mesh(grid, opacity=1, show_edges=False)
            else:
                # Show slices if specifically requested
                mesh = grid.slice_orthogonal()
                plotter.add_mesh(mesh)
            
            # Add header if provided
            if header:
                plotter.add_text(header)
            
            # Configure plotter
            plotter.show_grid()
            plotter.show_axes()
            plotter.set_background('white')
            
            # Show the plot
            plotter.show()
        
        # Replace the plot function
        mps_plot.plot = plot_fixed
        
        print("✅ MPSlib plot.plot() function patched to use PyVista ImageData")
        
    except ImportError:
        print("❌ Could not patch MPSlib plot function - mpslib.plot module not found")
    except Exception as e:
        print(f"❌ Error patching MPSlib plot function: {e}")


def apply_all_pyvista_patches():
    """
    Apply all PyVista-related patches to mpslib.
    
    This is the main function to call to fix all PyVista compatibility issues.
    """
    print("🔧 Applying PyVista compatibility patches...")
    
    # Apply the main plotting patches
    patch_mpslib_plotting()
    
    # Apply the plot function patch
    patch_mpslib_plot_function()
    
    print("✅ All PyVista patches applied successfully!")
