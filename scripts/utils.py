import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from scipy.ndimage import zoom

# PyVista compatibility function for deprecated UniformGrid
def numpy_to_pvgrid_fixed(Data, origin=(0,0,0), spacing=(1,1,1)):
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

def patch_mpslib_plotting():
    """
    Patch MPSlib plotting functions to use the fixed PyVista ImageData instead of deprecated UniformGrid.
    This function should be called before using any MPSlib plotting functions.
    """
    try:
        import mpslib.plot as mps_plot
        
        # Replace the deprecated numpy_to_pvgrid function
        mps_plot.numpy_to_pvgrid = numpy_to_pvgrid_fixed
        
        # Also patch the plot_3d_reals function that uses UniformGrid directly
        def plot_3d_reals_fixed(O, nshow=4, slice=0):
            '''Plot realizations in in O.sim in 3D using pyvista (fixed version)
            
            Paramaters
            ----------
            O : mpslib object
                
            nshow : int (def=4)
                show a maxmimum of 'nshow' realizations
            '''
            import numpy as np
            import pyvista as pv
            
            if not(hasattr(O,'sim')):
                print('No data to plot (no "sim" attribute)')
                return -1
            if (O.sim is None):
                print('No data to plot ("sim" attribute i "None")')
                return -1
            
            nr = O.par['n_real']
            nshow = np.min((nshow,nr))
            
            nxy = np.ceil(np.sqrt(nshow)).astype('int')
            
            plotter = pv.Plotter(shape=(nxy,nxy))

            i=-1
            for ix in range(nxy):
                for iy in range(nxy):
                    i=i+1
                    plotter.subplot(iy,ix)

                    Data = O.sim[i]
                    # Use the fixed function instead of UniformGrid
                    grid = numpy_to_pvgrid_fixed(Data, origin=O.par['origin'], spacing=O.par['grid_cell_size'])

                    if (slice==0):
                        plotter.add_mesh(grid.slice_orthogonal())
                    else:
                        plotter.add_mesh(grid.slice(normal=[1, 1, 0]))
                    plotter.add_text('#%d' % (i+1))
                    
                    # Configure grid and axes (simplified to avoid kernel crashes)
                    plotter.show_grid()
                    
                    # Set background for better visibility
                    plotter.set_background('white')

            plotter.show()
        
        # Replace the plot_3d_reals function
        mps_plot.plot_3d_reals = plot_3d_reals_fixed
        
        print("✅ MPSlib plotting functions patched to use PyVista ImageData instead of deprecated UniformGrid")
        
    except ImportError:
        print("❌ Could not patch MPSlib plotting - mpslib.plot module not found")
    except Exception as e:
        print(f"❌ Error patching MPSlib plotting: {e}")

def plot_3d_realizations_enhanced(O, n_realizations=4, slice_mode='full', 
                                 cmap='viridis', opacity=1.0, show_edges=False):
    """
    Enhanced 3D plotting function for MPSlib realizations using fixed PyVista ImageData.
    
    Args:
        O: MPSlib object containing simulation results
        n_realizations: Number of realizations to plot (default: 4)
        slice_mode: 'orthogonal', 'single', or 'full' (default: 'full')
        cmap: Colormap for visualization (default: 'viridis')
        opacity: Opacity of the mesh (default: 1.0)
        show_edges: Whether to show mesh edges (default: False)
    """
    try:
        import pyvista as pv
        
        # Get number of available realizations
        n_available = len(O.sim)
        n_to_plot = min(n_realizations, n_available)
        
        if n_to_plot == 0:
            print("No realizations found in O.sim")
            return
        
        # Calculate grid layout
        nxy = int(np.ceil(np.sqrt(n_to_plot)))
        
        # Create plotter
        plotter = pv.Plotter(shape=(nxy, nxy))
        
        i = 0
        for ix in range(nxy):
            for iy in range(nxy):
                if i >= n_to_plot:
                    break
                    
                plotter.subplot(iy, ix)
                
                # Get realization data
                Data = O.sim[i]
                
                # Create grid using fixed function
                grid = numpy_to_pvgrid_fixed(
                    Data, 
                    origin=O.par.get('origin', (0, 0, 0)), 
                    spacing=O.par.get('grid_cell_size', (1, 1, 1))
                )
                
                if grid is None:
                    print(f"❌ Failed to create grid for realization {i}")
                    continue
                
                # Add mesh based on slice mode
                if slice_mode == 'orthogonal':
                    mesh = grid.slice_orthogonal()
                elif slice_mode == 'single':
                    mesh = grid.slice(normal=[1, 1, 0])
                elif slice_mode == 'full':
                    mesh = grid
                else:
                    mesh = grid.slice_orthogonal()
                
                plotter.add_mesh(mesh, cmap=cmap, opacity=opacity, show_edges=show_edges)
                plotter.add_text(f'Realization {i+1}', font_size=12)
                
                # Configure grid and axes (simplified to avoid kernel crashes)
                plotter.show_grid()
                plotter.show_axes()
                
                # Set background for better visibility
                plotter.set_background('white')
                
                i += 1
        
        plotter.show()
        
    except ImportError:
        print("❌ PyVista is not installed. Install it with: pip install pyvista")
    except Exception as e:
        print(f"❌ Error in 3D plotting: {e}")

def load_binary_from_eleven_sandstones(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        unshaped_voxel = np.fromfile(f, dtype=np.uint8)

    size = int(round(unshaped_voxel.shape[0] ** (1 / 3)))
    if size**3 != unshaped_voxel.shape[0]:
        raise ValueError('Voxel is not a cube.')

    return unshaped_voxel.reshape((size, size, size))

def npy_to_hard_data(npy_path: str, threshold=0.5, max_points=None, downsample_factor=1, use_3d=True) -> np.ndarray:
    """
    Convert a .npy array to hard data structure format [X, Y, Z, VALUE].
    Now supports full 3D data processing.
    
    Args:
        npy_path (str): Path to the .npy file
        threshold (float): Threshold to convert continuous values to binary (default: 0.5)
        max_points (int): Maximum number of points to return (default: None = all points)
        downsample_factor (int): Factor to scale coordinates back to original system (default: 1)
        use_3d (bool): If True, use all Z layers for 3D arrays. If False, use only first Z layer (default: True)
        
    Returns:
        np.ndarray: Hard data array in format [X, Y, Z, VALUE]
        
    Example:
        # Load 3D array and convert to hard data with all Z layers
        hard_data = npy_to_hard_data('my_3d_array.npy', threshold=0.5, downsample_factor=4, use_3d=True)
        
        # Load 3D array but use only first Z layer (2D-like behavior)
        hard_data_2d = npy_to_hard_data('my_3d_array.npy', threshold=0.5, downsample_factor=4, use_3d=False)
        
        # Use in MPSlib
        O.d_hard = hard_data
    """
    # Load the array
    array = np.load(npy_path)
    
    # Get array dimensions
    if len(array.shape) == 2:
        # 2D array - add Z dimension
        nx, ny = array.shape
        nz = 1
        array_3d = array.reshape(nx, ny, nz)
        print(f"📊 2D array detected, added Z dimension")
    elif len(array.shape) == 3:
        nx, ny, nz_full = array.shape
        if use_3d:
            # 3D array - use all Z layers
            nz = nz_full
            array_3d = array  # Use the full 3D array
            print(f"📊 3D array detected, using all {nz} Z layers")
        else:
            # 3D array - use only the first layer of Z axis (legacy behavior)
            nz = 1
            array_3d = array[:, :, 0:1]  # Take only the first Z layer
            print(f"📊 3D array detected, using only first Z layer (2D mode)")
    else:
        raise ValueError(f"Array must be 2D or 3D, got shape {array.shape}")
    
    # Create coordinate grids
    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(nx), 
        np.arange(ny), 
        np.arange(nz), 
        indexing='ij'
    )
    
    # Flatten coordinates and values
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    z_flat = z_coords.flatten()
    values_flat = array_3d.flatten()
    
    # Scale coordinates by downsample factor
    x_flat = x_flat * downsample_factor
    y_flat = y_flat * downsample_factor
    z_flat = z_flat * downsample_factor
    
    # Apply threshold if needed (convert to binary)
    if threshold is not None:
        values_flat = (values_flat > threshold).astype(float)
    
    # Create hard data array
    hard_data = np.column_stack([x_flat, y_flat, z_flat, values_flat])
    
    # Filter out zero values (optional - comment out if you want all points)
    # hard_data = hard_data[hard_data[:, 3] != 0]
    
    # Limit number of points if specified
    if max_points is not None and len(hard_data) > max_points:
        # Randomly sample points
        indices = np.random.choice(len(hard_data), max_points, replace=False)
        hard_data = hard_data[indices]
    
    print(f"✅ Converted {npy_path} to hard data format")
    print(f"   Original array shape: {array.shape}")
    print(f"   Processed array shape: {array_3d.shape}")
    print(f"   Hard data points: {len(hard_data)}")
    print(f"   Value range: [{hard_data[:, 3].min():.2f}, {hard_data[:, 3].max():.2f}]")
    print(f"   Coordinate scaling factor: {downsample_factor}")
    print(f"   Scaled coordinate range: X[0, {hard_data[:, 0].max()}], Y[0, {hard_data[:, 1].max()}], Z[0, {hard_data[:, 2].max()}]")
    
    return hard_data

def plot_realizations_enhanced(O, n_realizations=4, figsize=None, cmap='viridis', 
                               title_prefix="Realization", save_path=None, dpi=150):
    """
    Enhanced plotting function to replace O.plot_reals() with better visualization.
    
    Args:
        O: MPSlib object containing simulation results
        n_realizations (int): Number of realizations to plot (default: 4)
        figsize (tuple): Figure size (width, height)
        cmap (str): Colormap for visualization
        title_prefix (str): Prefix for plot titles
        save_path (str): Path to save the plot (optional)
        dpi (int): DPI for saved image
    """
    
    # Get number of available realizations
    n_available = len(O.sim)
    n_to_plot = min(n_realizations, n_available)
    
    if n_to_plot == 0:
        print("No realizations found in O.sim")
        return
    
    # Calculate grid layout
    cols = min(3, n_to_plot)  # Max 3 columns
    rows = (n_to_plot + cols - 1) // cols  # Calculate needed rows
    
    # Create figure with better layout
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.3, hspace=0.4)
    
    # Plot each realization
    for i in range(n_to_plot):
        row = i // cols
        col = i % cols
        
        ax = fig.add_subplot(gs[row, col])
        
        # Get realization data
        real_data = O.sim[i]
        
        # Handle different data shapes
        if len(real_data.shape) == 3:
            # 3D data - take first slice
            plot_data = real_data[:, :, 0]
        elif len(real_data.shape) == 2:
            # 2D data
            plot_data = real_data
        else:
            # 1D data - reshape if possible
            grid_size = O.par['simulation_grid_size']
            if len(grid_size) >= 2:
                plot_data = real_data.reshape(grid_size[0], grid_size[1])
            else:
                plot_data = real_data.reshape(-1, 1)
        
        # Create the plot
        im = ax.imshow(plot_data.T, cmap=cmap, aspect='equal', origin='lower')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Value', fontsize=10)
        
        # Set title and labels with shape information
        shape_str = f"({plot_data.shape[0]}×{plot_data.shape[1]})"
        ax.set_title(f'{title_prefix} {i+1} - Shape: {shape_str}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Calculate porosity (fraction of non-zero values)
        porosity = np.sum(plot_data > 0) / plot_data.size * 100
        
        # Add statistics in text box
        stats_text = f'Min: {plot_data.min():.2f}\nMax: {plot_data.max():.2f}\nMean: {plot_data.mean():.2f}\nPorosity: {porosity:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add overall title
    fig.suptitle(f'MPSlib Simulation Results ({n_to_plot} realizations)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Add simulation info
    info_text = f'Grid size: {O.par["simulation_grid_size"]}\nMethod: {O.method}\nn_cond: {O.par["n_cond"]}'
    fig.text(0.02, 0.02, info_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # return fig

def plot_two_point_correlation_comparison(O, target_image, max_lag=None, figsize=None, 
                                         save_path=None, dpi=150):
    """
    Plot two-point correlation functions for all realizations compared to target image.
    
    Args:
        O: MPSlib object containing simulation results
        target_image: numpy array of the target/reference image
        max_lag: maximum lag distance to compute (default: min dimension / 4)
        figsize: figure size (width, height)
        save_path: path to save the plot (optional)
        dpi: DPI for saved image
    """
    
    def compute_two_point_correlation(image, max_lag):
        """Compute two-point correlation function for a 2D image."""
        if len(image.shape) == 3:
            image = image[:, :, 0]  # Take first slice if 3D
        
        # Ensure binary image
        binary_image = (image > 0.5).astype(float)
        
        # Get image dimensions
        nx, ny = binary_image.shape
        max_lag = min(max_lag or min(nx, ny) // 4, min(nx, ny) // 4)
        
        # Compute correlation function
        correlations = []
        lags = []
        
        for lag in range(1, max_lag + 1):
            corr_values = []
            
            # Horizontal correlation
            if lag < nx:
                corr_h = np.corrcoef(binary_image[:-lag, :].flatten(), 
                                   binary_image[lag:, :].flatten())[0, 1]
                if not np.isnan(corr_h):
                    corr_values.append(corr_h)
            
            # Vertical correlation
            if lag < ny:
                corr_v = np.corrcoef(binary_image[:, :-lag].flatten(), 
                                   binary_image[:, lag:].flatten())[0, 1]
                if not np.isnan(corr_v):
                    corr_values.append(corr_v)
            
            # Average the correlations for this lag
            if corr_values:
                correlations.append(np.mean(corr_values))
                lags.append(lag)
        
        return np.array(lags), np.array(correlations)
    
    # Set max_lag if not provided
    if max_lag is None:
        max_lag = min(target_image.shape[0], target_image.shape[1]) // 4
    
    # Compute target correlation
    target_lags, target_corr = compute_two_point_correlation(target_image, max_lag)
    
    # Compute correlations for all realizations
    realization_correlations = []
    for i, real_data in enumerate(O.sim):
        lags, corr = compute_two_point_correlation(real_data, max_lag)
        realization_correlations.append(corr)
    
    # Convert to numpy array for easier computation
    realization_correlations = np.array(realization_correlations)
    
    # Compute median correlation
    median_corr = np.median(realization_correlations, axis=0)
    
    # Create smooth curves using spline interpolation
    def smooth_curve(x, y, num_points=200):
        """Smooth a curve using cubic spline interpolation."""
        if len(x) < 4:  # Need at least 4 points for cubic spline
            return x, y
        
        # Create interpolation function
        f = interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        # Create smooth x values
        x_smooth = np.linspace(x.min(), x.max(), num_points)
        y_smooth = f(x_smooth)
        
        return x_smooth, y_smooth
    
    # Create the plot
    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual realizations (smoothed)
    for i, corr in enumerate(realization_correlations):
        x_smooth, y_smooth = smooth_curve(target_lags, corr)
        ax.plot(x_smooth, y_smooth, 'b-', alpha=0.3, linewidth=1)
    
    # Plot median of realizations (smoothed)
    x_smooth_median, y_smooth_median = smooth_curve(target_lags, median_corr)
    ax.plot(x_smooth_median, y_smooth_median, 'b-', linewidth=3, label=f'Median ({len(O.sim)} realizations)')
    
    # Plot target correlation (smoothed)
    x_smooth_target, y_smooth_target = smooth_curve(target_lags, target_corr)
    ax.plot(x_smooth_target, y_smooth_target, 'r-', linewidth=3, label='Target')
    
    # Customize plot
    ax.set_xlabel('Lag Distance')
    ax.set_ylabel('Two-Point Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Two-point correlation plot saved to: {save_path}")
    
    plt.show()


def plot_mse_comparison(
    O, 
    target_image,
    hard_data_image=None,
    n_realizations=3,
    figsize=None, 
    save_path=None,
):
    """
    Enhanced MSE comparison plot showing target, hard data, and multiple realizations with differences.
    Each row is plotted as a separate figure for better display.
    
    Args:
        O: MPSlib object containing simulation results
        target_image: numpy array of the target/reference image
        hard_data_image: numpy array of the hard data image (optional)
        n_realizations: number of realizations to plot (default: 3)
        figsize: figure size (width, height) for each individual plot
        save_path: path to save the plot (optional)
        dpi: DPI for saved image
    """

    # Limit number of realizations to available ones
    n_available = len(O.sim)
    n_to_plot = min(n_realizations, n_available)
    
    if n_to_plot == 0:
        print("No realizations found in O.sim")
        return
    
    # Set default figure size for individual plots
    if figsize is None:
        figsize = (12, 5)  # Good size for 2-column layout
    
    # Ensure target image is 2D
    if len(target_image.shape) == 3:
        target_2d = target_image[:, :, 0]
    else:
        target_2d = target_image
    
    # Calculate target porosity
    target_porosity = np.sum(target_2d > 0.5) / target_2d.size * 100
    
    # Plot 1: Target and Hard Data images
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig1.suptitle('Target vs Hard Data Images', fontsize=14, fontweight='bold')
    
    # Plot target image
    im1 = ax1.imshow(target_2d.T, cmap='viridis', aspect='equal', origin='lower')
    ax1.set_title(f'Target Image\nPorosity: {target_porosity:.1f}%', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, shrink=0.4)
    
    # Plot hard data image if provided
    if len(hard_data_image.shape) == 3:
        hard_data_2d = hard_data_image[:, :, 0]
    else:
        hard_data_2d = hard_data_image
    
    hard_data_porosity = np.sum(hard_data_2d > 0.5) / hard_data_2d.size * 100
    im2 = ax2.imshow(hard_data_2d.T, cmap='viridis', aspect='equal', origin='lower')
    ax2.set_title(f'Hard Data Image\nPorosity: {hard_data_porosity:.1f}%', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, shrink=0.4)

    plt.tight_layout()
    plt.show()
    
    # Subsequent plots: Realizations and their differences
    results = []
    for i in range(n_to_plot):
        # Get realization
        realization = O.sim[i]
        if len(realization.shape) == 3:
            real_2d = realization[:, :, 0]
        else:
            real_2d = realization

        # Calculate metrics
        mse_continuous = mean_squared_error(target_2d, real_2d)
        real_porosity = np.sum(real_2d > 0.5) / real_2d.size * 100
        diff = abs(target_2d - real_2d)
        
        # Calculate accuracy excluding hard data pixels
        hard_data_pixels = hard_data_2d.shape[0] * hard_data_2d.shape[1]
        total_pixels = real_2d.shape[0] * real_2d.shape[1]
        generated_pixels = total_pixels - hard_data_pixels
        accuracy_percentage = (1 - np.sum(diff) / generated_pixels) * 100
        
        # Create separate plot for this realization
        fig_real, (ax_real, ax_diff) = plt.subplots(1, 2, figsize=figsize)
        fig_real.suptitle(f'Realization {i} vs Target', fontsize=14, fontweight='bold')
        
        # Plot realization
        im_real = ax_real.imshow(real_2d.T, cmap='viridis', aspect='equal', origin='lower')
        ax_real.set_title(f'Realization {i}\nPorosity: {real_porosity:.1f}%', fontsize=12)
        ax_real.set_xlabel('X')
        ax_real.set_ylabel('Y')
        plt.colorbar(im_real, ax=ax_real, shrink=0.4)
        
        # Plot difference
        im_diff = ax_diff.imshow(diff.T, cmap='viridis', aspect='equal', origin='lower')
        ax_diff.set_title(f'Difference (Target - Realization {i})\nMSE: {mse_continuous:.4f}\nAccuracy: {accuracy_percentage:.1f}%', fontsize=12)
        ax_diff.set_xlabel('X')
        ax_diff.set_ylabel('Y')
        plt.colorbar(im_diff, ax=ax_diff, shrink=0.4)
        
        plt.tight_layout()
        plt.show()
        
        # Store results
        results.append({
            'realization_idx': i,
            'mse_continuous': mse_continuous,
            'realization_porosity': real_porosity,
            'porosity_difference': abs(target_porosity - real_porosity),
            'accuracy_percentage': accuracy_percentage,
            'wrong_pixels': np.sum(diff),
        })
    
    # Add summary statistics
    avg_mse = np.mean([r['mse_continuous'] for r in results])
    avg_accuracy = np.mean([r['accuracy_percentage'] for r in results])
    
    # Create summary plot
    fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
    fig_summary.suptitle('Summary Statistics', fontsize=14, fontweight='bold')
    
    # Add hard data info to statistics
    wrong_pixels_text = '\n'.join([f'Realization {r["realization_idx"]}: {r["wrong_pixels"]:.0f} wrong pixels' for r in results])
    stats_text = f'Average MSE: {avg_mse:.4f}\nAverage Accuracy (generated pixels): {avg_accuracy:.1f}%\nTarget Porosity: {target_porosity:.1f}%\nHard Data Pixels: {hard_data_pixels:,}\nGenerated Pixels: {generated_pixels:,}\n\nWrong Pixels per Realization:\n{wrong_pixels_text}'

    ax_summary.text(0.1, 0.5, stats_text, fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.4),
                   verticalalignment='center', transform=ax_summary.transAxes)
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)
    ax_summary.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    if save_path:
        print(f"Note: Each plot is displayed separately. To save, use individual save_path for each plot.")
    
    return {
        'total_pixels': total_pixels,
        'generated_pixels': generated_pixels,
        'target_porosity': target_porosity,
        'average_mse': avg_mse,
        'average_accuracy': avg_accuracy,
        'realization_results': results
    }

def extract_subcube(cube_3d, subcube_length, cube_index):
    """
    Extract a non-overlapping sub-cube from a 3D array.
    
    Args:
        cube_3d (numpy.ndarray): 3D input array
        subcube_length (int): Length of each side of the sub-cube
        cube_index (int): Index of the sub-cube to extract (0, 1, 2, 3, ...)
        
    Returns:
        numpy.ndarray: Extracted sub-cube
        
    Example:
        # Extract 4 sub-cubes from a 100x100x100 cube
        original_cube = np.random.rand(100, 100, 100)
        
        # Get the first sub-cube (index 0)
        subcube_0 = extract_subcube(original_cube, 50, 0)  # Shape: (50, 50, 50)
        
        # Get the second sub-cube (index 1) 
        subcube_1 = extract_subcube(original_cube, 50, 1)  # Shape: (50, 50, 50)
        
        # Get the third sub-cube (index 2)
        subcube_2 = extract_subcube(original_cube, 50, 2)  # Shape: (50, 50, 50)
        
        # Get the fourth sub-cube (index 3)
        subcube_3 = extract_subcube(original_cube, 50, 3)  # Shape: (50, 50, 50)
    """
    if len(cube_3d.shape) != 3:
        raise ValueError(f"Input must be 3D array, got shape {cube_3d.shape}")
    
    nx, ny, nz = cube_3d.shape
    
    # Calculate how many sub-cubes fit in each dimension
    n_cubes_x = nx // subcube_length
    n_cubes_y = ny // subcube_length
    n_cubes_z = nz // subcube_length
    
    # Calculate total number of possible sub-cubes
    total_cubes = n_cubes_x * n_cubes_y * n_cubes_z
    
    if cube_index >= total_cubes:
        raise ValueError(f"Cube index {cube_index} is out of range. "
                        f"Only {total_cubes} sub-cubes available "
                        f"({n_cubes_x}x{n_cubes_y}x{n_cubes_z})")
    
    # Calculate the position of the requested sub-cube
    # Convert linear index to 3D coordinates
    cube_z = cube_index // (n_cubes_x * n_cubes_y)
    cube_y = (cube_index % (n_cubes_x * n_cubes_y)) // n_cubes_x
    cube_x = cube_index % n_cubes_x
    
    # Calculate start and end indices for each dimension
    start_x = cube_x * subcube_length
    end_x = start_x + subcube_length
    start_y = cube_y * subcube_length
    end_y = start_y + subcube_length
    start_z = cube_z * subcube_length
    end_z = start_z + subcube_length
    
    # Extract the sub-cube
    subcube = cube_3d[start_x:end_x, start_y:end_y, start_z:end_z]
    
    print(f"✅ Extracted sub-cube {cube_index} from 3D array")
    print(f"   Original shape: {cube_3d.shape}")
    print(f"   Sub-cube shape: {subcube.shape}")
    print(f"   Sub-cube position: X[{start_x}:{end_x}], Y[{start_y}:{end_y}], Z[{start_z}:{end_z}]")
    print(f"   Total available sub-cubes: {total_cubes} ({n_cubes_x}x{n_cubes_y}x{n_cubes_z})")
    
    return subcube


def get_subcube_info(cube_3d, subcube_length):
    """
    Get information about how many sub-cubes can be extracted from a 3D array.
    
    Args:
        cube_3d (numpy.ndarray): 3D input array
        subcube_length (int): Length of each side of the sub-cube
        
    Returns:
        dict: Information about sub-cube extraction possibilities
    """
    if len(cube_3d.shape) != 3:
        raise ValueError(f"Input must be 3D array, got shape {cube_3d.shape}")
    
    nx, ny, nz = cube_3d.shape
    
    # Calculate how many sub-cubes fit in each dimension
    n_cubes_x = nx // subcube_length
    n_cubes_y = ny // subcube_length
    n_cubes_z = nz // subcube_length
    
    # Calculate total number of possible sub-cubes
    total_cubes = n_cubes_x * n_cubes_y * n_cubes_z
    
    # Calculate remaining space (unused)
    remaining_x = nx % subcube_length
    remaining_y = ny % subcube_length
    remaining_z = nz % subcube_length
    
    info = {
        'original_shape': cube_3d.shape,
        'subcube_length': subcube_length,
        'n_cubes_x': n_cubes_x,
        'n_cubes_y': n_cubes_y,
        'n_cubes_z': n_cubes_z,
        'total_cubes': total_cubes,
        'remaining_x': remaining_x,
        'remaining_y': remaining_y,
        'remaining_z': remaining_z,
        'efficiency': (total_cubes * subcube_length**3) / (nx * ny * nz) * 100
    }
    
    print(f"📊 Sub-cube extraction analysis:")
    print(f"   Original cube: {nx}×{ny}×{nz}")
    print(f"   Sub-cube size: {subcube_length}×{subcube_length}×{subcube_length}")
    print(f"   Available sub-cubes: {total_cubes} ({n_cubes_x}×{n_cubes_y}×{n_cubes_z})")
    print(f"   Remaining space: X:{remaining_x}, Y:{remaining_y}, Z:{remaining_z}")
    print(f"   Extraction efficiency: {info['efficiency']:.1f}%")
    
    return info


def extract_all_subcubes(cube_3d, subcube_length):
    """
    Extract all possible non-overlapping sub-cubes from a 3D array.
    
    Args:
        cube_3d (numpy.ndarray): 3D input array
        subcube_length (int): Length of each side of the sub-cube
        
    Returns:
        list: List of all extracted sub-cubes
    """
    info = get_subcube_info(cube_3d, subcube_length)
    total_cubes = info['total_cubes']
    
    subcubes = []
    for i in range(total_cubes):
        subcube = extract_subcube(cube_3d, subcube_length, i)
        subcubes.append(subcube)
    
    print(f"✅ Extracted all {total_cubes} sub-cubes")
    return subcubes
