import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_binary_from_eleven_sandstones(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        unshaped_voxel = np.fromfile(f, dtype=np.uint8)

    size = int(round(unshaped_voxel.shape[0] ** (1 / 3)))
    if size**3 != unshaped_voxel.shape[0]:
        raise ValueError('Voxel is not a cube.')

    return unshaped_voxel.reshape((size, size, size))

def npy_to_hard_data(npy_path: str, threshold=0.5, max_points=None, downsample_factor=1) -> np.ndarray:
    """
    Convert a .npy array to hard data structure format [X, Y, Z, VALUE].
    
    Args:
        npy_path (str): Path to the .npy file
        threshold (float): Threshold to convert continuous values to binary (default: 0.5)
        max_points (int): Maximum number of points to return (default: None = all points)
        downsample_factor (int): Factor to scale coordinates back to original system (default: 1)
        
    Returns:
        np.ndarray: Hard data array in format [X, Y, Z, VALUE]
        
    Example:
        # Load array and convert to hard data with coordinate scaling
        hard_data = npy_to_hard_data('my_array.npy', threshold=0.5, downsample_factor=4)
        
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
    elif len(array.shape) == 3:
        # 3D array - use only the first layer of Z axis
        nx, ny, nz_full = array.shape
        nz = 1
        array_3d = array[:, :, 0:1]  # Take only the first Z layer
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
    print(f"   Array shape: {array.shape}")
    print(f"   Hard data points: {len(hard_data)}")
    print(f"   Value range: [{hard_data[:, 3].min():.2f}, {hard_data[:, 3].max():.2f}]")
    print(f"   Coordinate scaling factor: {downsample_factor}")
    print(f"   Scaled coordinate range: X[0, {hard_data[:, 0].max()}], Y[0, {hard_data[:, 1].max()}], Z[0, {hard_data[:, 2].max()}]")
    
    return hard_data

def plot_realizations_enhanced(O, n_realizations=4, figsize=(15, 10), cmap='viridis', 
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
        
        # Add statistics in text box
        stats_text = f'Min: {plot_data.min():.2f}\nMax: {plot_data.max():.2f}\nMean: {plot_data.mean():.2f}'
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