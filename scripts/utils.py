import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from scipy.ndimage import zoom

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
