# Metrics Module

This module provides functions to compute and display pore structure metrics using `poregen.features` for simulated data and target data.

## Features

- Compute pore structure metrics (porosity, permeability, surface area density, Euler number density)
- Compare metrics between simulations and target data
- Generate visualizations (histograms, box plots)
- Export results to CSV files
- Support for analyzing a subset of simulations

## Functions

### `compute_and_display_metrics(simulated_data_path, target_data_path, N=None, voxel_length=1e-6, save_plots=True, output_dir=None)`

Main function to compute and display metrics for simulated data and target data.

**Parameters:**
- `simulated_data_path` (str): Path to directory containing simulated data files (.npy files)
- `target_data_path` (str): Path to target data file (.npy file)
- `N` (int, optional): Number of simulations to analyze. If None, analyzes all available simulations
- `voxel_length` (float): Voxel length for permeability calculation (default: 1e-6)
- `save_plots` (bool): Whether to save plots to files (default: True)
- `output_dir` (str, optional): Directory to save plots. If None, saves in current directory

**Returns:**
- `pd.DataFrame`: DataFrame containing all computed metrics

### `compare_metrics_distributions(simulated_data_path, target_data_path, N=None, voxel_length=1e-6, save_plots=True, output_dir=None)`

Create box plots comparing metric distributions between simulations and target.

**Parameters:**
- Same as `compute_and_display_metrics`

### `load_simulation_data(simulated_data_path, target_data_path, N=None)`

Load simulation data and target data.

**Parameters:**
- `simulated_data_path` (str): Path to directory containing simulated data files
- `target_data_path` (str): Path to target data file
- `N` (int, optional): Number of simulations to load

**Returns:**
- `Tuple[List[np.ndarray], np.ndarray]`: Tuple containing list of simulation arrays and target array

### `compute_metrics_for_data(data, voxel_length=1e-6)`

Compute pore structure metrics for a single dataset.

**Parameters:**
- `data` (np.ndarray): Binary pore structure data
- `voxel_length` (float): Voxel length for permeability calculation

**Returns:**
- `Dict[str, float]`: Dictionary containing computed metrics

## Usage Example

```python
from scripts.metrics import compute_and_display_metrics, compare_metrics_distributions

# Compute and display metrics
df = compute_and_display_metrics(
    simulated_data_path="/path/to/simulated/data",
    target_data_path="/path/to/target/data.npy",
    N=10,  # Analyze first 10 simulations
    voxel_length=1e-6,
    save_plots=True,
    output_dir="/path/to/output"
)

# Create box plot comparison
compare_metrics_distributions(
    simulated_data_path="/path/to/simulated/data",
    target_data_path="/path/to/target/data.npy",
    N=10,
    voxel_length=1e-6,
    save_plots=True,
    output_dir="/path/to/output"
)
```

## Output Files

The module generates the following output files:

1. `metrics_comparison.png`: Histogram comparison of metrics between simulations and target
2. `metrics_boxplots.png`: Box plot comparison of metric distributions
3. `metrics_detailed.csv`: Detailed CSV file with all computed metrics

## Dependencies

- `poregen.features`
- `torch`
- `numpy`
- `matplotlib`
- `pandas`
- `glob`
- `os`
- `typing`

## Notes

- The module expects binary pore structure data (0s and 1s)
- All data files should be in .npy format
- The module automatically handles batch processing of multiple simulations
- Results include statistical summaries and relative errors compared to target values
