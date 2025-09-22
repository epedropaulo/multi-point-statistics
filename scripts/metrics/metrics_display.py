"""
Functions for computing and displaying pore structure metrics.

This module provides functions to compute various pore structure metrics
using poregen.features for simulated data and target data.
"""

import csv
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List, Dict, Tuple
import poregen.features


def load_simulation_data(simulated_data_path: str, target_data_path: str, N: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load simulation data and target data.
    
    Parameters:
    -----------
    simulated_data_path : str
        Path to the directory containing simulated data files (.npy files)
    target_data_path : str
        Path to the target data file (.npy file)
    N : int, optional
        Number of simulations to load. If None, loads all available simulations.
        
    Returns:
    --------
    Tuple[List[np.ndarray], np.ndarray]
        Tuple containing list of simulation arrays and target array
    """
    # Load target data
    if not os.path.exists(target_data_path):
        raise FileNotFoundError(f"Target data file not found: {target_data_path}")
    target_data = np.load(target_data_path)
    
    # Find all simulation files
    sim_files = glob.glob(os.path.join(simulated_data_path, "*.npy"))
    sim_files.sort()  # Sort to ensure consistent ordering
    
    if not sim_files:
        raise FileNotFoundError(f"No .npy files found in {simulated_data_path}")
    
    # Limit number of simulations if N is specified
    if N is not None:
        sim_files = sim_files[:N]
    
    # Load simulation data
    simulation_data = []
    for sim_file in sim_files:
        sim_data = np.load(sim_file)
        simulation_data.append(sim_data)
    
    print(f"Loaded {len(simulation_data)} simulations from {simulated_data_path}")
    print(f"Loaded target data from {target_data_path}")
    
    return simulation_data, target_data


def compute_metrics_for_data(data: np.ndarray, voxel_length: float = 1e-6) -> Dict[str, float]:
    """
    Compute pore structure metrics for a single dataset.
    
    Parameters:
    -----------
    data : np.ndarray
        Binary pore structure data
    voxel_length : float
        Voxel length for permeability calculation
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing computed metrics
    """
    # Convert to torch tensor and add batch dimension
    data_tensor = torch.tensor(data).unsqueeze(0)
    
    # Initialize feature extractors
    porosity = poregen.features.feature_extractors.PorosityExtractor()
    permeability = poregen.features.feature_extractors.PermeabilityExtractor(voxel_length=voxel_length)
    surface_area_density = poregen.features.feature_extractors.SurfaceAreaDensityExtractor(voxel_size=voxel_length)
    euler_number_density = poregen.features.feature_extractors.EulerNumberDensityExtractor(voxel_size=voxel_length)
    
    # Compute all features
    all_features = poregen.features.feature_extractors.CompositeExtractor(
        [porosity, permeability, surface_area_density, euler_number_density]
    )
    
    # Extract features
    features = all_features(data_tensor)

    return features


def compute_and_display_metrics(simulated_data_path: str, target_data_path: str, N: Optional[int] = None, 
                               voxel_length: float = 1e-6, save_metrics: bool = False,
                               ) -> Dict:
    """
    Compute and display metrics for simulated data and target data.
    
    Parameters:
    -----------
    simulated_data_path : str
        Path to the directory containing simulated data files (.npy files)
    target_data_path : str
        Path to the target data file (.npy file)
    N : int, optional
        Number of simulations to analyze. If None, analyzes all available simulations.
    voxel_length : float
        Voxel length for permeability calculation
    save_metrics : bool
        Whether to save metrics to a file
    
    Returns:
    --------
    Dict
        Dictionary containing all computed metrics
    """
    # Load data
    simulation_data, target_data = load_simulation_data(simulated_data_path, target_data_path, N)
    
    # Compute metrics for all simulations
    sim_metrics = []
    for i, sim_data in enumerate(simulation_data):
        print(f"Computing metrics for simulation {i+1}/{len(simulation_data)}")
        metrics = compute_metrics_for_data(sim_data, voxel_length)
        metrics['simulation_id'] = i
        metrics['data_type'] = 'simulation'
        sim_metrics.append(metrics)
    
    # Compute metrics for target
    print("Computing metrics for target data")
    target_metrics = compute_metrics_for_data(target_data, voxel_length)
    target_metrics['simulation_id'] = -1  # Use -1 to indicate target
    target_metrics['data_type'] = 'target'
    
    # Create visualizations
    metric_names = ['porosity', 'permeability', 'surface_area_density', 'euler_number_density']

    # Print summary statistics
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    
    for metric in metric_names:
        sim_data = [m[metric] for m in sim_metrics]
        target_value = target_metrics[metric]
        
        mean_sim = np.mean(sim_data)
        std_sim = np.std(sim_data)
        min_sim = np.min(sim_data)
        max_sim = np.max(sim_data)
        
        if metric != 'permeability':
            # Calculate relative error
            rel_error = abs(target_value - mean_sim) / target_value * 100 if target_value != 0 else 0
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Target: {target_value}")
        print(f"  Simulations: {mean_sim} ± {std_sim}")
        print(f"  Range: [{min_sim}, {max_sim}]")
        if metric != 'permeability':
            print(f"  Relative Error: {rel_error}%")

    # Save metrics to a file
    if save_metrics:
        with open('metrics.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['metric'] + metric_names)
            for metric in metric_names:
                writer.writerow([metric] + [m[metric] for m in sim_metrics] + [target_metrics[metric]])
            writer.writerows(sim_metrics)
            writer.writerow(target_metrics)

    # Return dictionary with all metrics
    return {
        'simulations': sim_metrics,
        'target': target_metrics
    }


def compare_metrics_distributions(simulated_data_path: str, target_data_path: str, 
                                N: Optional[int] = None, voxel_length: float = 1e-6,
                                save_plots: bool = True, output_dir: str = None) -> None:
    """
    Create box plots comparing metric distributions between simulations and target.
    
    Parameters:
    -----------
    simulated_data_path : str
        Path to the directory containing simulated data files (.npy files)
    target_data_path : str
        Path to the target data file (.npy file)
    N : int, optional
        Number of simulations to analyze. If None, analyzes all available simulations.
    voxel_length : float
        Voxel length for permeability calculation
    save_plots : bool
        Whether to save plots to files
    output_dir : str, optional
        Directory to save plots. If None, saves in current directory.
    """
    # Load data and compute metrics
    metrics_dict = compute_and_display_metrics(simulated_data_path, target_data_path, N, 
                                             voxel_length, save_plots=False)
    
    # Create box plots
    metric_names = ['porosity', 'permeability', 'surface_area_density', 'euler_number_density']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        
        # Prepare data for box plot
        sim_data = [m[metric] for m in metrics_dict['simulations']]
        target_data = [metrics_dict['target'][metric]]
        
        # Create box plot
        box_data = [sim_data, target_data]
        labels = ['Simulations', 'Target']
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution Comparison')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'metrics_boxplots.png'), dpi=300, bbox_inches='tight')
        print(f"Box plots saved to {os.path.join(output_dir, 'metrics_boxplots.png')}")
    
    plt.show()
