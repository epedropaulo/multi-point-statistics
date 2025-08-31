import numpy as np
import os
import mpslib as mps

def transform_ti(path, downsample_factor=2):
    """
    Transform training image data with downsampling.
    Handles both 2D and 3D data automatically.
    
    Args:
        path (str): Path to the .dat file
        downsample_factor (int): Factor for downsampling (default: 2)
        
    Returns:
        numpy.ndarray: Transformed training image data
    """
    
    # Read the EAS file
    Deas = mps.eas.read(path)
    TI = Deas['Dmat']
    
    print(f"📖 Loaded TI from {path}")
    print(f"   Original shape: {TI.shape}")
    
    # Apply downsampling if needed
    if downsample_factor > 1:
        if len(TI.shape) == 3 and TI.shape[2] > 1:
            # 3D data with multiple Z layers
            print(f"🔄 Downsampling 3D data with factor {downsample_factor}")
            Dmat = TI
            TI = Dmat[::downsample_factor, ::downsample_factor, ::downsample_factor]
        elif len(TI.shape) == 3 and TI.shape[2] == 1:
            # 2D data stored as 3D with single Z layer
            print(f"🔄 Downsampling 2D data with factor {downsample_factor}")
            Dmat = TI
            TI = mps.trainingimages.coarsen_2d_ti(Dmat, downsample_factor)
        else:
            # Fallback for other cases
            print(f"🔄 Downsampling with factor {downsample_factor}")
            Dmat = TI
            TI = Dmat[::downsample_factor, ::downsample_factor, :]
        
        print(f"   Transformed shape: {TI.shape}")
        
        # Save the transformed data back to the file
        mps.eas.write_mat(TI, path)
        print(f"💾 Saved transformed data to {path}")
    
    return TI

def save_binary_as_ti_and_npy(binary_data, output_path, downsample_factor=2):
    """
    Save binary data as both .dat (EAS format) and .npy (NumPy array) files.
    Handles both 2D and 3D data automatically.
    
    Args:
        binary_data (numpy.ndarray): The binary data array to save (2D or 3D)
        output_path (str): Base path for output files (without extension)
        downsample_factor (int): Factor for downsampling (default: 2)
        
    Returns:
        tuple: (dat_path, npy_path, npy_transformed_path) - paths to the saved files
    """
    
    # Remove extensions if present
    if output_path.endswith('.dat'):
        output_path = output_path[:-4]
    if output_path.endswith('.npy'):
        output_path = output_path[:-4]
    
    # Check data dimensions and handle accordingly
    if len(binary_data.shape) == 2:
        # 2D data - add Z dimension for EAS format
        print(f"📊 Input: 2D data with shape {binary_data.shape}")
        data_for_eas = binary_data[:, :, None]  # Add Z dimension
        is_3d = False
    elif len(binary_data.shape) == 3:
        # 3D data - use as is
        print(f"📊 Input: 3D data with shape {binary_data.shape}")
        data_for_eas = binary_data
        is_3d = True
    else:
        raise ValueError(f"Data must be 2D or 3D, got shape {binary_data.shape}")
    
    # Save as .dat file (EAS format)
    dat_path = output_path + '.dat'
    mps.eas.write_mat(filename=dat_path, D=data_for_eas)
    print(f"✅ Saved as .dat: {dat_path}")
    
    # Save original data as .npy file
    npy_path = output_path + '.npy'
    np.save(npy_path, binary_data)
    print(f"✅ Saved original as .npy: {npy_path}")
    print(f"   Shape: {binary_data.shape}, Type: {binary_data.dtype}")
    
    # Apply transformation (downsampling) if needed
    if downsample_factor > 1:
        print(f"🔄 Applying downsampling with factor {downsample_factor}")
        binary_data_transformed = transform_ti(dat_path, downsample_factor)
        
        # Save transformed data as .npy file
        npy_transformed_path = output_path + '_transformed.npy'
        np.save(npy_transformed_path, binary_data_transformed)
        print(f"✅ Saved transformed as .npy: {npy_transformed_path}")
        print(f"   Shape: {binary_data_transformed.shape}, Type: {binary_data_transformed.dtype}")
    else:
        print("⏭️  Skipping transformation (downsample_factor = 1)")
        npy_transformed_path = None
    
    return dat_path, npy_path, npy_transformed_path

def load_ti_from_npy(npy_path):
    """
    Load training image array from .npy file
    
    Args:
        npy_path (str): Path to the .npy file
        
    Returns:
        numpy.ndarray: The loaded training image array
    """
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"File not found: {npy_path}")
    
    ti_array = np.load(npy_path)
    print(f"✅ Loaded TI from: {npy_path}")
    print(f"   Shape: {ti_array.shape}, Type: {ti_array.dtype}")
    
    return ti_array
