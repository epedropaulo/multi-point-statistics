import numpy as np
import os
import mpslib as mps

def transform_ti(path):
    di = 2
    coarse3d = 1

    Deas = mps.eas.read(path)
    TI = Deas['Dmat']
    
    TI = Deas['Dmat']
    if di>1:
        if coarse3d==0:
            Dmat = TI
            TI = Dmat[::di,::di, :]
        else:
            Dmat = TI
            TI = mps.trainingimages.coarsen_2d_ti(Dmat, di)
    
    mps.eas.write_mat(TI, path)

    return TI

def save_binary_as_ti_and_npy(binary_data, output_path):
    """
    Save binary data as both .dat (EAS format) and .npy (NumPy array) files
    
    Args:
        binary_data (numpy.ndarray): The binary data array to save
        output_path (str): Base path for output files (without extension)
        
    Returns:
        tuple: (dat_path, npy_path) - paths to the saved files
    """
    
    # Remove extensions if present
    if output_path.endswith('.dat'):
        output_path = output_path[:-4]
    if output_path.endswith('.npy'):
        output_path = output_path[:-4]
    
    # Save as .dat file (EAS format)
    dat_path = output_path + '.dat'
    mps.eas.write_mat(filename=dat_path, D=binary_data)
    
    npy_path = output_path + '.npy'
    np.save(npy_path, binary_data[:,:, None])

    binary_data_transformed = transform_ti(dat_path)
    
    print(f"✅ Saved as .dat: {dat_path}")

    # Save as .npy file (NumPy array)
    npy_trasnformed_path = output_path + '_transformed.npy'
    np.save(npy_trasnformed_path, binary_data_transformed)
    print(f"✅ Saved as .npy: {npy_trasnformed_path}")
    print(f"   Shape: {binary_data_transformed.shape}, Type: {binary_data_transformed.dtype}")
    
    return dat_path, npy_path, npy_trasnformed_path

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
