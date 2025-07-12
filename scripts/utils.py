import numpy as np

def load_binary_from_eleven_sandstones(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        unshaped_voxel = np.fromfile(f, dtype=np.uint8)

    size = int(round(unshaped_voxel.shape[0] ** (1 / 3)))
    if size**3 != unshaped_voxel.shape[0]:
        raise ValueError('Voxel is not a cube.')

    return unshaped_voxel.reshape((size, size, size))