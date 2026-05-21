import h5py
import numpy as np
from torch.utils.data import Dataset
from moleculekit.util import uniformRandomRotation
from scipy.interpolate import RegularGridInterpolator
from .ecloud_utils.grid import BuildGridCenters

class ecloud_dataset(Dataset):
    def __init__(self, h5_path, rotate = False):
        super().__init__()
        self.h5_path = h5_path
        self.f = h5py.File(self.h5_path, 'r')
        self.rotate = rotate

    def __len__(self) -> int:
        return len(self.f[list(self.f.keys())[0]])
    
    def __getitem__(self, i: int):
        batch = {}
        for k in self.f.keys():
            batch[k] = self.f[k][i]
            if 'ecloud' in k and self.rotate:
                batch[k] = rotate_voxel(batch[k], protocol(grid=64))

        return batch
    
def protocol(grid, resolution=0.5):
    '''
    Define the grid protocol, including grid size, resolution, and grid centers
        grid size is the number of grids in each dimension
        resolution is the distance between two grid points

    '''
    N = [grid, grid, grid]
    llc = (np.zeros(3) - float(grid * resolution / 2)) + resolution / 2  # lower left corner
    grids = BuildGridCenters(llc, N, resolution)    
    return {'grids':grids, 'N':N, 'llc':llc}

def rotate_voxel(voxel, protocol):
    centers = protocol['grids']
    coords = centers.reshape(-1, 3)
    rotMat = uniformRandomRotation()
    rotated_coords = np.dot(coords, rotMat.T)
    x_range = centers[:, 0, 0, 0]
    y_range = centers[0, :, 0, 1]
    z_range = centers[0, 0, :, 2]
    interpolator = RegularGridInterpolator((x_range, y_range, z_range), voxel, bounds_error=False, fill_value=0)
    new_density = interpolator(rotated_coords).reshape(*protocol['N'])
    return new_density