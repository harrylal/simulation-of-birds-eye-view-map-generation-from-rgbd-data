import numpy as np

def clip_pointcloud(pointcloud, x_range=(-40, 40), y_range=(-40, 40), z_range=(-1, 1)):
    """
        Clip pointcloud to a specific range
        Arguments:
            pointcloud: shape = [n_points, 3]
            x_range: tuple of (min, max)
            y_range: tuple of (min, max)
            z_range: tuple of (min, max)
        Return:
            clipped pointcloud
    """
    mask = np.where(
        (pointcloud[:,0] > x_range[0]) & (pointcloud[:,0] < x_range[1]) &
        (pointcloud[:,1] > y_range[0]) & (pointcloud[:,1] < y_range[1]) &
        (pointcloud[:,2] > z_range[0]) & (pointcloud[:,2] < z_range[1])
    )
    
    return pointcloud[mask] 

   