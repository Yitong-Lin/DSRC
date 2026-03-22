
"""
Utility functions related to point cloud
"""

import struct
import numpy as np


def pcd_to_np(pcd_file):
    """
    Read pcd and return numpy array.

    Parameters
    ----------
    pcd_file : str
        The pcd file that contains the point cloud.

    Returns
    -------
    pcd_np : np.ndarray
        The lidar data in numpy format, shape:(n, 4)
    """
    with open(pcd_file, 'rb') as f:
        header = {}
        header_lines = 0
        while True:
            line = f.readline().decode('utf-8', errors='replace').strip()
            header_lines += 1
            if line.startswith('#') or not line:
                continue
            key, *vals = line.split()
            header[key.upper()] = vals
            if key.upper() == 'DATA':
                data_type = vals[0].lower()
                break

        fields = header.get('FIELDS', [])
        sizes = [int(s) for s in header.get('SIZE', [])]
        types = header.get('TYPE', [])
        counts = [int(c) for c in header.get('COUNT', ['1'] * len(fields))]
        num_points = int(header.get('POINTS', [0])[0])

        # Build dtype
        dtype_map = {'F': 'f', 'I': 'i', 'U': 'u'}
        dt_list = []
        for i, (fld, sz, tp, cnt) in enumerate(zip(fields, sizes, types, counts)):
            np_type = dtype_map.get(tp, 'f') + str(sz)
            if cnt == 1:
                dt_list.append((fld, np_type))
            else:
                dt_list.append((fld, np_type, (cnt,)))
        dt = np.dtype(dt_list)

        if data_type == 'binary':
            data = np.frombuffer(f.read(dt.itemsize * num_points), dtype=dt)
        elif data_type == 'ascii':
            rows = []
            for _ in range(num_points):
                row = f.readline().decode('utf-8').strip().split()
                rows.append(tuple(row))
            data = np.array(rows, dtype=dt)
        else:
            raise ValueError(f'Unsupported PCD data type: {data_type}')

    # Extract x, y, z and intensity
    xyz = np.column_stack([data['x'].astype(np.float32),
                           data['y'].astype(np.float32),
                           data['z'].astype(np.float32)])
    if 'intensity' in data.dtype.names:
        intensity = data['intensity'].astype(np.float32).reshape(-1, 1)
        if intensity.max() > 1.0:
            intensity = intensity / 255.0
    elif 'rgb' in data.dtype.names or 'rgba' in data.dtype.names:
        # packed float rgb: bytes are [B, G, R, 0] in little-endian
        # R channel is intensity (open3d pcd convention)
        rgb_bytes = np.ascontiguousarray(data['rgb']).view(np.uint8).reshape(-1, 4)
        intensity = (rgb_bytes[:, 2].astype(np.float32) / 255.0).reshape(-1, 1)
    else:
        intensity = np.zeros((len(xyz), 1), dtype=np.float32)

    pcd_np = np.hstack((xyz, intensity))
    return np.asarray(pcd_np, dtype=np.float32)


def mask_points_by_range(points, limit_range):
    """
    Remove the lidar points out of the boundary.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    limit_range : list
        [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """

    mask = (points[:, 0] > limit_range[0]) & (points[:, 0] < limit_range[3])\
           & (points[:, 1] > limit_range[1]) & (
                   points[:, 1] < limit_range[4]) \
           & (points[:, 2] > limit_range[2]) & (
                   points[:, 2] < limit_range[5])

    points = points[mask]

    return points


def mask_ego_points(points):
    """
    Remove the lidar points of the ego vehicle itself.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """
    mask = (points[:, 0] >= -1.95) & (points[:, 0] <= 2.95) \
           & (points[:, 1] >= -1.1) & (points[:, 1] <= 1.1)
    points = points[np.logical_not(mask)]

    return points


def shuffle_points(points):
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]

    return points


def lidar_project(lidar_data, extrinsic):
    """
    Given the extrinsic matrix, project lidar data to another space.

    Parameters
    ----------
    lidar_data : np.ndarray
        Lidar data, shape: (n, 4)

    extrinsic : np.ndarray
        Extrinsic matrix, shape: (4, 4)

    Returns
    -------
    projected_lidar : np.ndarray
        Projected lida data, shape: (n, 4)
    """

    lidar_xyz = lidar_data[:, :3].T
    # (3, n) -> (4, n), homogeneous transformation
    lidar_xyz = np.r_[lidar_xyz, [np.ones(lidar_xyz.shape[1])]]
    lidar_int = lidar_data[:, 3]

    # transform to ego vehicle space, (3, n)
    project_lidar_xyz = np.dot(extrinsic, lidar_xyz)[:3, :]
    # (n, 3)
    project_lidar_xyz = project_lidar_xyz.T
    # concatenate the intensity with xyz, (n, 4)
    projected_lidar = np.hstack((project_lidar_xyz,
                                 np.expand_dims(lidar_int, -1)))

    return projected_lidar


def projected_lidar_stack(projected_lidar_list):
    """
    Stack all projected lidar together.

    Parameters
    ----------
    projected_lidar_list : list
        The list containing all projected lidar.

    Returns
    -------
    stack_lidar : np.ndarray
        Stack all projected lidar data together.
    """
    stack_lidar = []
    for lidar_data in projected_lidar_list:
        stack_lidar.append(lidar_data)

    return np.vstack(stack_lidar)


def downsample_lidar(pcd_np, num):
    """
    Downsample the lidar points to a certain number.

    Parameters
    ----------
    pcd_np : np.ndarray
        The lidar points, (n, 4).

    num : int
        The downsample target number.

    Returns
    -------
    pcd_np : np.ndarray
        The downsampled lidar points.
    """
    assert pcd_np.shape[0] >= num

    selected_index = np.random.choice((pcd_np.shape[0]),
                                      num,
                                      replace=False)
    pcd_np = pcd_np[selected_index]

    return pcd_np


def downsample_lidar_minimum(pcd_np_list):
    """
    Given a list of pcd, find the minimum number and downsample all
    point clouds to the minimum number.

    Parameters
    ----------
    pcd_np_list : list
        A list of pcd numpy array(n, 4).
    Returns
    -------
    pcd_np_list : list
        Downsampled point clouds.
    """
    minimum = np.Inf

    for i in range(len(pcd_np_list)):
        num = pcd_np_list[i].shape[0]
        minimum = num if minimum > num else minimum

    for (i, pcd_np) in enumerate(pcd_np_list):
        pcd_np_list[i] = downsample_lidar(pcd_np, minimum)

    return pcd_np_list
