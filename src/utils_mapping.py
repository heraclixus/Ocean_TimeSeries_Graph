import numpy as np
import torch

def get_bounding_box_indices(lat_size, lon_size, min_lat, max_lat, min_lon, max_lon):
    """
    Get the 1D indices corresponding to a bounding box in a 2D grid after reshaping.
    
    Args:
        lat_size (int): The latitude dimension size
        lon_size (int): The longitude dimension size
        min_lat (int): Minimum latitude index (inclusive)
        max_lat (int): Maximum latitude index (inclusive)
        min_lon (int): Minimum longitude index (inclusive) 
        max_lon (int): Maximum longitude index (inclusive)
    
    Returns:
        np.ndarray: 1D indices corresponding to the bounding box in flattened grid
    """
    indices = []
    
    # Generate all indices within the bounding box
    for lat in range(min_lat, max_lat + 1):
        for lon in range(min_lon, max_lon + 1):
            # Convert 2D coordinates to 1D index
            flat_idx = lat * lon_size + lon
            indices.append(flat_idx)
    
    return np.array(indices, dtype=np.int32)

def grid_coord_to_flat_index(lat, lon, lon_size):
    """
    Convert a 2D grid coordinate to a 1D index after flattening.
    
    Args:
        lat (int): Latitude index
        lon (int): Longitude index
        lon_size (int): Size of the longitude dimension
    
    Returns:
        int: Flat index
    """
    return lat * lon_size + lon

def flat_index_to_grid_coord(idx, lon_size):
    """
    Convert a 1D flat index back to 2D grid coordinates.
    
    Args:
        idx (int): Flat index
        lon_size (int): Size of the longitude dimension
    
    Returns:
        tuple: (lat, lon) grid coordinates
    """
    lat = idx // lon_size
    lon = idx % lon_size
    return lat, lon

def extract_bounding_box(data, min_lat, max_lat, min_lon, max_lon):
    """
    Extract data from a bounding box in a 3D tensor (time, lat, lon).
    
    Args:
        data (np.ndarray or torch.Tensor): Input data of shape (time, lat, lon)
        min_lat (int): Minimum latitude index (inclusive)
        max_lat (int): Maximum latitude index (inclusive)
        min_lon (int): Minimum longitude index (inclusive)
        max_lon (int): Maximum longitude index (inclusive)
    
    Returns:
        np.ndarray or torch.Tensor: Extracted data from the bounding box
    """
    if isinstance(data, torch.Tensor):
        return data[:, min_lat:max_lat+1, min_lon:max_lon+1]
    else:
        return data[:, min_lat:max_lat+1, min_lon:max_lon+1]

def extract_bounding_box_from_flat(data, lat_size, lon_size, min_lat, max_lat, min_lon, max_lon):
    """
    Extract data from a bounding box in a flattened 2D tensor (time, lat*lon).
    
    Args:
        data (np.ndarray or torch.Tensor): Input data of shape (time, lat*lon)
        lat_size (int): Original latitude dimension size
        lon_size (int): Original longitude dimension size
        min_lat (int): Minimum latitude index (inclusive)
        max_lat (int): Maximum latitude index (inclusive)
        min_lon (int): Minimum longitude index (inclusive)
        max_lon (int): Maximum longitude index (inclusive)
    
    Returns:
        np.ndarray or torch.Tensor: Extracted data from the bounding box
    """
    # Get the indices corresponding to the bounding box
    indices = get_bounding_box_indices(lat_size, lon_size, min_lat, max_lat, min_lon, max_lon)
    
    if isinstance(data, torch.Tensor):
        return data[:, indices]
    else:
        return data[:, indices]