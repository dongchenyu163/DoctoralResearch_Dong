import numpy
from typing import Any

class GPTParams:
    search_radius: float = 0.025
    mu: float = 2.5
    max_nearest_neighbors: int = 100
    max_surface_angle: float = 45.0
    min_angle: float = 10.0
    max_angle: float = 120.0
    normal_consistency: bool = False
    
    def __init__(self) -> None: ...

def compute_mesh(input_points: numpy.ndarray, params: GPTParams) -> numpy.ndarray:
    """
    Compute mesh from Nx6 point cloud (XYZ+Normal).
    Returns (M, 3) array of triangle indices.
    """
    ...
