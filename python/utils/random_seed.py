"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: Optional[int]) -> None:
    """Set random seeds for Python and NumPy."""
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
