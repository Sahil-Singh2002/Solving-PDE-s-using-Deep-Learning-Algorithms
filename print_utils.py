"""
print_utils.py
===============

Utility function for nicely printing matrices and their associated
vectors.  When running from a terminal this function will colour
entries that are nearly zero in red for better visual distinction.

Functions
---------
pretty_print_matrix_and_vector(matrix_name, vector_name, mat, vec)
    Print a matrix and vector side by side with optional zero
    highlighting.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from geometry_utils import TOL


def pretty_print_matrix_and_vector(matrix_name: str,
                                   vector_name: str,
                                   mat: Sequence[Sequence[float]],
                                   vec: Sequence[float]) -> None:
    """Nicely format and print a matrix with a vector side by side.

    Parameters
    ----------
    matrix_name, vector_name : str
        Labels for the matrix and vector being printed.
    mat : array-like
        2D matrix of values (any shape, but typically square).
    vec : array-like
        1D vector of values (length equal to the number of rows of
        ``mat``).
    """
    RED = "\033[91m"
    RESET = "\033[0m"

    mat = np.array(mat, dtype=float)
    vec = np.ravel(vec).astype(float)
    print(f"\n{matrix_name:<50} | {vector_name}")
    for i in range(mat.shape[0]):
        # Format matrix row with coloured zeros
        row_parts = [
            f"{RED}{0:8.4f}{RESET}" if abs(val) < TOL else f"{val:8.4f}"
            for val in mat[i]
        ]
        row_str = "  ".join(row_parts)
        # Format vector entry
        if i < len(vec):
            vec_str = f"{RED}{0:8.4f}{RESET}" if abs(vec[i]) < TOL else f"{vec[i]:8.4f}"
        else:
            vec_str = ""
        print(f"  {row_str}  | {vec_str}")
