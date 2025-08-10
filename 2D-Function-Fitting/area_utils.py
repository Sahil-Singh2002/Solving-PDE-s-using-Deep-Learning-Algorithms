"""
area_utils.py
==============

Routines for computing areas of intersection polygons arising in the
ReLU network integration problem.  These functions wrap the methods
exposed by ``Domain2D`` in ``geometry_utils`` to compute the area
contributed by diagonal and lower matrix entries, as well as the
boundary length vector used in the mass matrix assembly.  The
formulations follow those from the original notebook but are
reorganised into standalone functions.

Functions
---------
diagonal_matrix_entry(domain, w_i, b_i)
    Compute the diagonal entry corresponding to a single hyperplane.

lower_matrix_entry(domain, w_i, b_i, w_j, b_j)
    Compute the off‑diagonal entry between two hyperplanes.

compute_F_from_step1(domain)
    Compute the vector of edge lengths of the base triangle polygon.

compute_f_integral(domain, w_f, b_f)
    Compute the integral of a single ReLU over the base domain.

Note that all functions return scalars (or arrays) of type ``float`` or
``numpy.ndarray`` and are pure – they perform no plotting or I/O.
"""

from __future__ import annotations

import numpy as np
from typing import List

from geometry_utils import Domain2D, TOL


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute the Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def diagonal_matrix_entry(domain: Domain2D, w_i: np.ndarray, b_i: float) -> float:
    """Compute the area contribution for a diagonal mass matrix entry.

    Given a domain and a single hyperplane defined by ``w_i`` and ``b_i``,
    this function calls ``Domain2D.step_2`` to obtain the polygon of
    intersection and sums over its edges (length multiplied by offset
    ratio) following the divergence theorem derivation from the
    notebook.  If the polygon has fewer than three vertices, zero
    contribution is returned.

    Parameters
    ----------
    domain : Domain2D
        The base triangular domain.
    w_i, b_i : np.ndarray, float
        The normal vector and offset of the hyperplane.

    Returns
    -------
    float
        The diagonal mass matrix entry (area of ReLU squared region).
    """
    nodes_poly = domain.step_2(w_i, b_i)
    if len(nodes_poly) < 3:
        return 0.0

    # Identify the bounding edges that lie on a specific hyperplane and
    # accumulate their contributions.  The boundaries to test include
    # the domain boundaries and the current hyperplane.
    boundaries = domain.boundaries_as_tuples + [(w_i, b_i)]
    a_list: List[np.ndarray] = []
    b_list: List[float] = []
    for i in range(len(nodes_poly)):
        p1 = nodes_poly[i]["coordinates"]
        p2 = nodes_poly[(i + 1) % len(nodes_poly)]["coordinates"]
        # Determine which hyperplane both points lie on
        for a, b in boundaries:
            if (abs(np.dot(a, p1) - b) < TOL and
                    abs(np.dot(a, p2) - b) < TOL):
                a_list.append(a)
                b_list.append(b)
                break

    summation = 0.0
    for i in range(len(a_list)):
        a = a_list[i]
        b = b_list[i]
        edge_len = euclidean_distance(nodes_poly[i]["coordinates"],
                                      nodes_poly[(i + 1) % len(nodes_poly)]["coordinates"])
        summation += (b / np.linalg.norm(a)) * edge_len
    return abs(summation) * 0.5


def lower_matrix_entry(domain: Domain2D, w_i: np.ndarray, b_i: float,
                       w_j: np.ndarray, b_j: float) -> float:
    """Compute the area contribution for an off‑diagonal mass matrix entry.

    Similar to ``diagonal_matrix_entry`` but uses ``Domain2D.step_3``
    to incorporate two hyperplanes.  If the resulting polygon is
    degenerate (fewer than three vertices), zero is returned.

    Parameters
    ----------
    domain : Domain2D
        The base triangular domain.
    w_i, b_i : np.ndarray, float
        The first hyperplane.
    w_j, b_j : np.ndarray, float
        The second hyperplane.

    Returns
    -------
    float
        The lower (and by symmetry, upper) mass matrix entry.
    """
    nodes_poly = domain.step_3(w_i, b_i, w_j, b_j)
    if len(nodes_poly) < 3:
        return 0.0
    boundaries = domain.boundaries_as_tuples + [(w_i, b_i), (w_j, b_j)]
    a_list: List[np.ndarray] = []
    b_list: List[float] = []
    for i in range(len(nodes_poly)):
        p1 = nodes_poly[i]["coordinates"]
        p2 = nodes_poly[(i + 1) % len(nodes_poly)]["coordinates"]
        for a, b in boundaries:
            if (abs(np.dot(a, p1) - b) < TOL and
                    abs(np.dot(a, p2) - b) < TOL):
                a_list.append(a)
                b_list.append(b)
                break
    summation = 0.0
    for i in range(len(a_list)):
        a = a_list[i]
        b = b_list[i]
        edge_len = euclidean_distance(nodes_poly[i]["coordinates"],
                                      nodes_poly[(i + 1) % len(nodes_poly)]["coordinates"])
        summation += (b / np.linalg.norm(a)) * edge_len
    return abs(summation) * 0.5


def compute_F_from_step1(domain: Domain2D) -> float:
    """Compute the area of the base triangle used as the normalising factor.

    The base triangle area is needed for the cost function and is
    computed via the same edge‑based sum used in the original
    notebook.  This function returns the area as a scalar.

    Parameters
    ----------
    domain : Domain2D
        The domain for which to compute the area.

    Returns
    -------
    float
        The area of the triangle (positive scalar).
    """
    nodes_1 = domain.step_1()
    coords = [node["coordinates"] for node in nodes_1]
    n = len(coords)
    F = np.zeros((n, 1))
    for i in range(n):
        j = (i + 1) % n
        F[i, 0] = euclidean_distance(coords[i], coords[j])
    a_list: List[np.ndarray] = []
    b_list: List[float] = []
    # Determine which edges lie on each boundary line
    for i in range(n):
        j = (i + 1) % n
        Ai, bi = nodes_1[i]["A"], nodes_1[i]["b"]
        Aj, bj = nodes_1[j]["A"], nodes_1[j]["b"]
        for idx_i, row_i in enumerate(Ai):
            for idx_j, row_j in enumerate(Aj):
                if (np.allclose(row_i, row_j, atol=TOL) and
                        np.isclose(bi[idx_i], bj[idx_j], atol=TOL)):
                    a_list.append(row_i)
                    b_list.append(bi[idx_i])
    summation_result = 0.0
    for i in range(min(len(a_list), n)):
        beta_i = b_list[i]
        a_i = a_list[i]
        norm_a = np.linalg.norm(a_i)
        edge_len = F[i, 0]
        summation_result += (beta_i / norm_a) * edge_len
    return 0.5 * abs(summation_result)


def compute_f_integral(domain: Domain2D, w_f: np.ndarray, b_f: float) -> float:
    """Compute the integral of ``ReLU(w_f^T x - b_f)`` over the base domain.

    This is simply the diagonal matrix entry for a single hyperplane and
    is provided as a convenience wrapper.
    """
    return diagonal_matrix_entry(domain, w_f, b_f)
