"""
geometry_utils.py
===================

This module contains geometric helpers and a simple domain class used to
describe the triangular domain and compute the vertices created when
intersecting half‑spaces in 2D.  The functions in the original
notebook (intersection, domain checks, pairwise intersection routines)
have been consolidated here and organised into a class oriented style.

Classes
-------
Domain2D
    Encapsulates the definition of a triangular domain in the form
    ``a^T x >= b`` for three bounding half‑planes.  Provides methods
    to determine whether points lie inside the domain, construct
    intersection polygons when additional hyperplanes are introduced
    and compute the ordered list of polygon vertices.

Functions
---------
intersection(line1, line2)
    Solve the 2×2 linear system defined by two lines in normal form.

is_inside(point, inequalities)
    Check if a point satisfies all inequalities of the form ``A·x >= b``.

global_intersection(inequalities)
    Compute all pairwise intersections of a set of inequalities and
    return the unique vertices sorted counterclockwise.  Used as a
    robust fallback when the sequential approach fails to produce a
    valid polygon.

remove_duplicate_coordinates(nodes)
    Remove nearly duplicate vertices from a list of node dictionaries
    based on a global tolerance.

The tolerance used throughout the module is exposed via the
module‑level variable ``TOL``.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Iterable, Optional, Dict

import numpy as np

# Global tolerance used for almost‑equality checks.  Exposed here so
# other modules may import and reuse the same constant.
TOL: float = 1e-10


def intersection(line1: Tuple[np.ndarray, float], line2: Tuple[np.ndarray, float]) -> Optional[np.ndarray]:
    """Solve the intersection of two lines in normal form.

    Each line is given as a tuple ``(A, b)`` where ``A`` is a
    length‑2 normal vector and ``b`` a scalar such that all points
    ``x`` on the line satisfy ``A·x = b``.  The routine uses a simple
    Gaussian elimination with partial pivoting to compute the unique
    intersection point.  If the determinant of the resulting system is
    below ``TOL`` the lines are treated as parallel and ``None`` is
    returned.

    Parameters
    ----------
    line1, line2 : Tuple[np.ndarray, float]
        The lines to intersect.  ``line[i][0]`` is a 2‑element normal
        vector and ``line[i][1]`` is the corresponding scalar.

    Returns
    -------
    Optional[np.ndarray]
        The coordinates of the unique intersection point or ``None`` if
        the lines are nearly parallel.
    """
    A1, b1 = line1
    A2, b2 = line2

    # Form the augmented matrix for the two equations.
    A = np.vstack((A1, A2)).astype(float)
    b = np.array([b1, b2], dtype=float)

    # Forward elimination with partial pivoting.
    for k in range(2):
        p = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[p, k]) < TOL:
            return None  # Lines are parallel or coincident.
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        for i in range(k + 1, 2):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Back substitution.
    x = np.empty(2)
    for i in (1, 0):
        rhs = b[i] - A[i, i + 1:] @ x[i + 1:]
        if abs(A[i, i]) < TOL:
            return None
        x[i] = rhs / A[i, i]
    return x


def is_inside(point: np.ndarray, inequalities: Iterable[Tuple[np.ndarray, float]]) -> bool:
    """Check if a point lies inside all given half‑plane inequalities.

    Parameters
    ----------
    point : np.ndarray
        The point to test.
    inequalities : Iterable[Tuple[np.ndarray, float]]
        An iterable of ``(A, b)`` where ``A·x >= b`` defines a half‑plane.

    Returns
    -------
    bool
        ``True`` if the point satisfies every inequality, ``False``
        otherwise.
    """
    for A, b in inequalities:
        if np.dot(A, point) < b - TOL:
            return False
    return True


def global_intersection(inequalities: List[Tuple[np.ndarray, float]]) -> List[Dict[str, np.ndarray]]:
    """Robust fallback intersection routine.

    Given a list of half‑planes, compute all pairwise line
    intersections, discard points lying outside any of the half‑planes,
    remove duplicates and return the vertices sorted in counterclockwise
    order.  Each vertex is packaged in a dictionary with keys ``node``
    and ``coordinates`` so that the format matches the sequential
    routines used in the original notebook.

    Parameters
    ----------
    inequalities : List[Tuple[np.ndarray, float]]
        List of half‑planes ``(A, b)``.

    Returns
    -------
    List[Dict[str, np.ndarray]]
        Ordered vertices of the intersection polygon.  If fewer than
        three unique vertices can be found, an empty list is returned.
    """
    vertices: List[np.ndarray] = []
    n = len(inequalities)
    # Compute pairwise intersections.
    for i in range(n):
        for j in range(i + 1, n):
            pt = intersection(inequalities[i], inequalities[j])
            if pt is not None and is_inside(pt, inequalities):
                vertices.append(pt)
    # Remove duplicates.
    unique_vertices: List[np.ndarray] = []
    for pt in vertices:
        if not any(np.allclose(pt, uv, atol=TOL, rtol=0) for uv in unique_vertices):
            unique_vertices.append(pt)
    if len(unique_vertices) < 3:
        return []
    centroid = np.mean(unique_vertices, axis=0)
    unique_vertices.sort(key=lambda p: math.atan2(p[1] - centroid[1], p[0] - centroid[0]))
    return [{"node": idx, "coordinates": pt} for idx, pt in enumerate(unique_vertices)]


def remove_duplicate_coordinates(nodes: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    """Filter out nearly duplicate node entries.

    Two nodes with coordinates within ``TOL`` in every component are
    considered duplicates and only the first encountered is kept.

    Parameters
    ----------
    nodes : List[Dict[str, np.ndarray]]
        Nodes with ``coordinates`` keys.

    Returns
    -------
    List[Dict[str, np.ndarray]]
        The filtered list of unique nodes.
    """
    unique_nodes: List[Dict[str, np.ndarray]] = []
    for node in nodes:
        c = node["coordinates"]
        if not any(np.allclose(c, un["coordinates"], atol=TOL, rtol=0) for un in unique_nodes):
            unique_nodes.append(node)
    return unique_nodes


class Domain2D:
    """Represents a triangular domain defined by three half‑planes.

    The domain is assumed to be the intersection of three half‑planes
    ``a_i^T x >= b_i`` for ``i=1,2,3``.  Additional hyperplanes may
    subsequently intersect the domain during piecewise integration.  The
    class provides helper methods for generating the triangular
    vertices (step 1), intersecting the domain with one extra
    hyperplane (step 2) or two extra hyperplanes (step 3), along with
    convenience checks for whether points lie inside the domain.
    """

    def __init__(self, boundary_1: np.ndarray, b1: float,
                 boundary_2: np.ndarray, b2: float,
                 boundary_3: np.ndarray, b3: float,
                 tol: float = TOL) -> None:
        self.boundaries: List[np.ndarray] = [np.array(boundary_1, dtype=float),
                                             np.array(boundary_2, dtype=float),
                                             np.array(boundary_3, dtype=float)]
        self.bs: List[float] = [float(b1), float(b2), float(b3)]
        self.tol: float = tol

    # ---------------------------------------------------------------------
    # Domain checks
    # ---------------------------------------------------------------------
    def domain_1(self, x: np.ndarray) -> bool:
        """Return ``True`` if ``x`` is inside the base triangular domain."""
        return (np.dot(self.boundaries[0], x) >= self.bs[0] - self.tol and
                np.dot(self.boundaries[1], x) >= self.bs[1] - self.tol and
                np.dot(self.boundaries[2], x) >= self.bs[2] - self.tol)

    def domain_2(self, x: np.ndarray, w_i: np.ndarray, b_i: float) -> bool:
        """Return ``True`` if ``x`` is inside the domain and satisfies
        the additional half‑plane ``w_i^T x >= b_i``."""
        return self.domain_1(x) and (np.dot(w_i, x) >= b_i - self.tol)

    def domain_3(self, x: np.ndarray, w_i: np.ndarray, b_i: float,
                 w_j: np.ndarray, b_j: float) -> bool:
        """Return ``True`` if ``x`` is inside the domain and satisfies two
        additional half‑planes ``w_i^T x >= b_i`` and ``w_j^T x >= b_j``."""
        return (self.domain_2(x, w_i, b_i) and
                (np.dot(w_j, x) >= b_j - self.tol))

    # ---------------------------------------------------------------------
    # Sequential intersection routines
    # ---------------------------------------------------------------------
    def step_1(self) -> List[Dict[str, np.ndarray]]:
        """Compute the vertices of the base triangular domain.

        Uses ``numpy.linalg.solve`` on each pair of boundary lines
        ``a_i^T x = b_i`` to produce the three vertices of the
        triangle.  Returns a list of node dictionaries containing the
        coordinates and the 2×2 systems solved for each vertex.
        """
        boundary_1, boundary_2, boundary_3 = self.boundaries
        b1, b2, b3 = self.bs
        # Solve pairwise intersections of the three bounding lines
        A12 = np.array([boundary_1, boundary_2])
        b12 = np.array([b1, b2], dtype=float)
        x12 = np.linalg.solve(A12, b12)

        A13 = np.array([boundary_1, boundary_3])
        b13 = np.array([b1, b3], dtype=float)
        x13 = np.linalg.solve(A13, b13)

        A23 = np.array([boundary_2, boundary_3])
        b23 = np.array([b2, b3], dtype=float)
        x23 = np.linalg.solve(A23, b23)

        nodes_1 = [
            {"node": 1, "coordinates": x12, "A": A12, "b": b12},
            {"node": 2, "coordinates": x13, "A": A13, "b": b13},
            {"node": 3, "coordinates": x23, "A": A23, "b": b23},
        ]
        return nodes_1

    def step_2(self, w_i: np.ndarray, b_i: float) -> List[Dict[str, np.ndarray]]:
        """Intersect the domain with a single additional hyperplane.

        Starts with the base triangle (``step_1``), filters out
        vertices lying outside ``w_i^T x >= b_i``, computes the
        intersections between the new hyperplane and each of the three
        bounding lines, and returns the resulting polygon vertices
        sorted in counterclockwise order.  If the sequential approach
        yields fewer than three vertices, the robust ``global_intersection``
        fallback is used.
        """
        nodes_1 = self.step_1()
        nodes_1_filtered = [n for n in nodes_1 if self.domain_2(n["coordinates"], w_i, b_i)]

        # Intersections between the new hyperplane and each boundary line
        valid_intersections: List[Dict[str, np.ndarray]] = []
        boundaries = list(zip(self.boundaries, self.bs))
        for boundary, b_boundary in boundaries:
            A = np.vstack([w_i, boundary])
            try:
                x = np.linalg.solve(A, [b_i, b_boundary])
                if self.domain_1(x):
                    valid_intersections.append({
                        "node": len(valid_intersections) + 4,
                        "coordinates": x,
                        "A": A,
                        "b": [b_i, b_boundary],
                    })
            except np.linalg.LinAlgError:
                # Parallel lines; ignore this intersection
                continue
        merged = nodes_1_filtered + valid_intersections
        unique_nodes = remove_duplicate_coordinates(merged)
        if len(unique_nodes) < 3:
            # Fallback to robust global intersection
            inequalities = list(zip(self.boundaries, self.bs)) + [(w_i, b_i)]
            return global_intersection(inequalities)
        centroid = np.mean([n["coordinates"] for n in unique_nodes], axis=0)
        unique_nodes.sort(key=lambda n: math.atan2(
            n["coordinates"][1] - centroid[1], n["coordinates"][0] - centroid[0]
        ))
        return unique_nodes

    def step_3(self, w_i: np.ndarray, b_i: float, w_j: np.ndarray, b_j: float) -> List[Dict[str, np.ndarray]]:
        """Intersect the domain with two additional hyperplanes.

        This method filters the polygon obtained from ``step_2`` using the
        second hyperplane ``w_j^T x >= b_j`` and computes the new
        intersections between ``w_j`` and each of the existing boundary
        lines and the first hyperplane ``w_i``.  The resulting vertices
        are again sorted counterclockwise.  If fewer than three unique
        vertices are obtained, an empty list is returned, signalling
        there is no polygon to integrate over in this region.
        """
        nodes_2 = self.step_2(w_i, b_i)
        nodes_2_filtered = [n for n in nodes_2 if self.domain_3(n["coordinates"], w_i, b_i, w_j, b_j)]

        valid_intersections: List[Dict[str, np.ndarray]] = []
        # Boundaries include the original three and the first hyperplane
        boundaries = list(zip(self.boundaries, self.bs)) + [(w_i, b_i)]
        node_counter = 7
        for boundary, b_boundary in boundaries:
            A = np.vstack([w_j, boundary])
            try:
                x = np.linalg.solve(A, [b_j, b_boundary])
                if self.domain_3(x, w_i, b_i, w_j, b_j):
                    valid_intersections.append({
                        "node": node_counter,
                        "coordinates": x,
                        "A": A,
                        "b": [b_j, b_boundary],
                    })
                    node_counter += 1
            except np.linalg.LinAlgError:
                continue

        merged = nodes_2_filtered + valid_intersections
        unique_nodes = remove_duplicate_coordinates(merged)
        if len(unique_nodes) < 3:
            return []  # No valid polygon could be formed
        centroid = np.mean([n["coordinates"] for n in unique_nodes], axis=0)
        unique_nodes.sort(key=lambda n: math.atan2(
            n["coordinates"][1] - centroid[1], n["coordinates"][0] - centroid[0]
        ))
        return unique_nodes

    # ---------------------------------------------------------------------
    # Accessors
    # ---------------------------------------------------------------------
    @property
    def boundaries_as_tuples(self) -> List[Tuple[np.ndarray, float]]:
        """Return the boundaries as ``(A, b)`` tuples for external use."""
        return list(zip(self.boundaries, self.bs))
