"""
solver_utils.py
================

Helpers for solving the linear systems arising in the 2D Heaviside fitting
and for evaluating the associated cost and gradient.

Solve strategy (robust path)
----------------------------
1) Try a custom **Cholesky** factorization (fastest; requires SPD).
2) If Cholesky fails, estimate the **condition number** κ(A).
   • If A is not ill-conditioned → solve with **PLU** (partial-pivot LU).
   • Otherwise → fall back to **SVD** pseudo-inverse.

Functions
---------
compute_cost(c, M, F, area)
    Value of the reduced energy A - Fᵀc (prints cᵀMc, Fᵀc, A for diagnostics).

compute_cost_grad_c(c, M, F)
    Gradient 2(Mc - F) w.r.t. c (if you include the quadratic part).

clean_solution(c, tol)
    Zero small entries for pretty printing.

is_positive_definite(matrix, tol)
is_positive_definite_cholesky(matrix)
    PD checks via eigenvalues and NumPy’s cholesky.

cholesky(A)
    Custom lower-triangular Cholesky factor (A = L Lᵀ).

lu_decomposition_pivot(A)
    Partial-pivoted LU: returns P, L, U with P A = L U.

solve_for_c_custom(M, F, tol)
    Robust solver: Cholesky → (if needed) PLU → (if needed) SVD.

"""

from __future__ import annotations
import numpy as np

# ----------------------------- Cost & gradient -------------------------------

def compute_cost(c: np.ndarray, M: np.ndarray, F: np.ndarray, area: float) -> float:
    """Return reduced energy A - Fᵀc. Prints diagnostics (cᵀMc, Fᵀc, A)."""
    c = np.asarray(c).reshape(-1, 1)
    F = np.asarray(F).reshape(-1, 1)
    quadratic_term = float(c.T @ M @ c)
    linear_term = float(F.T @ c)
    print(f"cMc: {quadratic_term}")
    print(f"cF: {linear_term}")
    print(f"Area: {area}")
    return area - linear_term


def compute_cost_grad_c(c: np.ndarray, M: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Gradient of (cᵀMc - Fᵀc) is 2(Mc - F)."""
    c = np.asarray(c).reshape(-1, 1)
    F = np.asarray(F).reshape(-1, 1)
    return 2 * (M @ c - F)


def clean_solution(c: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Set |c_i| < tol to 0 (pretty printing)."""
    c = np.asarray(c).copy()
    c[np.abs(c) < tol] = 0.0
    return c

# ----------------------------- PD checks -------------------------------------

def is_positive_definite(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Check PD by eigenvalues of the symmetrized matrix."""
    A = (matrix + matrix.T) / 2.0
    eigvals = np.linalg.eigvalsh(A)
    return np.all(eigvals > tol)


def is_positive_definite_cholesky(matrix: np.ndarray) -> bool:
    """Check PD via NumPy cholesky."""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

# ----------------------------- Cholesky (L) ----------------------------------

def cholesky(A: np.ndarray) -> np.ndarray:
    """Custom lower-triangular Cholesky of SPD A (A = L Lᵀ)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            tmp = np.dot(L[i, :j], L[j, :j])
            if i == j:
                val = A[i, i] - tmp
                if val <= 0:
                    raise np.linalg.LinAlgError("Matrix is not positive definite")
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - tmp) / L[j, j]
    return L

# ----------------------------- PLU (partial pivot) ---------------------------

def lu_decomposition_pivot(A: np.ndarray, pivot_tol: float = 1e-14):
    """
    Partial-pivoted LU factorization.
    Returns P, L, U with P @ A = L @ U.
    L has unit diagonal; P is a permutation matrix.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n, dtype=float)
    P = np.eye(n, dtype=float)

    for k in range(n - 1):
        # pivot row index
        pivot = k + np.argmax(np.abs(U[k:, k]))
        if np.abs(U[pivot, k]) < pivot_tol:
            raise np.linalg.LinAlgError("LU pivot too small (singular or nearly singular).")

        # swap rows in U and P; swap past columns of L
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        # elimination below the pivot
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0

    return P, L, U


def _forward_sub(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve L y = b for lower-triangular L with unit/non-unit diag."""
    n = L.shape[0]
    y = np.zeros_like(b, dtype=float)
    for i in range(n):
        s = b[i] - L[i, :i] @ y[:i]
        y[i] = s / L[i, i]
    return y


def _back_sub(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve U x = y for upper-triangular U."""
    n = U.shape[0]
    x = np.zeros_like(y, dtype=float)
    for i in range(n - 1, -1, -1):
        s = y[i] - U[i, i + 1:] @ x[i + 1:]
        x[i] = s / U[i, i]
    return x

# ----------------------------- Ill-conditioning check ------------------------

def _is_ill_conditioned(A: np.ndarray, cond_thresh: float = 1e12) -> bool:
    """
    Return True if κ₂(A) is large (or infinite/NaN).
    Threshold 1e12 is a typical, conservative cut-off.
    """
    try:
        kappa = np.linalg.cond(A)
    except np.linalg.LinAlgError:
        return True
    return (not np.isfinite(kappa)) or (kappa > cond_thresh)

# ----------------------------- Robust solver ---------------------------------

def solve_for_c_custom(M: np.ndarray, F: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Solve M c = F robustly:
      1) Try custom Cholesky (SPD fast path).
      2) If that fails:
         • If M is not ill-conditioned, use PLU with partial pivoting.
         • Else, fall back to SVD pseudo-inverse.

    Returns c with shape (N,1).
    """
    M = np.asarray(M, dtype=float)
    F = np.asarray(F, dtype=float).reshape(-1, 1)
    n = M.shape[0]

    # --- Path 1: Cholesky (SPD) ---
    try:
        L = cholesky(M)
        # Solve L y = F
        y = np.zeros((n, 1), dtype=float)
        for i in range(n):
            y[i, 0] = (F[i, 0] - np.dot(L[i, :i], y[:i, 0])) / L[i, i]
        # Solve Lᵀ c = y
        c = np.zeros((n, 1), dtype=float)
        for i in range(n - 1, -1, -1):
            c[i, 0] = (y[i, 0] - np.dot(L[i + 1:, i], c[i + 1:, 0])) / L[i, i]
        return c
    except Exception as e:
        print("Custom Cholesky failed:", e)

    # --- Path 2: PLU if not ill-conditioned ---
    if not _is_ill_conditioned(M):
        try:
            P, L, U = lu_decomposition_pivot(M)
            # Solve L y = P F
            PF = P @ F
            y = _forward_sub(L, PF)
            # Solve U c = y
            c = _back_sub(U, y)
            return c.reshape(-1, 1)
        except Exception as e:
            print("PLU solve failed:", e)

    # --- Path 3: SVD pseudo-inverse (most robust) ---
    U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_inv = np.array([1.0 / x if x > tol else 0.0 for x in s], dtype=float)
    M_pinv = (Vt.T * s_inv) @ U_svd.T
    return (M_pinv @ F).reshape(-1, 1)
