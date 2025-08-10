import numpy as np
from typing import Optional, Tuple
from geometry_utils import Domain2D
from area_utils import (
    diagonal_matrix_entry,   # area(domain ∩ {w^T x ≥ b})
    lower_matrix_entry,      # area(domain ∩ {w_i^T x ≥ b_i} ∩ {w_j^T x ≥ b_j})
    compute_F_from_step1     # area(over domain)
)

def build_mass_matrix_unified(domain: "Domain2D", W: np.ndarray, B: np.ndarray,
                              w_f: Optional[np.ndarray] = None, 
                              b_f: Optional[float] = None, zero_atol: float = 1e-10,
                              ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Assemble (M, F, A) for Heaviside basis/target over domain.
    See module header for detailed definition and special cases.
    """
    W = np.asarray(W, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    N = len(W)
    assert W.shape == (N, 2) and B.shape == (N,), "W must be (N,2), B must be (N,)"

    # M (independent of target)
    M = np.zeros((N, N), dtype=np.float64)
    diag_vals = np.array([diagonal_matrix_entry(domain, W[i], B[i]) for i in range(N)], dtype=np.float64)
    np.fill_diagonal(M, diag_vals)
    for i in range(1, N):
        row = [lower_matrix_entry(domain, W[i], B[i], W[j], B[j]) for j in range(i)]
        M[i, :i] = row
        M[:i, i] = row  # symmetric

    # F and A depend on target
    if w_f is None or b_f is None:
        F = diag_vals.reshape(-1, 1)
        A = compute_F_from_step1(domain)
        return M, F, A

    w_f = np.asarray(w_f, dtype=np.float64)
    b_f = float(b_f)

    if np.allclose(w_f, 0.0, atol=zero_atol, rtol=0.0):
        if b_f < 0.0:
            F = diag_vals.reshape(-1, 1)
            A = compute_F_from_step1(domain)
        else:
            F = np.zeros((N, 1), dtype=np.float64)
            A = 0.0
        return M, F, A

    F = np.array([lower_matrix_entry(domain, W[i], B[i], w_f, b_f) for i in range(N)],
                 dtype=np.float64).reshape(-1, 1)
    A = diagonal_matrix_entry(domain, w_f, b_f)
    return M, F, A

