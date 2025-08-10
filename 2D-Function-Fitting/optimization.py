# optimization.py
import numpy as np
from typing import Optional, Tuple
from scipy.optimize import minimize

from geometry_utils import Domain2D
from hyperplane_utils import generate_uniform_hyperplanes
from mass_matrix_utils import build_mass_matrix_unified
from solver_utils import solve_for_c_custom, clean_solution, compute_cost

TOL = 1e-10
EPS = 1e-8

# ---- pack / unpack -----------------------------------------------------------
def pack_inner_params(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([W.ravel(), b.ravel()])

def unpack_inner_params(theta: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    W_flat = theta[:2*N]
    W = W_flat.reshape(N, 2)
    b = theta[2*N:3*N]
    return W, b

# ---- core objective ----------------------------------------------------------
class HeavisideObjective:
    """
    Reduced objective for Heaviside basis:
      cost(W,b) = A - F^T c*,   with M c* = F

    where (M, F, A) come from build_mass_matrix_unified(domain, W, B, w_f, b_f).
    """
    def __init__(self, domain: Domain2D, N: int, w_f: np.ndarray, b_f: float,
                 tol: float = TOL, eps: float = EPS):
        self.domain = domain
        self.N = N
        self.w_f = np.asarray(w_f, float)
        self.b_f = float(b_f)
        self.tol = tol
        self.eps = eps

    def assemble(self, W: np.ndarray, B: np.ndarray):
        return build_mass_matrix_unified(self.domain, W, B, self.w_f, self.b_f)

    def solve_c(self, M: np.ndarray, F: np.ndarray) -> np.ndarray:
        return clean_solution(solve_for_c_custom(M, F, tol=self.tol))

    def reduced_objective(self, theta: np.ndarray) -> float:
        W, B = unpack_inner_params(theta, self.N)
        M, F, A = self.assemble(W, B)
        try:
            c = self.solve_c(M, F)
        except Exception:
            return 1e6
        # energy at optimum: A - F^T c
        return float(A - (F.T @ c).item())

    def grad_fd(self, theta: np.ndarray) -> np.ndarray:
        g = np.zeros_like(theta)
        f0 = self.reduced_objective(theta)
        for i in range(theta.size):
            th = theta.copy()
            th[i] += self.eps
            g[i] = (self.reduced_objective(th) - f0) / self.eps
        return g

    def single_bfgs_step(self, theta: np.ndarray):
        res = minimize(self.reduced_objective, theta,
                       method="BFGS",
                       jac=lambda t: self.grad_fd(t),
                       options={"maxiter": 1, "disp": False})
        return res.x, float(res.fun)

    # ----- outer loop (with early stopping) -----------------------------------
    def run_outer(self, num_outer_iter: int,
                  W0: Optional[np.ndarray] = None,
                  B0: Optional[np.ndarray] = None):
        if W0 is None or B0 is None:
            if self.N == 1:
                W0 = np.array([[1.0, 0.0]])
                B0 = np.array([0.5])
            else:
                W0, B0 = generate_uniform_hyperplanes(self.N)

        theta = pack_inner_params(W0, B0)

        cost_hist, c_hist, W_hist, b_hist = [], [], [], []
        prev = np.inf

        for k in range(num_outer_iter):
            cost = self.reduced_objective(theta)
            W, B = unpack_inner_params(theta, self.N)
            M, F, A = self.assemble(W, B)
            c = self.solve_c(M, F)

            cost_hist.append(cost)
            c_hist.append(c.copy())
            W_hist.append(W.copy())
            b_hist.append(B.copy())

            print(f"[outer {k}] cost = {cost:.3e}")
            if abs(cost - prev) < self.tol:
                print(f"Early stop: |Δcost| = {abs(cost - prev):.2e} < {self.tol}")
                break
            prev = cost

            theta, _ = self.single_bfgs_step(theta)

        return theta, cost_hist, c_hist, W_hist, b_hist
