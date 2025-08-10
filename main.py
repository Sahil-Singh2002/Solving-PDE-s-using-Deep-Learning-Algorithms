"""
main.py
=======

Entrypoint for 2D Heaviside fitting demos.

Modes
-----
  demo          : plot domain hyperplanes, assemble/solve once (target & f≡1)
  surfaces      : grid of NN surfaces + ground truth (experiments.py)
  convergence   : two-target convergence comparison (experiments.py)
  optim         : run outer optimisation and all diagnostics/contours (experiments.py)

Examples
--------
python main.py --mode demo
python main.py --mode optim --N 2 --wf "1,-1" --bf 0.5 --iters 120
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from geometry_utils import Domain2D
from hyperplane_utils import (plot_triangle_with_hyperplanes,generate_uniform_hyperplanes,
)

from mass_matrix_utils import build_mass_matrix_unified
from solver_utils import (
    solve_for_c_custom,
    clean_solution,
    compute_cost,
    is_positive_definite,
    is_positive_definite_cholesky,
)
from print_utils import pretty_print_matrix_and_vector
from experiments import (run_surface_demo, run_convergence_demo,
                         run_optimisation_demo,)

def quick_diagnostics(M, F, A, c, tag, tol=1e-10):
    r = np.linalg.norm(M @ c - F)
    L_full = (A - 2*c.T@F + c.T@M@c).item()
    L_red  = (A - (c.T@F)).item()  # valid at optimum if Mc=F
    bounds_ok = np.all(F.ravel() <= np.diag(M) + tol) and np.all(F >= -tol)
    print(f"[{tag}] ||Mc-F|| = {r:.2e}")
    print(f"[{tag}] L_full    = {L_full:.6g}   L_reduced = {L_red:.6g}   diff = {abs(L_full-L_red):.2e}")
    print(f"[{tag}] F bounds  : {'OK' if bounds_ok else 'VIOLATION'}")

def run_demo() -> None:
    """Small self-contained demo for the Heaviside setting."""
    # Domain Ω = {(x,y): x≥0, y≥0, x+y≤1}
    domain = Domain2D(np.array([1.0, 0.0]), 0.0,
                      np.array([0.0, 1.0]), 0.0,
                      np.array([-1.0, -1.0]), -1.0)

    # Heaviside target parameters: f(x) = H(w_f^T x - b_f)
    w_f, b_f = np.array([1.0, -1.0]), 0.0

    # Plot triangle + hyperplanes for multiple n
    n_values = [2, 4, 8, 16, 32, 64, 128, 256]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for idx, n in enumerate(n_values):
        plot_triangle_with_hyperplanes(n, axes[idx])
        axes[idx].set_title(f"n = {n}")
    for ax in axes[len(n_values):]:
        ax.axis('off')
    plt.tight_layout()
    plt.suptitle("Triangle Domain with Uniform Hyperplanes", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Assemble/solve for a specific n
    n_example = 8
    W, B = generate_uniform_hyperplanes(n_example)

    # --- Heaviside target ---
    M0, F0, A0 = build_mass_matrix_unified(domain, W, B, w_f, b_f)
    pretty_print_matrix_and_vector("MassMatrix M (target)", "Projection vector F", M0, F0)
    print(f"\nTarget area (Heaviside ∫Ω H): {A0:.6f}")
    c0 = clean_solution(solve_for_c_custom(M0, F0))
    print("Cleaned Outer Parameter (target):")
    print(c0)
    print("==========================")
    print("Final Cost (target):", compute_cost(c0, M0, F0, A0))
    print("Positive Definiteness (eigs):", is_positive_definite(M0))
    print("Positive Definiteness (cholesky):", is_positive_definite_cholesky(M0))
    quick_diagnostics(M0, F0, A0, c0, tag="Heaviside target")

    # --- f ≡ 1 on Ω ---
    M1, F1, A1 = build_mass_matrix_unified(domain, W, B)  # no w_f,b_f → constant 1
    pretty_print_matrix_and_vector("MassMatrix M (f ≡ 1)", "Projection vector F", M1, F1)
    print(f"\nArea of Ω: {A1:.6f}")
    c1 = clean_solution(solve_for_c_custom(M1, F1))
    print("Cleaned Outer Parameter (f ≡ 1):")
    print(c1)
    print("==========================")
    print("Final Cost (f ≡ 1):", compute_cost(c1, M1, F1, A1))
    print("Positive Definiteness (eigs):", is_positive_definite(M1))
    print("Positive Definiteness (cholesky):", is_positive_definite_cholesky(M1))
    quick_diagnostics(M1, F1, A1, c1, tag="f ≡ 1")

def parse_vec2(text: str) -> np.ndarray:
    # parse "a,b" → np.array([a,b])
    arr = np.fromstring(text, sep=',')
    if arr.size != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated numbers, e.g. '1,-1'.")
    return arr

def main():
    ap = argparse.ArgumentParser(description="2D Heaviside fitting demos")
    ap.add_argument("--mode",
                    choices=["demo", "surfaces", "convergence", "optim", "all"],
                    default="demo",
                    help="which demo(s) to run")
    ap.add_argument("--N", type=int, default=2, help="number of neurons (optim mode)")
    ap.add_argument("--wf", type=parse_vec2, default=None, help="target weights 'a,b' (optim mode)")
    ap.add_argument("--bf", type=float, default=0.5, help="target bias (optim mode)")
    ap.add_argument("--iters", type=int, default=120, help="outer iterations (optim mode)")
    ap.add_argument("--fit-range", type=str, default="5,100", help="kmin,kmax for order fit (optim mode)")
    args, _ = ap.parse_known_args()

    def run_optim():
        kmin, kmax = (int(x) for x in args.fit_range.split(","))
        res = run_optimisation_demo(
            N=args.N,
            w_f=args.wf,
            b_f=args.bf,
            num_outer_iter=args.iters,
            fit_range=(kmin, kmax),
        )
        print("\n--- optimisation summary ---")
        print("best_W:\n", res["best_W"])
        print("best_b:\n", res["best_b"])

    if args.mode == "demo":
        run_demo()
        return

    if args.mode == "surfaces":
        run_surface_demo()
        return

    if args.mode == "convergence":
        run_convergence_demo()
        return

    if args.mode == "optim":
        run_optim()
        return

    # --- args.mode == "all" ---
    # Run everything in sequence so all figures are produced:
    print("[all] Running 'demo'…")
    run_demo()
    print("[all] Running 'surfaces'…")
    run_surface_demo()
    print("[all] Running 'convergence'…")
    run_convergence_demo()
    print("[all] Running 'optim'…")
    run_optim()


if __name__ == "__main__":
    main()

 
"""
%run main.py --mode all
%run main.py --mode all --N 2 --wf 1,-1 --bf 0.5 --iters 120 --fit-range 5,100
"""

