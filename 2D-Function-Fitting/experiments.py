"""
experiments.py
==============

Reproducible experiments and figure generation for the 2D Heaviside-based
function fitting project.

What this module provides
-------------------------
Helpers:
  - make_domain()                     → standard unit triangle domain Ω
  - make_grid(n)                      → plotting grid + Ω mask
  - triangle_boundary()               → triangle polyline for overlays
  - eval_heaviside_nn(W, B, c, X, Y)  → evaluate NN sum of Heavisides
  - loglog_fit(neurons, errors)       → slope/intercept + fitted curve

Core study:
  - run_series(domain, w_f, b_f, n_values, grid_n)
      For each N in n_values:
        • assemble (M, F, A) with Heaviside target,
        • solve M c = F (custom Cholesky + cleanup),
        • compute energy error,
        • evaluate NN on a grid (masked outside Ω).

Plotting:
  - plot_surfaces(approx_data, X, Y, fname)
  - plot_ground_truth(w_f, b_f, X, Y, mask, fname)
  - plot_convergence(...), plot_convergence_comparison(...)
  - plot_norm_planes_neuron0(...)   # normalized (w/||w||, b/||w||) planes for neuron 0


High-level demos (one call from main.py):
  - run_surface_demo()
  - run_convergence_demo()
  - run_optimisation_demo(N=2, w_f=[1,-1], b_f=0.5, num_outer_iter=120, fit_range=(5,100))

Notes
-----
• All integrals/areas follow Heaviside indicator semantics H(wᵀx - b).
• Depends on:
    geometry_utils.Domain2D
    hyperplane_utils.generate_uniform_hyperplanes (+ plotting helpers)
    mass_matrix_utils.build_mass_matrix_unified
    solver_utils.solve_for_c_custom / clean_solution / compute_cost
    optimization.HeavisideObjective / unpack_inner_params
• Designed so you can generate publication-ready figures with a single call.
"""

import numpy as np
import matplotlib.pyplot as plt

from geometry_utils import Domain2D
from mass_matrix_utils import build_mass_matrix_unified
from solver_utils import solve_for_c_custom, clean_solution, compute_cost
from optimization import HeavisideObjective, unpack_inner_params
from hyperplane_utils import (
        plot_loss_vs_iterations, plot_hyperplanes,
        plot_regression_order_split, plot_weight_contour,
        plot_norm_planes_neuron0, smart_axis_format,
        generate_uniform_hyperplanes
    )



# ---------- helpers -----------------------------------------------------------

def make_domain() -> Domain2D:
    # Ω = {(x,y): x>=0, y>=0, x+y<=1}
    return Domain2D(np.array([1.0, 0.0]), 0.0,
                    np.array([0.0, 1.0]), 0.0,
                    np.array([-1.0,-1.0]), -1.0)

def triangle_boundary():
    verts = np.array([[0,0],[1,0],[0,1],[0,0]])
    return verts[:,0], verts[:,1], np.zeros(4)

def make_grid(n=200):
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, y)
    # mask for Ω (fast explicit form for our Ω)
    mask = (X >= 0) & (Y >= 0) & (X + Y <= 1)
    return X, Y, mask

def eval_heaviside_nn(W: np.ndarray, B: np.ndarray, c: np.ndarray, X, Y):
    # Sum_i c_i * H(w_i^T [x,y] - b_i). Boundary value at 0 is 0.
    Z = np.zeros_like(X, dtype=float)
    for i in range(len(B)):
        Z += c[i] * np.heaviside(W[i,0]*X + W[i,1]*Y - B[i], 0.0)
    return Z

def loglog_fit(neurons: np.ndarray, errors: np.ndarray):
    valid = errors > 0
    if np.count_nonzero(valid) < 2:
        return np.nan, np.nan, np.full_like(errors, np.nan, dtype=float)
    logN, logE = np.log10(neurons[valid]), np.log10(errors[valid])
    slope, intercept = np.polyfit(logN, logE, 1)
    fit = np.full_like(errors, np.nan, dtype=float)
    fit[valid] = 10**(intercept + slope*np.log10(neurons[valid]))
    return slope, intercept, fit

# ---------- core studies ------------------------------------------------------

def run_series(domain: Domain2D, w_f: np.ndarray, b_f: float,
               n_values, grid_n=200):
    """
    For each N in n_values:
      - build M,F,A for Heaviside target,
      - solve for c,
      - compute error (energy) = A - 2 c^T F + c^T M c,
      - evaluate NN on a plotting grid (masked outside Ω).
    Returns:
      approx_data: list of (Z, N, error)
      results: dict {'neurons': array, 'error': array}
    """
    X, Y, mask = make_grid(grid_n)
    approx_data = []
    neurons, errors = [], []

    for N in n_values:
        W, B = generate_uniform_hyperplanes(N)
        M, F, A = build_mass_matrix_unified(domain, W, B, w_f, b_f)
        c = clean_solution(solve_for_c_custom(M, F))

        err = float(np.asarray(compute_cost(c, M, F, A)).squeeze())
        Z = eval_heaviside_nn(W, B, c.ravel(), X, Y)
        Z[~mask] = np.nan

        approx_data.append((Z, N, err))
        neurons.append(N)
        errors.append(err)

    return approx_data, {"neurons": np.array(neurons, int),
                         "error":  np.array(errors, float)}

# ---------- plotting ----------------------------------------------------------

def plot_surfaces(approx_data, X, Y, fname="nn_surfaces.png"):
    n_plots = len(approx_data)
    n_cols = min(4, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig = plt.figure(figsize=(4*n_cols, 4*n_rows), dpi=150)
    tx, ty, tz = triangle_boundary()

    for idx, (Z, N, err) in enumerate(approx_data):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.plot(tx, ty, tz, 'k-', linewidth=2)
        ax.set_title(f'N = {N}\nError = {err:.2e}', fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('NN')
        fig.colorbar(surf, ax=ax, shrink=0.55, aspect=10)

    fig.suptitle('Approximated Heaviside NN Surfaces', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(fname, dpi=300)
    plt.show()

def plot_ground_truth(w_f: np.ndarray, b_f: float, X, Y, mask,
                      fname="ground_truth_heaviside.png"):
    Z_true = np.heaviside(w_f[0]*X + w_f[1]*Y - b_f, 0.0)
    Z_true[~mask] = np.nan
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(1,1,1, projection='3d')
    surf = ax.plot_surface(X, Y, Z_true, cmap='coolwarm', edgecolor='none', alpha=0.9)
    tx, ty, tz = triangle_boundary()
    ax.plot(tx, ty, tz, 'k-', linewidth=2)
    ax.set_title('Ground Truth Heaviside', fontsize=14)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x)')
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()

def plot_convergence(neurons, errors, slope, intercept, fit,
                     label_data, label_fit, fname="convergence_plot.png"):
    fig = plt.figure(figsize=(8, 6), dpi=150)
    plt.loglog(neurons, errors, 'o-', linewidth=2, markersize=6, label=label_data)
    if np.isfinite(slope):
        plt.loglog(neurons, fit, '--', linewidth=2,
                   label=fr'{label_fit}: $\log_{{10}}E={slope:.2f}\log_{{10}}N{intercept:+.2f}$')
    plt.xlabel('Number of Neurons (log10)')
    plt.ylabel(r'$L^2$ Error / Energy (log10)')
    plt.title('Convergence of NN Approximation', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()

def plot_convergence_comparison(neurons1, err1, s1, b1, fit1,
                                neurons2, err2, s2, b2, fit2,
                                label1, label2,
                                fname="convergence_comparison.png"):
    plt.figure(figsize=(8, 6), dpi=150)
    plt.loglog(neurons1, err1, 'o', markersize=6, label=label1)
    if np.isfinite(s1):
        plt.loglog(neurons1, fit1, '--', alpha=0.8,
                   label=fr'{label1} fit: $\log_{{10}}E={s1:.2f}\log_{{10}}N{b1:+.2f}$')
    plt.loglog(neurons2, err2, 's', markersize=6, label=label2)
    if np.isfinite(s2):
        plt.loglog(neurons2, fit2, '--', alpha=0.8,
                   label=fr'{label2} fit: $\log_{{10}}E={s2:.2f}\log_{{10}}N{b2:+.2f}$')
    plt.xlabel('Number of Neurons (log10)')
    plt.ylabel(r'$L^2$ Error / Energy (log10)')
    plt.title('Convergence Comparison', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.show()

# ---------- high-level recipes you can call from main -------------------------

def run_surface_demo():
    """Reproduce the surface grid + ground-truth plot for one target."""
    domain = make_domain()
    # Example target 1: H(y - 0.5)
    w_f, b_f = np.array([0.0, 1.0]), 0.5
    n_values = [8, 16, 32, 64]

    approx_data, _ = run_series(domain, w_f, b_f, n_values, grid_n=200)
    X, Y, mask = make_grid(200)
    plot_surfaces(approx_data, X, Y, fname="nn_surfaces_heaviside_y_ge_0p5.png")
    plot_ground_truth(w_f, b_f, X, Y, mask, fname="ground_truth_y_ge_0p5.png")

def run_convergence_demo():
    """Run two targets and compare convergence curves."""
    domain = make_domain()
    X, Y, mask = make_grid(200)

    # Target A: H(y - 0.5)
    wA, bA = np.array([0.0, 1.0]), 0.5
    n_vals = [4, 8, 16, 32, 64, 128]
    _, resA = run_series(domain, wA, bA, n_vals, grid_n=200)
    sA, bIA, fitA = loglog_fit(resA["neurons"], resA["error"])

    # Target B: H(x + y - 0) == 1 on Ω (constant 1)
    wB, bB = np.array([1.0, 1.0]), 0.0
    _, resB = run_series(domain, wB, bB, n_vals, grid_n=200)
    sB, bIB, fitB = loglog_fit(resB["neurons"], resB["error"])

    plot_convergence_comparison(
        resA["neurons"], resA["error"], sA, bIA, fitA,
        resB["neurons"], resB["error"], sB, bIB, fitB,
        label1=r'$f_1(\mathbf{x}) = H(y-0.5)$',
        label2=r'$f_2(\mathbf{x}) \equiv 1$',
        fname="convergence_comparison.png"
    )
    
def run_optimisation_demo(N=2,
                          w_f=None,
                          b_f=0.5,
                          num_outer_iter=120,
                          fit_range=(5, 100)):
    """
    One-call demo:
      - domain: unit right triangle  {x>=0, y>=0, x+y<=1}
      - target: Heaviside(w_f^T x - b_f)
      - runs outer optimisation and produces all diagnostic plots

    Returns a dict with the final params and histories.
    """
    if w_f is None:
        w_f = np.array([1.0, -1.0], dtype=float)

    domain = Domain2D(np.array([1.0, 0.0]), 0.0,
                      np.array([0.0, 1.0]), 0.0,
                      np.array([-1.0, -1.0]), -1.0)

    obj = HeavisideObjective(domain, N, np.asarray(w_f, float), float(b_f))
    theta_final, cost_hist, c_hist, W_hist, b_hist = obj.run_outer(num_outer_iter=num_outer_iter)
    best_W, best_b = unpack_inner_params(theta_final, N)

    # Loss curve + final hyperplanes
    plot_loss_vs_iterations(cost_hist)
    plot_hyperplanes(best_W, best_b)

    # Order-of-convergence
    plot_regression_order_split(
        W_hist, b_hist, c_hist,
        W_true=np.tile(np.asarray(w_f, float), (N, 1)),
        b_true=np.full(N, float(b_f)),
        fit_range=fit_range,
        robust=True
    )

    # Contour slices around the training path
    neuron_ids = list(range(min(N, 2)))
    if neuron_ids:
        # (w_x, w_y) planes for up to two neurons
        fig, axes = plt.subplots(1, len(neuron_ids), figsize=(6*len(neuron_ids), 5),
                                 constrained_layout=True)
        if len(neuron_ids) == 1:
            axes = [axes]
        cfs = []
        for ax, idx in zip(axes, neuron_ids):
            cf = plot_weight_contour(ax, obj, W_hist, best_W, best_b, neuron_idx=idx)
            cfs.append(cf)
        # Apply nice tick formatting on both axes for each subplot
        for ax in axes:
            smart_axis_format(ax, 'x')
            smart_axis_format(ax, 'y')
        fig.colorbar(cfs[0], ax=axes, shrink=0.8, label=r'$\log_{10}\mathcal{L}$')
        plt.show()

        # Normalised planes for neuron 0 only: (w_x/||w||, b/||w||) and (w_y/||w||, b/||w||)
        fig, axes2 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        cf_left, cf_right = plot_norm_planes_neuron0(
            axes2[0], axes2[1], obj,
            W_hist, b_hist, best_W, best_b
        )
        # The plotting function already applies smart_axis_format internally
        fig.colorbar(cf_left, ax=axes2, shrink=0.8, label=r'$\log_{10}\mathcal{L}$')
        plt.show()

    return {
        "theta_final": theta_final,
        "best_W": best_W,
        "best_b": best_b,
        "cost_history": cost_hist,
        "c_history": c_hist,
        "W_history": W_hist,
        "b_history": b_hist,
        "domain": domain,
        "w_f": np.asarray(w_f, float),
        "b_f": float(b_f),
    }