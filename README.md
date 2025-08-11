# Solving PDEs with Deep Neural Networks

This repo explores a **non-quadrature, closed-form** approach to fitting 2D variational PDEs using a shallow neural network with *Heaviside* activations. We replace all integrals in the least-squares energy by exact **areas of polygonal intersections** inside the domain, so the mass matrix and RHS are assembled analytically (no sampling).

We study the Poisson model problem on the unit right triangle

$$
-\Delta u(x) = f(x)\ \text{in }\Omega,\qquad 
u=0\ \text{on }\partial\Omega,\qquad
\Omega=\{(x,y): x\ge0,\ y\ge0,\ x+y\le 1\}.
$$

## Network parameterisation

We approximate $u$ by a piecewise-constant Heaviside network

$$
u_N(x)=\sum_{i=1}^{N} c_i\,\sigma^k (w_i^\top x + b_i),
$$

with inner parameters $(w_i,b_i)\in\mathbb{R}^2\times\mathbb{R}$ hyperplanes and outer coefficients $c_i\in\mathbb{R}$.
The target $f$ can also be a Heaviside indicator $f(x)=\mathbf 1_{\{w_f^\top x\ge b_f\}}$ (the constant case $f\equiv1$ is handled as a limit).

## Closed-form assembly

Let $H_i=\{x:\,w_i^\top x\ge b_i\}$ and $H_f=\{x:\,w_f^\top x\ge b_f\}$.
The least-squares energy reduces to areas

$$
M_{ij} = \int_{\Omega \cap H_i \cap H_j}dx,\qquad
F_i    = \int_{\Omega \cap H_i \cap H_f}dx,\qquad
A      = \int_{\Omega \cap H_f}dx\quad(\text{since }\mathbf 1^2=\mathbf 1).
$$

Minimizing the energy $I(c)=A-2c^\top F + c^\top M c$ gives the normal equations $Mc=F$, for the inner parameters.

**Linear solve strategy**

1. custom **Cholesky** (SPD),
2. **PLU** if well-conditioned but not SPD,
3. **SVD** pseudo-inverse fallback.

Optionally, optimise inner parameters $(W,B)$ via the reduced objective
$J^\*(W,B)=A - F^\top M^{-1}F$ with a single **BFGS** step per outer iteration.

## Repository layout

```
Code/
  main.py                 # CLI entry point with multiple demo modes
  experiments.py          # One-call figure generators and study runners
  geometry_utils.py       # Triangular domain (half-spaces, intersections)
  area_utils.py           # Exact area formulas inside Ω
  mass_matrix_utils.py    # Build (M, F, A) for Heaviside target and basis
  solver_utils.py         # Cost, Cholesky/PLU/SVD solvers, PD checks
  hyperplane_utils.py     # Generate hyperplanes + plotting helpers
  optimization.py         # HeavisideObjective: reduced objective + outer loop
  print_utils.py          # Pretty console printing of M and F
  sanity_tests.py         # Analytic area/unit tests for Ω
Pictures/                 # Figures saved by the demos
```

> **Heaviside vs ReLU.**
> All derivations and code here use **Heaviside** activations (indicators), *not* ReLU.

## Quick start

### Environment

```bash
# Python >= 3.9
pip install numpy scipy matplotlib tqdm
```

### Sanity checks (analytic areas)

```bash
cd Code
python sanity_tests.py
# Expect: "All sanity tests PASSED ✅"
```

### Run the demos (`main.py`)

```bash
# 1) Small demo: plot hyperplanes, assemble once (target & f ≡ 1)
python Code/main.py --mode demo

# 2) Grid of NN surface plots + ground truth
python Code/main.py --mode surfaces

# 3) Convergence comparison (two targets)
python Code/main.py --mode convergence

# 4) Outer optimisation + diagnostics/contours
python Code/main.py --mode optim --N 2 --wf 1,-1 --bf 0.5 --iters 120 --fit-range 5,100

# Everything in sequence
python Code/main.py --mode all --N 2 --wf 1,-1 --bf 0.5 --iters 120 --fit-range 5,100
```

Figures are saved to `Pictures/` and also displayed.

## What each demo does

* **Demo** (`--mode demo`): plots the domain with uniform hyperplanes for several $N$; assembles $(M,F,A)$ for a Heaviside target and for $f\equiv 1$; solves $Mc=F$; prints diagnostics and positive-definiteness checks.
* **Surfaces** (`--mode surfaces`): for a set of $N$, solves once per $N$, evaluates the NN on a grid over $\Omega$, and shows a grid of 3D surfaces with the ground truth.
* **Convergence** (`--mode convergence`): runs two targets (default $H(y-0.5)$ and $1$) and plots log-log error vs neurons with a fitted slope.
* **Optimisation** (`--mode optim`): runs the outer loop on $(W,B)$; shows loss history, final hyperplanes, order-of-convergence scatter (($W,B$) and $c$), and contour slices of the reduced objective:

  * $(w_x, w_y)$ plane for up to two neurons,
  * normalised planes for neuron 1: $(w_x/\|w\|, b/\|w\|)$ and $(w_y/\|w\|, b/\|w\|)$.

## Reproducing figures via `experiments.py`

```python
from experiments import (
    run_surface_demo,          # grid of surfaces + ground truth
    run_convergence_demo,      # two-target comparison
    run_optimisation_demo      # full optimisation + diagnostics
)

# Example:
run_optimisation_demo(
    N=2, w_f=[1.0, -1.0], b_f=0.5,
    num_outer_iter=120, fit_range=(5, 100)
)
```

## Notes and tips

* Domains and half-spaces use inequalities $a^\top x \ge b$; intersections remain convex and areas are computed by clipping.
* `solver_utils.solve_for_c_custom` tries **Cholesky → PLU → SVD** automatically with conditioning heuristics.
* If you only need Heaviside assembly (no optimisation):

  ```python
  from mass_matrix_utils import build_mass_matrix_unified
  from solver_utils import solve_for_c_custom

  M, F, A = build_mass_matrix_unified(domain, W, B, w_f, b_f)
  c       = solve_for_c_custom(M, F)
  ```
* Change the target $f$ by passing a different $(w_f,b_f)$ to `build_mass_matrix_unified`; omit them for $f\equiv 1$.

## Reference

E, Weinan; Yu, Bing. *The Deep Ritz Method: A Deep Learning-Based Numerical Algorithm for Solving Variational Problems*. **Communications in Mathematics and Statistics** (2018).
[https://link.springer.com/article/10.1007/s40304-018-0127-z](https://link.springer.com/article/10.1007/s40304-018-0127-z)

## License & citation

If you use this code or the generated figures, please cite the repository.
**License:** (MIT).
