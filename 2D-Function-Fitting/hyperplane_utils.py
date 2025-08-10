"""
hyperplane_utils.py
====================

Utilities for:
  - generating families of uniformly spaced hyperplanes inside the unit triangle, and
  - visualising both the geometry (domain + hyperplanes) and optimisation diagnostics
    (loss curves, regression-order estimates, and loss contours in parameter slices).

This keeps all plotting in one place so the core geometry/integration code stays lean.

Contents
--------
Generation:
  - generate_uniform_hyperplanes(n)
      Returns (W, B) alternating vertical (x = c) and horizontal (y = c) lines.
  - plot_triangle_with_hyperplanes(n, ax)
      Draw the unit triangle and the n generated hyperplanes on a Matplotlib axis.

Optimisation visuals (Heaviside setting):
  - plot_loss_vs_iterations(cost_history)
      Log–log loss (energy) vs iteration.
  - plot_hyperplanes(W, b)
      Draw arbitrary hyperplanes (W, b) over the triangle.
  - plot_regression_order_split(weight_history, biases_history, c_history, ...)
      Two-panel figure estimating the order of convergence for inner (W, b) and outer c.
  - plot_weight_contour(ax, obj, weight_history, best_W, best_b, ...)
      Filled contour of log10(loss) in (w_x, w_y) for a chosen neuron; overlays the training path.
      Expects `obj` to provide `reduced_objective(theta)` for packed (W, b).
  - plot_intercept_slope_contour(ax, obj, weight_history, biases_history, best_W, best_b, ...)
      Filled contour of log10(loss) in (b/w_x, w_y/w_x) with the training path.

Notes
-----
- All visuals are compatible with the Heaviside indicator basis used elsewhere.
- The contour helpers evaluate the reduced objective through a provided object `obj`
  (e.g., an instance of `HeavisideObjective` from optimization.py) to keep concerns separate.
- SciPy is optional. If unavailable, `plot_regression_order_split` falls back to numpy polyfit.

"""


from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FuncFormatter
from scipy.stats import theilslopes

def _pack_inner_params_local(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Local packer to avoid circular import: (W,b) -> theta."""
    return np.concatenate([W.ravel(), b.ravel()])



def generate_uniform_hyperplanes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ``n`` uniformly spaced hyperplanes within the unit triangle.

    When ``n == 1`` a single vertical line at ``x = 0.5`` is returned.
    For even ``n > 1`` the hyperplanes alternate between vertical and
    horizontal and are evenly spaced so as to partition the domain.
    The returned arrays ``W`` and ``B`` satisfy ``W[i]·x = B[i]`` for
    each hyperplane.

    Parameters
    ----------
    n : int
        The number of hyperplanes to generate.  Must be even when
        greater than one.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple ``(W, B)`` where ``W`` is an ``n×2`` array of normal
        vectors and ``B`` is a length ``n`` array of offsets.
    """
    if n == 1:
        return np.array([[1.0, 0.0]]), np.array([0.5])

    if n % 2 != 0:
        raise ValueError("Number of hyperplanes (n) must be even for n > 1.")

    # Evenly spaced positions excluding boundaries of the triangle
    lines = np.linspace(0, 1, (n // 2) + 2)[1:-1]

    # Prepare vertical (x = c) and horizontal (y = 1 - c) components
    W_vertical = np.tile([1.0, 0.0], (len(lines), 1))
    B_vertical = lines
    W_horizontal = np.tile([0.0, 1.0], (len(lines), 1))
    B_horizontal = 1.0 - lines

    # Interleave vertical and horizontal entries
    W = np.empty((n, 2), dtype=float)
    B = np.empty(n, dtype=float)
    W[0::2], W[1::2] = W_vertical, W_horizontal
    B[0::2], B[1::2] = B_vertical, B_horizontal
    return W, B


def plot_triangle_with_hyperplanes(n: int, ax: plt.Axes) -> None:
    """Plot the unit triangular domain together with generated hyperplanes.

    Parameters
    ----------
    n : int
        Number of hyperplanes to generate and plot.
    ax : matplotlib.axes.Axes
        The axes on which to draw the triangle and hyperplanes.

    Notes
    -----
    This function does not shade the triangle, it only draws the
    boundary and the hyperplanes.  Use tight_layout/suptitle
    externally if arranging multiple subplots.
    """
    W, B = generate_uniform_hyperplanes(n)
    # Plot the triangle
    triangle = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2, label='Triangle Domain')
    # Plot hyperplanes: vertical lines (x = c) and horizontal lines (y = c)
    for w, b in zip(W, B):
        if np.array_equal(w, [1.0, 0.0]):
            # Vertical hyperplane: x = b
            ax.axvline(x=b, color='blue', linestyle='--', alpha=0.7)
        elif np.array_equal(w, [0.0, 1.0]):
            # Horizontal hyperplane: y = b
            ax.axhline(y=b, color='red', linestyle='--', alpha=0.7)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"n = {n} Hyperplanes")
    ax.grid(True, linestyle='--', alpha=0.5)


# -------- simple plots --------------------------------------------------------
def plot_loss_vs_iterations(cost_history):
    it = np.arange(len(cost_history))
    plt.figure(figsize=(8,4))
    plt.loglog(it + 1, np.maximum(cost_history, 1e-16), 'o-')
    plt.xlabel("Iteration (log)")
    plt.ylabel(r"$L^2$ Error / Energy (log)")
    plt.title("Loss vs Iteration")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

def plot_hyperplanes(W, b):
    tri = np.array([[0,0],[1,0],[0,1],[0,0]])
    plt.figure(figsize=(6,6))
    plt.plot(tri[:,0], tri[:,1], 'k-', lw=2, label="Domain")
    x = np.linspace(0,1,200)
    for i in range(len(b)):
        wx, wy = W[i]
        bi = b[i]
        if abs(wy) > 1e-12:
            y = (bi - wx*x)/wy
            m = (x>=0)&(y>=0)&(x+y<=1)
            plt.plot(x[m], y[m], '--')
        elif abs(wx) > 1e-12:
            xv = bi/wx
            yv = np.linspace(0,1,200)
            m = (xv>=0)&(xv<=1)&(xv+yv<=1)
            plt.plot(np.full_like(yv[m], xv), yv[m], '--')
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel('x'); plt.ylabel('y'); plt.title('Hyperplanes')
    plt.grid(True); plt.tight_layout(); plt.show()

# -------- regression-order figure --------------------------------------------
def plot_regression_order_split(weight_history, biases_history, c_history,
                                W_true=None, b_true=None, c_true=None,
                                fit_range: Optional[Tuple[int,int]] = None,
                                drop_plateau=True, plateau_tol=1e-10,
                                min_error=1e-14, robust=False, pad_frac=0.08):
    """Two-panel plot: order estimate for inner (W,b) and outer c."""
    inner_vecs = np.vstack([np.concatenate([W.ravel(), b.ravel()])
                            for W,b in zip(weight_history, biases_history)])
    tgt_inner = (np.concatenate([W_true.ravel(), b_true.ravel()])
                 if (W_true is not None and b_true is not None) else inner_vecs[-1])
    err_in = np.linalg.norm(inner_vecs - tgt_inner, axis=1)

    c_mat = np.vstack([c.ravel() for c in c_history])
    tgt_c = c_mat[-1] if c_true is None else np.asarray(c_true).ravel()
    err_c = np.linalg.norm(c_mat - tgt_c, axis=1)

    K = len(err_in) - 1
    idx = np.arange(K)

    keep = np.ones_like(idx, bool)
    if fit_range is not None:
        k0,k1 = fit_range
        keep &= (idx>=k0)&(idx<=k1)
    if drop_plateau:
        keep &= (err_in[:-1]>plateau_tol) & (err_c[:-1]>plateau_tol)
    keep &= (err_in[:-1]>min_error) & (err_c[:-1]>min_error)
    if keep.sum() < 4:
        print("Too few points for a reliable fit."); return

    x_in, y_in = np.log(err_in[:-1][keep]), np.log(err_in[1:][keep])
    x_c , y_c  = np.log(err_c  [:-1][keep]), np.log(err_c  [1:][keep])

    if robust:
        p_in, a_in, *_ = theilslopes(y_in, x_in)
        p_c , a_c , *_ = theilslopes(y_c , x_c )
    else:
        p_in, a_in = np.polyfit(x_in, y_in, 1)
        p_c , a_c  = np.polyfit(x_c , y_c , 1)

    def pad(l, r):
        lo, hi = min(l.min(), r.min()), max(l.max(), r.max())
        s = hi - lo
        return lo - pad_frac*s, hi + pad_frac*s

    fig, axes = plt.subplots(1,2, figsize=(12,5), constrained_layout=True)

    ax = axes[0]
    ax.scatter(x_in, y_in, s=20, alpha=.7)
    xx = np.linspace(*pad(x_in, x_in), 200); ax.plot(xx, p_in*xx + a_in, lw=2)
    ax.set_xlim(*pad(x_in, x_in)); ax.set_ylim(*pad(y_in, y_in))
    ax.set_title('Inner (W,b)'); ax.set_xlabel('ln||e_k||'); ax.set_ylabel('ln||e_{k+1}||'); ax.grid(True)

    ax = axes[1]
    ax.scatter(x_c, y_c, s=20, alpha=.7, color='orange')
    xx = np.linspace(*pad(x_c, x_c), 200); ax.plot(xx, p_c*xx + a_c, color='orange', lw=2)
    ax.set_xlim(*pad(x_c, x_c)); ax.set_ylim(*pad(y_c, y_c))
    ax.set_title('Outer c'); ax.set_xlabel('ln||e_k||'); ax.grid(True)

    fig.suptitle('Estimated order of convergence')
    plt.show()
    
def _robust_contourf(ax, X, Y, Z, *, levels=30, cmap='viridis', rasterize=True):
    """
    Safe contourf:
      • ignores non-finite values
      • chooses robust vmin/vmax
      • guarantees strictly increasing level vector
      • applies rasterization to collections (no kwarg warning)
    """
    A = np.array(Z, dtype=float)
    mask = np.isfinite(A)
    if not np.any(mask):
        ax.text(0.5, 0.5, "no finite values", transform=ax.transAxes,
                ha='center', va='center', fontsize=10)
        return None

    vmin = float(np.nanmin(A[mask]))
    vmax = float(np.nanmax(A[mask]))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:                 # flat field → pad a hair
        vmax = vmin + 1e-9

    nlev = int(levels) if isinstance(levels, (int, np.integer)) else 30
    nlev = max(nlev, 2)
    levs = np.linspace(vmin, vmax, nlev)

    cf = ax.contourf(X, Y, A, levels=levs, cmap=cmap)
    if rasterize and cf is not None:
        for coll in cf.collections:
            coll.set_rasterized(True)
    return cf


# -------- loss contours in (w_x, w_y) ----------------------------------------
def plot_weight_contour(ax, obj, weight_history, best_W, best_b,
                        neuron_idx=0, grid_pts=120, pad_frac=0.15):
    w_path = np.array([W[neuron_idx] for W in weight_history])
    rng = np.ptp(w_path, axis=0)
    pad = np.maximum(pad_frac * rng, 2e-2)
    wx_min, wx_max = w_path[:,0].min()-pad[0], w_path[:,0].max()+pad[0]
    wy_min, wy_max = w_path[:,1].min()-pad[1], w_path[:,1].max()+pad[1]

    wx = np.linspace(wx_min, wx_max, grid_pts)
    wy = np.linspace(wy_min, wy_max, grid_pts)
    WX, WY = np.meshgrid(wx, wy)
    Z = np.zeros_like(WX)

    W_tmpl, b_tmpl = best_W.copy(), best_b.copy()
    for i in range(grid_pts):
        for j in range(grid_pts):
            W_tmp = W_tmpl.copy()
            W_tmp[neuron_idx] = [WX[j, i], WY[j, i]]
            theta_tmp = _pack_inner_params_local(W_tmp, b_tmpl)
            Z[j, i] = obj.reduced_objective(theta_tmp)

    # log10 safely: mask non-positive entries
    Zp = np.array(Z, copy=True)
    Zp[Zp <= 0] = np.nan
    L = np.log10(Zp)

    cf = _robust_contourf(ax, WX, WY, L, levels=30, cmap='viridis', rasterize=True)
    ax.plot(w_path[:,0], w_path[:,1], '-o', color='white', lw=2, ms=3, zorder=2)
    ax.set_xlabel(r'$w_x$'); ax.set_ylabel(r'$w_y$'); ax.set_aspect('equal')
    ax.set_title(fr'$\phi_{{{neuron_idx+1}}}$: $(w_x,w_y)$ slice')
    return cf

def plot_intercept_slope_contour(ax, obj, weight_history, biases_history,
                                 best_W, best_b, neuron_idx=0,
                                 grid_pts=120, pad_frac=0.15, clip_pct=99):
    w_path = np.array([W[neuron_idx] for W in weight_history])
    slope_path = w_path[:,1] / (w_path[:,0] + 1e-12)
    intercept_path = np.array([b[neuron_idx] for b in biases_history]) / (w_path[:,0] + 1e-12)

    s_clip = np.percentile(np.abs(slope_path[np.isfinite(slope_path)]), clip_pct) if slope_path.size else 10
    i_clip = np.percentile(np.abs(intercept_path[np.isfinite(intercept_path)]), clip_pct) if intercept_path.size else 10
    slope_path = np.clip(slope_path, -s_clip, s_clip)
    intercept_path = np.clip(intercept_path, -i_clip, i_clip)

    rng_i = np.ptp(intercept_path); rng_s = np.ptp(slope_path)
    pad_i = max(pad_frac * rng_i, 1e-3); pad_s = max(pad_frac * rng_s, 1e-3)
    i_min, i_max = intercept_path.min()-pad_i, intercept_path.max()+pad_i
    s_min, s_max = slope_path.min()-pad_s,     slope_path.max()+pad_s

    I = np.linspace(i_min, i_max, grid_pts)
    S = np.linspace(s_min, s_max, grid_pts)
    II, SS = np.meshgrid(I, S)
    Z = np.zeros_like(II)

    W_tmpl, b_tmpl = best_W.copy(), best_b.copy()
    norm_fixed = np.linalg.norm(best_W[neuron_idx])

    for i in range(grid_pts):
        for j in range(grid_pts):
            slope_val = np.clip(SS[j, i], -s_clip, s_clip)
            intercept_val = np.clip(II[j, i], -i_clip, i_clip)
            w_x = norm_fixed / np.sqrt(1 + slope_val**2)
            w_y = slope_val * w_x
            b_val = intercept_val * w_x
            W_tmp = W_tmpl.copy(); W_tmp[neuron_idx] = [w_x, w_y]
            b_tmp = b_tmpl.copy(); b_tmp[neuron_idx] = b_val
            theta_tmp = _pack_inner_params_local(W_tmp, b_tmp)
            Z[j, i] = obj.reduced_objective(theta_tmp)

    Zp = np.array(Z, copy=True)
    Zp[Zp <= 0] = np.nan
    L = np.log10(Zp)

    cf = _robust_contourf(ax, II, SS, L, levels=30, cmap='viridis', rasterize=True)
    ax.plot(intercept_path, slope_path, '-', color='white', lw=2, zorder=2)
    ax.set_xlabel(r'$b/w_x$'); ax.set_ylabel(r'$w_y/w_x$')
    ax.set_title(fr'$\phi_{{{neuron_idx+1}}}$: $(b/w_x,w_y/w_x)$ slice')
    return cf



# -------- normalized planes for neuron 0: (w_x/||w||, b/||w||) and (w_y/||w||, b/||w||) ----
def _loss_at(obj, W_base, b_base, k, wxn=None, wyn=None, bn=None, sign=(1.0, 1.0), norm_fixed=1.0):
    """
    Build (W,b) for neuron k given normalized coords.
    If wxn is given, use wy from sqrt(1-wxn^2) with wy sign; vice versa for wyn.
    b = bn * ||w||.
    """
    W = W_base.copy()
    b = b_base.copy()

    if wxn is not None:
        wxn = np.clip(wxn, -1.0, 1.0)
        wyn_val = sign[1] * np.sqrt(max(0.0, 1.0 - wxn**2))
        w_vec = np.array([wxn, wyn_val]) * norm_fixed
    else:
        wyn = np.clip(wyn, -1.0, 1.0)
        wxn_val = sign[0] * np.sqrt(max(0.0, 1.0 - wyn**2))
        w_vec = np.array([wxn_val, wyn]) * norm_fixed

    b_val = bn * norm_fixed
    W[k] = w_vec
    b[k] = b_val
    theta = _pack_inner_params_local(W, b)
    try:
        return obj.reduced_objective(theta)
    except Exception:
        return np.nan


def plot_norm_planes_neuron0(ax_left, ax_right, obj,
                             weight_history, biases_history,
                             best_W, best_b,
                             grid_pts=160, pad_frac=0.18,
                             levels=40, clip_pct=(2, 98)):
    """
    Two clean panels for neuron 0:
      left : x = w_x/||w||, y = b/||w||
      right: x = w_y/||w||, y = b/||w||
    The complementary component is chosen so ||w|| is fixed and sign
    matches the final solution.
    """
    k = 0  # neuron index fixed per request
    w_path = np.array([W[k] for W in weight_history])
    b_path = np.array([b[k] for b in biases_history])
    norms = np.linalg.norm(w_path, axis=1) + 1e-12
    wxn_path = w_path[:, 0] / norms
    wyn_path = w_path[:, 1] / norms
    bn_path  = b_path / norms

    # Windows with padding
    def window(arr):
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        span = max(hi - lo, 1e-3)
        pad = pad_frac * span
        return lo - pad, hi + pad

    wxn_min, wxn_max = window(wxn_path)
    wyn_min, wyn_max = window(wyn_path)
    bn_min,  bn_max  = window(bn_path)

    # Fixed norm and signs from the final solution
    norm_fixed = max(np.linalg.norm(best_W[k]), 1e-12)
    sign = (np.sign(best_W[k, 0]) or 1.0, np.sign(best_W[k, 1]) or 1.0)

    # Grids
    X1 = np.linspace(wxn_min, wxn_max, grid_pts)  # wx/||w||
    X2 = np.linspace(wyn_min, wyn_max, grid_pts)  # wy/||w||
    Y  = np.linspace(bn_min,  bn_max,  grid_pts)  # b/||w||
    X1G, Y1G = np.meshgrid(X1, Y)  # left plane
    X2G, Y2G = np.meshgrid(X2, Y)  # right plane

    Z1 = np.empty_like(X1G)
    Z2 = np.empty_like(X2G)

    W_base, B_base = best_W.copy(), best_b.copy()
    for j in range(grid_pts):
        for i in range(grid_pts):
            Z1[j, i] = _loss_at(obj, W_base, B_base, k,
                                wxn=X1G[j, i], bn=Y1G[j, i],
                                sign=sign, norm_fixed=norm_fixed)
            Z2[j, i] = _loss_at(obj, W_base, B_base, k,
                                wyn=X2G[j, i], bn=Y2G[j, i],
                                sign=sign, norm_fixed=norm_fixed)

    def draw_panel(ax, X, Y, Z, xlab):
        """
        Draw a filled contour plot for a slice of the normalised loss surface.

        Uses log10 scaling with robust percentile clipping and delegates to
        _robust_contourf to ensure contour levels are strictly increasing and
        to avoid passing unsupported kwargs. The axes limits are set from
        the meshgrid and a light grid is drawn. The y-axis label is always
        ``$b/\|w\|$``.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to draw.
        X, Y : ndarray
            Meshgrid arrays defining the slice grid.
        Z : ndarray
            Array of loss values on the grid (non-log-scaled).
        xlab : str
            Label for the x-axis.

        Returns
        -------
        cf : QuadContourSet or None
            The contour set returned by contourf, or None if nothing was
            drawn.
        """
        # Mask non-finite and non-positive values before log scaling
        Z = Z.astype(float)
        Z[~np.isfinite(Z)] = np.nan
        # Avoid log(0) by clipping to the smallest positive value present
        if np.any(Z > 0):
            zpos = np.nanmin(Z[Z > 0])
        else:
            zpos = 1e-16
        L = np.log10(np.clip(Z, zpos, np.nanmax(Z)))
        # Compute robust colour bounds using percentiles
        vmin, vmax = np.nanpercentile(L, clip_pct)
        # Ensure vmin < vmax; if equal, pad slightly
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-9
        # Use the helper to draw the contour and rasterise properly
        # Delegate to the robust contour helper with a numeric level count.
        # Passing an array for ``levels`` to _robust_contourf would cause it to
        # ignore our vmin/vmax and recompute its own, so we pass the desired
        # number of levels instead of a linspace array.
        cf = _robust_contourf(ax, X, Y, L, levels=levels, cmap='viridis', rasterize=True)
        ax.set_xlabel(xlab)
        ax.set_ylabel(r'$b/\|w\|$')
        ax.set_xlim(float(X.min()), float(X.max()))
        ax.set_ylim(float(Y.min()), float(Y.max()))
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
        return cf

    # Path in normalized coordinates
    ax_left.plot(wxn_path, bn_path, '-', color='white', lw=2, alpha=0.9, zorder=2)
    ax_left.plot(wxn_path[0], bn_path[0], 's', color='lime', ms=6, zorder=3)
    ax_left.plot(wxn_path[-1], bn_path[-1], '*', color='red',  ms=9, zorder=3)
    cf1 = draw_panel(ax_left, X1G, Y1G, Z1, r'$w_x/\|w\|$')
    ax_left.set_title(r'$\phi_1$: $(w_x/\|w\|,\ b/\|w\|)$')

    ax_right.plot(wyn_path, bn_path, '-', color='white', lw=2, alpha=0.9, zorder=2)
    ax_right.plot(wyn_path[0], bn_path[0], 's', color='lime', ms=6, zorder=3)
    ax_right.plot(wyn_path[-1], bn_path[-1], '*', color='red',  ms=9, zorder=3)
    cf2 = draw_panel(ax_right, X2G, Y2G, Z2, r'$w_y/\|w\|$')
    ax_right.set_title(r'$\phi_1$: $(w_y/\|w\|,\ b/\|w\|)$')

    # Apply smart axis formatting to both panels for nicer tick spacing
    for ax in (ax_left, ax_right):
        smart_axis_format(ax, 'x')
        smart_axis_format(ax, 'y')

    return cf1, cf2


def smart_axis_format(ax, axis='x', num_ticks=5):
    """
    Format an axis with a fixed number of ticks and rounded scientific notation.

    This helper mirrors the function originally defined in the notebook.  It sets
    exactly ``num_ticks`` major ticks on the chosen axis and formats them in
    scientific notation with 3 significant figures (except zero which is
    displayed as ``"0"``).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis whose ticks should be formatted.
    axis : {'x', 'y'}, optional
        Which axis to format.  Defaults to `'x'`.
    num_ticks : int, optional
        Desired number of major ticks.  Defaults to 5.
    """
    if axis == 'x':
        axis_obj = ax.xaxis
    else:
        axis_obj = ax.yaxis

    def fmt_func(x, pos):
        # Format tick with 3 significant figures; special case zero
        if x == 0:
            return "0"
        return f"{x:.3g}"

    formatter = FuncFormatter(fmt_func)
    axis_obj.set_major_formatter(formatter)
    locator = MaxNLocator(nbins=num_ticks)
    axis_obj.set_major_locator(locator)
