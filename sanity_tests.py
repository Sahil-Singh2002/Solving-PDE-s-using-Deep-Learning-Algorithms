# sanity_tests.py
import numpy as np
from numpy.linalg import eigvalsh
from geometry_utils import Domain2D
from hyperplane_utils import generate_uniform_hyperplanes
from area_utils import diagonal_matrix_entry, lower_matrix_entry, compute_F_from_step1
from mass_matrix_utils import build_mass_matrix_unified

ATOL = 1e-10

def assert_close(a, b, msg, atol=ATOL):
    if not np.isclose(a, b, rtol=0.0, atol=atol):
        raise AssertionError(f"{msg}: {a} != {b} (atol={atol})")

def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)

# Analytic areas in Omega = {(x,y): x>=0, y>=0, x+y<=1}
def area_vertical(c):
    if c <= 0: return 0.5
    if c >= 1: return 0.0
    return 0.5 * (1 - c)**2

def area_horizontal(d):
    if d <= 0: return 0.5
    if d >= 1: return 0.0
    return 0.5 * (1 - d)**2

def area_intersection_x_ge_c_y_ge_d(c, d):
    if c + d >= 1: return 0.0
    c = max(c, 0.0); d = max(d, 0.0)
    return 0.5 * (1 - c - d)**2

def make_domain():
    return Domain2D(
        np.array([1.0, 0.0]), 0.0,      # x >= 0
        np.array([0.0, 1.0]), 0.0,      # y >= 0
        np.array([-1.0,-1.0]), -1.0     # x + y <= 1
    )

def test_single_halfspaces():
    d = make_domain()
    # Diagonal set {x - y >= 0} has area 0.25
    A = diagonal_matrix_entry(d, np.array([1.0, -1.0]), 0.0)
    assert_close(A, 0.25, "Area of {x - y >= 0} inside Omega")

    # Vertical x >= c
    for c in [0.0, 0.2, 0.6, 1.0]:
        A_num = diagonal_matrix_entry(d, np.array([1.0, 0.0]), c)
        A_ref = area_vertical(c)
        assert_close(A_num, A_ref, f"Area of {{x>={c}}}")

    # Horizontal y >= d
    for e in [0.0, 0.3, 0.75, 1.0]:
        A_num = diagonal_matrix_entry(d, np.array([0.0, 1.0]), e)
        A_ref = area_horizontal(e)
        assert_close(A_num, A_ref, f"Area of {{y>={e}}}")

def test_intersections():
    d = make_domain()
    for c, e in [(0.2, 0.25), (0.1, 0.1), (0.7, 0.1), (0.4, 0.61)]:
        A_num = lower_matrix_entry(d,
                                   np.array([1.0, 0.0]), c,
                                   np.array([0.0, 1.0]), e)
        A_ref = area_intersection_x_ge_c_y_ge_d(c, e)
        assert_close(A_num, A_ref, f"Area of x>={c} & y>={e}")

def test_unified_builder_branches():
    d = make_domain()
    W, B = generate_uniform_hyperplanes(8)

    # f == 1 (no target)
    M1, F1, A1 = build_mass_matrix_unified(d, W, B)
    area = compute_F_from_step1(d)  # 0.5
    assert_close(A1, area, "A with f==1")
    assert_true(np.allclose(np.diag(M1), F1.ravel(), rtol=0.0, atol=ATOL),
                "F == diag(M) when f==1")
    assert_true(np.allclose(M1, M1.T, rtol=0.0, atol=ATOL), "M symmetric")

    # f == 0 (w_f ~ 0, b_f >= 0)
    M0, F0, A0 = build_mass_matrix_unified(d, W, B, np.zeros(2), +1.0)
    assert_close(A0, 0.0, "A with f==0")
    assert_true(np.allclose(F0, 0.0, rtol=0.0, atol=ATOL), "F zeros when f==0")

    # general Heaviside target {x - y >= 0}
    wf, bf = np.array([1.0, -1.0]), 0.0
    M, F, A = build_mass_matrix_unified(d, W, B, wf, bf)
    assert_close(A, 0.25, "A for target {x - y >= 0}")
    assert_true(np.all(F >= -ATOL), "F nonnegative")
    assert_true(np.all(F.ravel() <= np.diag(M) + ATOL), "F <= diag(M) elementwise")

def test_invariance_and_symmetry():
    d = make_domain()
    # invariance to positive scaling of (w,b): set {w^T x >= b} == {alpha w^T x >= alpha b}
    wf, bf = np.array([0.3, -0.7]), 0.1
    A1 = diagonal_matrix_entry(d, wf, bf)
    A2 = diagonal_matrix_entry(d, 2*wf, 2*bf)
    assert_close(A1, A2, "Positive scaling invariance of target")

    # symmetry of lower_matrix_entry
    w1, b1 = np.array([1.0, 0.0]), 0.2
    w2, b2 = np.array([0.0, 1.0]), 0.3
    a = lower_matrix_entry(d, w1, b1, w2, b2)
    b = lower_matrix_entry(d, w2, b2, w1, b1)
    assert_close(a, b, "Symmetry of intersection area")

def test_psd_of_M():
    d = make_domain()
    W, B = generate_uniform_hyperplanes(16)
    M, F, A = build_mass_matrix_unified(d, W, B)   # f==1 branch OK
    # since M_ij = <phi_i, phi_j>_L2 and phi_i are indicators, M is PSD
    evals = eigvalsh(0.5*(M+M.T))
    assert_true(np.all(evals >= -1e-12), "M should be PSD (within tol)")

def test_target_equals_basis_column():
    d = make_domain()
    W, B = generate_uniform_hyperplanes(8)
    k = 3
    wf, bf = W[k], B[k]
    M, F, A = build_mass_matrix_unified(d, W, B, wf, bf)
    # If target == basis k, projection should be the k-th column of M
    assert_true(np.allclose(F.ravel(), M[:, k], rtol=0.0, atol=ATOL),
                "F equals M[:,k] when target == basis k")
    assert_close(A, M[k, k], "A equals M[k,k] when target == basis k")

def test_monotonicity_vertical_horizontal():
    d = make_domain()
    # x >= c is decreasing in c
    for c1, c2 in [(0.0,0.2),(0.2,0.6)]:
        A1 = diagonal_matrix_entry(d, np.array([1.0,0.0]), c1)
        A2 = diagonal_matrix_entry(d, np.array([1.0,0.0]), c2)
        assert_true(A2 <= A1 + ATOL, "Area decreases as c increases (vertical)")
    # y >= d is decreasing in d
    for d1, d2 in [(0.0,0.3),(0.3,0.75)]:
        A1 = diagonal_matrix_entry(d, np.array([0.0,1.0]), d1)
        A2 = diagonal_matrix_entry(d, np.array([0.0,1.0]), d2)
        assert_true(A2 <= A1 + ATOL, "Area decreases as d increases (horizontal)")

def test_random_monte_carlo_sanity(n_samples=100000, seed=0):
    # Cross-check one random half-space via Monte Carlo
    rng = np.random.default_rng(seed)
    d = make_domain()

    # sample uniformly in Omega via triangle sampling
    # sample u,v ~ U(0,1), accept if u+v <= 1; else reflect
    U = rng.random(n_samples)
    V = rng.random(n_samples)
    mask = U + V <= 1
    x = np.where(mask, U, 1 - U)
    y = np.where(mask, V, 1 - V)

    wf, bf = np.array([0.7, -0.4]), 0.1
    A_true = diagonal_matrix_entry(d, wf, bf)
    # MC estimate
    inside = (wf[0]*x + wf[1]*y >= bf)
    A_mc = 0.5 * np.mean(inside.astype(float))  # scale by area(Omega)=0.5
    assert_true(abs(A_mc - A_true) < 5e-3, f"Monte Carlo sanity: {A_mc} vs {A_true}")

def run_all():
    test_single_halfspaces()
    test_intersections()
    test_unified_builder_branches()
    test_invariance_and_symmetry()
    test_psd_of_M()
    test_target_equals_basis_column()
    test_monotonicity_vertical_horizontal()
    test_random_monte_carlo_sanity()
    print("All sanity tests PASSED")

if __name__ == "__main__":
    run_all()
