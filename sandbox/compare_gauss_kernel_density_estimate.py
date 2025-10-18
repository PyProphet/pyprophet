import numpy as np
import scipy.stats
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.stats import gaussian_kde


# Reproduce bw_nrd0 from stats.py (Scott's rule variant)
def bw_nrd0(x):
    """Bandwidth selector matching R's bw.nrd0 (used in stats.py)"""
    if len(x) < 2:
        raise ValueError("need at least 2 data points")

    hi = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    lo = min(hi, iqr / 1.34)

    if lo == 0:
        lo = hi if hi != 0 else abs(x[0]) if x[0] != 0 else 1

    return 0.9 * lo * len(x) ** (-0.2)


# ----- test data (simulate probit-transformed p-values) -----
rng = np.random.default_rng(42)
n = 4000

# Simulate p-values, then apply probit transform like in lfdr()
p_values = np.concatenate(
    [
        rng.beta(0.5, 5, int(n * 0.7)),  # enriched near 0 (true positives)
        rng.uniform(0.01, 0.99, int(n * 0.3)),  # uniform null
    ]
)
eps = np.power(10.0, -8)
p = np.maximum(p_values, eps)
p = np.minimum(p, 1 - eps)
x = scipy.stats.norm.ppf(p, loc=0, scale=1)  # probit transform

print(
    f"Data: n={len(x)}, range=[{x.min():.3f}, {x.max():.3f}], std={x.std(ddof=1):.3f}"
)

# ----- Parameters from stats.py -----
adj = 1.5  # bandwidth adjustment factor from lfdr()
bw = bw_nrd0(x)
print(f"\nBandwidth (bw_nrd0): {bw:.6f}")
print(f"Adjusted bandwidth: {adj * bw:.6f}")

# ----- statsmodels KDE (original stats.py implementation) -----
myd_sm = KDEUnivariate(x)
myd_sm.fit(bw=adj * bw, gridsize=512)

# Evaluate at original data points (like in stats.py after spline interpolation)
import scipy as sp

splinefit = sp.interpolate.splrep(myd_sm.support, myd_sm.density)
y_sm = sp.interpolate.splev(x, splinefit)

# ----- SciPy KDE (proposed replacement) -----
# Convert absolute bandwidth to scipy's bandwidth factor
bw_method_factor = (adj * bw) / x.std(ddof=1)
print(f"SciPy bw_method factor: {bw_method_factor:.6f}")

kde_sp = gaussian_kde(x, bw_method=bw_method_factor)
y_sp = kde_sp(x)

# ----- Comparison at data points -----
abs_diff = np.abs(y_sm - y_sp)
rel_diff = abs_diff / np.maximum(y_sp, 1e-12)

print("\n=== Comparison at original data points (x) ===")
print(f"statsmodels y range: [{y_sm.min():.6e}, {y_sm.max():.6e}]")
print(f"scipy y range:       [{y_sp.min():.6e}, {y_sp.max():.6e}]")
print(f"Max abs diff: {abs_diff.max():.3e}")
print(f"Mean abs diff: {abs_diff.mean():.3e}")
print(
    f"Median rel diff: {np.median(rel_diff):.3e} | 95th pct: {np.percentile(rel_diff, 95):.3e}"
)

# ----- Test on a finer grid (more comprehensive) -----
grid = np.linspace(x.min() - 2 * x.std(), x.max() + 2 * x.std(), 800)
dens_sm = sp.interpolate.splev(grid, splinefit)
dens_sp = kde_sp(grid)

abs_diff_grid = np.abs(dens_sm - dens_sp)
rel_diff_grid = abs_diff_grid / np.maximum(dens_sp, 1e-12)

print("\n=== Comparison on evaluation grid ===")
print(f"Max abs diff: {abs_diff_grid.max():.3e}")
print(
    f"Median rel diff: {np.median(rel_diff_grid):.3e} | 95th pct: {np.percentile(rel_diff_grid, 95):.3e}"
)
print(
    f"Area under curve (trapezoid): SM={np.trapezoid(dens_sm, grid):.6f}, SP={np.trapezoid(dens_sp, grid):.6f}"
)

# ----- L2 norm -----
l2 = np.sqrt(np.trapezoid((dens_sm - dens_sp) ** 2, grid))
print(f"L2 difference: {l2:.3e}")

# ----- Test lfdr output directly -----
print("\n=== Testing actual lfdr computation ===")
pi0 = 0.5
lfdr_sm = pi0 * scipy.stats.norm.pdf(x) / y_sm
lfdr_sp = pi0 * scipy.stats.norm.pdf(x) / y_sp

# Apply same truncation as in stats.py
lfdr_sm_trunc = np.clip(lfdr_sm, 0, 1)
lfdr_sp_trunc = np.clip(lfdr_sp, 0, 1)

lfdr_diff = np.abs(lfdr_sm_trunc - lfdr_sp_trunc)
print(f"lfdr max diff: {lfdr_diff.max():.3e}")
print(f"lfdr mean diff: {lfdr_diff.mean():.3e}")
print(f"lfdr median diff: {np.median(lfdr_diff):.3e}")

# ----- Summary -----
print("\n=== Summary ===")
if abs_diff.max() < 1e-6 and lfdr_diff.max() < 1e-4:
    print("✓ scipy.stats.gaussian_kde is a suitable replacement")
    print("  Differences are negligible for practical purposes")
else:
    print("⚠ Check differences - may need adjustment")

print(f"\nTo apply changes, remove statsmodels from pyproject.toml:")
print("  - statsmodels >= 0.8.0")
