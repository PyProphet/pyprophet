"""Compare our FFT-based KDE with statsmodels"""

import numpy as np
import scipy.stats

from pyprophet.stats import (
    _fast_linbin,
    _grid_kde_fft,
    bw_nrd0,
)

# Import statsmodels for reference
try:
    from statsmodels.nonparametric.kde import KDEUnivariate

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not installed")


print("=" * 70)
print("Comparison of FFT-based KDE with statsmodels KDEUnivariate")


# Test data
p_values = np.full(16, 0.0625)
pi0 = 1.0
eps = 1e-8
adj = 1.5

p = np.maximum(p_values, eps)
p = np.minimum(p, 1 - eps)
x = scipy.stats.norm.ppf(p, loc=0, scale=1)

print(f"Data: n={len(x)}, x={x[0]:.6f} (all identical)")
print()

# Bandwidth
bw = bw_nrd0(x)
print(f"Bandwidth (bw_nrd0): {bw:.6f}")
print(f"Adjusted bandwidth: {adj * bw:.6f}")
print()

if HAS_STATSMODELS:
    print("=" * 70)
    print("STATSMODELS kdensityfft:")
    print("=" * 70)

    myd = KDEUnivariate(x)
    myd.fit(bw=adj * bw, gridsize=512, fft=True)

    print(f"Grid range: [{myd.support.min():.6f}, {myd.support.max():.6f}]")
    print(f"Grid size: {len(myd.support)}")
    print(f"Grid delta: {myd.support[1] - myd.support[0]:.6f}")

    # Find density at data point
    import scipy.interpolate as sp_interp

    splinefit = sp_interp.splrep(myd.support, myd.density)
    y_statsmodels = sp_interp.splev(x, splinefit)

    print(f"Density at x={x[0]:.6f}: {y_statsmodels[0]:.6f}")

    null_density = scipy.stats.norm.pdf(x[0])
    print(f"Null density: {null_density:.6f}")

    lfdr_statsmodels = pi0 * null_density / y_statsmodels[0]
    print(f"LFDR = {lfdr_statsmodels:.6f}")
    print()

print("=" * 70)
print("OUR FFT-based KDE:")
print("=" * 70)

density, grid = _grid_kde_fft(x, adj * bw, gridsize=512, cut=3)

print(f"Grid range: [{grid.min():.6f}, {grid.max():.6f}]")
print(f"Grid size: {len(grid)}")
print(f"Grid delta: {grid[1] - grid[0]:.6f}")

# Interpolate to data point
y_ours = np.interp(x[0], grid, density)

print(f"Density at x={x[0]:.6f}: {y_ours:.6f}")

null_density = scipy.stats.norm.pdf(x[0])
print(f"Null density: {null_density:.6f}")

lfdr_ours = pi0 * null_density / y_ours
print(f"LFDR = {lfdr_ours:.6f}")
print()

if HAS_STATSMODELS:
    print("=" * 70)
    print("COMPARISON:")
    print("=" * 70)
    print(f"Statsmodels LFDR: {lfdr_statsmodels:.6f}")
    print(f"Our LFDR:         {lfdr_ours:.6f}")
    print(f"Difference:       {abs(lfdr_statsmodels - lfdr_ours):.6f}")

    # Compare densities around the data point
    print("\nDensity comparison around data point:")
    idx = np.argmin(np.abs(grid - x[0]))
    print(f"Grid index near x: {idx}")
    for i in range(max(0, idx - 3), min(len(grid), idx + 4)):
        sm_dens = np.interp(grid[i], myd.support, myd.density)
        print(
            f"  grid[{i}]={grid[i]:8.6f}: ours={density[i]:.6f}, statsmodels={sm_dens:.6f}, diff={abs(density[i] - sm_dens):.6f}"
        )


### Debugging FFT steps in detail
print("\n" + "=" * 70)
print("DEBUGGING FFT STEPS:")

# Test data
p_values = np.full(16, 0.0625)
eps = 1e-8
adj = 1.5

p = np.maximum(p_values, eps)
p = np.minimum(p, 1 - eps)
x = scipy.stats.norm.ppf(p, loc=0, scale=1)

bw = bw_nrd0(x)
nobs = len(x)
gridsize = 512
cut = 3

a = np.min(x) - cut * adj * bw
b = np.max(x) + cut * adj * bw

grid = np.linspace(a, b, gridsize)
delta = grid[1] - grid[0]
range_ = b - a

print("Setup:")
print(f"  nobs = {nobs}")
print(f"  bw = {bw:.6f}")
print(f"  adj * bw = {adj * bw:.6f}")
print(f"  a = {a:.6f}, b = {b:.6f}")
print(f"  range = {range_:.6f}")
print(f"  delta = {delta:.6f}")
print()

# Linear binning
binned_raw = _fast_linbin(x, a, b, gridsize)
print(f"Raw binned sum: {binned_raw.sum():.6f} (should be {nobs})")

binned = binned_raw / (delta * nobs)
print(f"Normalized binned sum: {binned.sum():.6f}")
print(
    f"Normalized binned * delta: {(binned * delta).sum():.6f} (should be 1/nobs = {1 / nobs:.6f})"
)
print()

# Check where the data landed in the grid
idx = np.argmin(np.abs(grid - x[0]))
print(f"Data point x={x[0]:.6f} is closest to grid[{idx}]={grid[idx]:.6f}")
print("Binned values around data:")
for i in range(max(0, idx - 3), min(gridsize, idx + 4)):
    print(
        f"  grid[{i}]={grid[i]:8.6f}: binned_raw={binned_raw[i]:.6f}, binned={binned[i]:.6f}"
    )
print()

# FFT
y_fft = np.fft.fft(binned)
print(f"FFT output[0] (DC component): {y_fft[0]:.6f}")
print()

# Silverman transform
omega = np.arange(gridsize) * np.pi / range_
silverman = np.exp(-0.5 * (adj * bw * omega) ** 2)
print(f"Silverman transform[0]: {silverman[0]:.6f}")
print(f"Silverman transform[1]: {silverman[1]:.6f}")
print()

# Apply kernel
zstar = silverman * y_fft
print(f"zstar[0]: {zstar[0]:.6f}")
print()

# Inverse FFT
density = np.real(np.fft.ifft(zstar))
print(f"Density sum: {density.sum():.6f}")
print(f"Density * delta: {(density * delta).sum():.6f} (should integrate to 1)")
print(f"Density at grid[{idx}]: {density[idx]:.6f}")
print()

# Interpolate to data point
y_interp = np.interp(x[0], grid, density)
print(f"Interpolated density at x={x[0]:.6f}: {y_interp:.6f}")


# Test
p_values = np.full(16, 0.0625)
eps = 1e-8
adj = 1.5

p = np.maximum(p_values, eps)
p = np.minimum(p, 1 - eps)
x = scipy.stats.norm.ppf(p, loc=0, scale=1)

bw = bw_nrd0(x)

print("=" * 70)
print("Testing our FFT implementation:")
print("=" * 70)
density, grid = _grid_kde_fft(x, adj * bw, gridsize=512, cut=3)

y = np.interp(x[0], grid, density)
print(f"\nInterpolated density at x={x[0]:.6f}: {y:.6f}")
print("Expected (statsmodels): 0.334720")
print(f"Difference: {abs(y - 0.334720):.6f}")
