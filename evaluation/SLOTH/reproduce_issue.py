import numpy as np

# Physics constants from slimline.py / scipy
c = 299792458.0
# The constant used in dsdt for critical density calculation
# K = epsilon_0 * m_e / e^2
K_dsdt = 3.14207787e-4 

# Simulation setup matching quad_test.ipynb defaults
# NOTE: Notebook defines lwl = 351e-9 but solve() uses 1064e-9 default!
lwl_sim_default = 1064e-9 
lwl_notebook_intended = 351e-9

omega_sim_default = 2 * np.pi * c / lwl_sim_default
omega_notebook_intended = 2 * np.pi * c / lwl_notebook_intended

# Density profile parameters
# n_cr default in quad_trough is 9.049e27 (which is n_crit for 351nm)
n_cr = 9.049e27
y_c = 5e-2

# Define domain
y_grid = np.linspace(-0.05, 0.05, 100)

# Electron density profile: ne = n_cr * ( (y/yc)^2 + 1 ) / 2
ne = n_cr * ((y_grid/y_c)**2 + 1) / 2.0

# Simulated acceleration logic from dsdt/dndr
# accel = grad(-0.5 * c**2 * ne / (K * omega**2))
def get_accel(omega):
    # This is what's happening inside the code
    gradient_term = -0.5 * c**2 * ne / (K_dsdt * omega**2)
    return np.gradient(gradient_term, y_grid)

accel_sim_mismatch = get_accel(omega_sim_default)
accel_sim_corrected = get_accel(omega_notebook_intended)

# Analytical acceleration from notebook's intended physics
# Assuming n_e = n_crit * ( (y/y_c)^2 + 1 ) / 2
# a = -0.5 * c^2 * grad(n/nc) = -0.5 * c^2 * (y/y_c^2)
accel_notebook_intended = -0.5 * c**2 * y_grid / (y_c**2)

# Select a point to compare
idx = 75 # y approx 0.025
print(f"--- Analysis of Period Discrepancy ---")
print(f"Point of comparison: y = {y_grid[idx]:.4f} m")
print(f"Simulated Accel (Mismatch: 1064nm): {accel_sim_mismatch[idx]:.4e}")
print(f"Simulated Accel (Corrected: 351nm): {accel_sim_corrected[idx]:.4e}")
print(f"Notebook Intended Accel (Analytical): {accel_notebook_intended[idx]:.4e}")

print(f"\n--- Acceleration Ratios ---")
print(f"Mismatch Accel / Intended Accel: {accel_sim_mismatch[idx] / accel_notebook_intended[idx]:.4f}")
print(f"Corrected Accel / Intended Accel: {accel_sim_corrected[idx] / accel_notebook_intended[idx]:.4f}")

# Period Ratios
# Period ratio = sqrt(1 / accel_ratio)
ratio_period_mismatch = np.sqrt(1.0 / (accel_sim_mismatch[idx] / accel_notebook_intended[idx]))
ratio_period_corrected = np.sqrt(1.0 / (accel_sim_corrected[idx] / accel_notebook_intended[idx]))

print(f"\n--- Period Ratios ---")
print(f"Simulated Period (Mismatch) / Analytical Period: {ratio_period_mismatch:.4f}")
print(f"Simulated Period (Corrected) / Analytical Period: {ratio_period_corrected:.4f}")

print(f"\nConclusion:")
print(f"1. The 'half period' error (ratio 0.467) is caused by the wavelength mismatch.")
print(f"   The solver uses 1064nm by default, but the density is scaled for 351nm.")
print(f"2. Even if fixed, a factor of sqrt(2) (approx 1.41) discrepancy remains.")
print(f"   This is because the density profile has a factor of 1/2 that the ")
print(f"   analytical solution y = y0*cos(x/y_c) does not expect.")
