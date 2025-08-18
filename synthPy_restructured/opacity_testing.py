import numpy as np
import domain as d
import sys
import os
import matplotlib.pyplot as plt
sys.path.append('utils') 

from SpK_reader import open_emi_files
from scipy.interpolate import RegularGridInterpolator
def density_step(domain, rho1, rho2):
    """
    A density function which has value rho1 for y>0 and rho2 for y<=0
    """
    density = np.zeros_like(domain.YY)
    density[domain.YY > 0] = rho1
    density[domain.YY <= 0] = rho2
    return density

wavelength = 0.1e-9
energy = 6.63e-34*3e8/(wavelength*1.6e-19)

extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

n_cells = 100
probing_extent = extent_z
probing_direction = 'z'

lengths = 2 * np.array([extent_x, extent_y, extent_z])
ScalarDomain = d.ScalarDomain(lengths, n_cells) # B_on = False by default
ScalarDomain.external_mass_density(density_step(ScalarDomain, 1e4, 1e3))
ScalarDomain.external_Te(np.full((n_cells, n_cells, n_cells), 2e3))

grp_centres, grps, rho, Te, opa_data = open_emi_files("opa_multi_planck_CH_LTE_210506_Hydra_ColdOpa.spk")
opa_max=ScalarDomain.x_n/(ScalarDomain.x_length)
opa_data_capped=np.minimum(opa_max, opa_data)
opacity_interp = RegularGridInterpolator((grp_centres, rho, Te), opa_data_capped, bounds_error = False, fill_value = 0.0)
opacity_grid = opacity_interp((energy, ScalarDomain.rho, ScalarDomain.Te))
opacity_spatial_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), opacity_grid, bounds_error = False, fill_value = 0.0)

XX, YY = np.meshgrid(ScalarDomain.x, ScalarDomain.y)
opacity = opacity_spatial_interp((XX, YY, 0))

plt.scatter(XX, YY, c = opacity)
plt.colorbar()
# axes = plt.axes(projection = "3d")
# plt.plot(XX, YY, opacity, ".")
plt.show()