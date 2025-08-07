import numpy as np
import sys
import os
from jax.scipy.interpolate import RegularGridInterpolator
import jax.numpy as jnp
from scipy.constants import c
import domain as d

sys.path.insert(0, '../../utils')
from SpK_reader import open_emi_files

def distance(x,y,z):
    return np.sqrt(x**2 + y**2 + z**2)

def spherical(domain, radii, ne, rho):
    radii = (0,) + radii
    domain.spherical = True
    domain.num_layers = len(ne)
    ne_grid = np.zeros_like(domain.XX)
    initialise  = [np.zeros_like(domain.XX)]
    rho_grid_list = len(rho)*initialise
    distance_grid = distance(domain.XX, domain.YY, domain.ZZ)
    for i in range (0, len(radii)-1):
        ne_grid[(radii[i] < distance_grid) & (distance_grid<= radii[i+1])] = ne[i]
        rho_grid_list[i][(radii[i] < distance_grid) & (distance_grid<= radii[i+1])] = rho[i]

    return ne_grid, rho_grid_list

def spherical_interps(Propagator, num_layers, files, rho_grid_list):
    opa_max = Propagator.ScalarDomain.x_n/(Propagator.ScalarDomain.x_length)
    Propagator.opacity_grids = [0]*num_layers
    for i in range (num_layers):
        grp_centres, grps, rho, Te, opa_data = open_emi_files(f"../../{files[i]}")
        opa_data_capped = jnp.minimum(opa_max, opa_data)
        Propagator.opacity_interp = RegularGridInterpolator((grp_centres, rho, Te), opa_data_capped, bounds_error = False, fill_value = 0.0)
        #Propagator.rho_interp = RegularGridInterpolator(
            #(Propagator.ScalarDomain.x, Propagator.ScalarDomain.y, Propagator.ScalarDomain.z), Propagator.ScalarDomain.rho, bounds_error = False, fill_value = 0.0, method = "nearest")
        Propagator.opacity_grids[i] = Propagator.opacity_interp((Propagator.energy, rho_grid_list[i], Propagator.ScalarDomain.Te))
        # Propagator.opacity_spatial_interps.append(RegularGridInterpolator((
        #     Propagator.ScalarDomain.x, Propagator.ScalarDomain.y, Propagator.ScalarDomain.z), opacity_grid, bounds_error = False, fill_value = 0.0))
    
    Propagator.opacity_grids_tot = np.sum(Propagator.opacity_grids, axis = 0)
    Propagator.opacity_spatial_interp_tot = RegularGridInterpolator((
         Propagator.ScalarDomain.x, Propagator.ScalarDomain.y, Propagator.ScalarDomain.z), Propagator.opacity_grids_tot, bounds_error = False, fill_value = 0.0)
    
        
def spherical_atten(Propagator, x):
    opacity = Propagator.opacity_spatial_interp_tot(x.T)
    return -opacity*c
