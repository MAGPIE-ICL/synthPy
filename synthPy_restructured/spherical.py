import numpy as np
import sys
import os
from jax.scipy.interpolate import RegularGridInterpolator
import jax.numpy as jnp
from scipy.constants import c
import jax
import domain as d
sys.path.insert(0, '../../utils')
print(os.getcwd())
from SpK_reader import open_emi_files

def distance(x,y,z):
    return np.sqrt(x**2 + y**2 + z**2)

def spherical_2(domain, radii, ne, rho):
    radii = (0,) + radii
    domain.spherical = True
    domain.num_layers = len(ne)
    domain.densities = jnp.array((0,) + rho)
    ne_grid = np.zeros_like(domain.XX)
    rho_grid = np.zeros_like(domain.XX)
    distance_grid = distance(domain.XX, domain.YY, domain.ZZ)
    for i in range (0, len(radii)-1):
        ne_grid[(radii[i] < distance_grid) & (distance_grid<= radii[i+1])] = ne[i]
        rho_grid[(radii[i] < distance_grid) & (distance_grid<= radii[i+1])] = rho[i]
    
    return ne_grid, rho_grid

def spherical_interps(Propagator, num_layers, files):
    opa_max=Propagator.ScalarDomain.x_n/(Propagator.ScalarDomain.x_length)
    Propagator.opacity_spatial_interps = []
    for i in range (num_layers):
        grp_centres, grps, rho, Te, opa_data = open_emi_files(f"../../{files[i]}")
        opa_data_capped = jnp.minimum(opa_max, opa_data)
        Propagator.opacity_interp = RegularGridInterpolator((grp_centres, rho, Te), opa_data_capped, bounds_error = False, fill_value = 0.0)
        Propagator.rho_interp = RegularGridInterpolator(
            (Propagator.ScalarDomain.x, Propagator.ScalarDomain.y, Propagator.ScalarDomain.z), Propagator.ScalarDomain.rho, bounds_error = False, fill_value = 0.0, method = "nearest")
        opacity_grid = Propagator.opacity_interp((Propagator.energy, Propagator.ScalarDomain.rho, Propagator.ScalarDomain.Te))
        Propagator.opacity_spatial_interps.append(RegularGridInterpolator((
            Propagator.ScalarDomain.x, Propagator.ScalarDomain.y, Propagator.ScalarDomain.z), opacity_grid, bounds_error = False, fill_value = 0.0))
        
def interp_selector(interp_array, indexs, x):
    result = jnp.array([interp_array[i-1](x[j, :]) for j, i in enumerate(indexs)])
    result = jnp.ravel(result)
    return result
    
def spherical_atten(Propagator, x):
    density_values = Propagator.rho_interp(x.T)
    #jax.jit(jnp.where, static_argnames = "size")
    index = jnp.array([jnp.where(jnp.isclose(
    Propagator.ScalarDomain.densities, num), size = 1)[0] for num in density_values])
    index = jnp.ravel(index)
    print(index)
    opacity = jnp.zeros_like(index, dtype = jnp.float64)
    #opacity[index != 0] = Propagator.opacity_spatial_interps[index[index != 0]](x.T)
    opacity = opacity.at[index != 0].set(interp_selector(Propagator.opacity_spatial_interps, index[index != 0], x.T[index != 0, :]))
    print("hi1",opacity)
    return -opacity*c