import numpy as np
import sys

from jax.scipy.interpolate import RegularGridInterpolator
import jax.numpy as jnp
from scipy.constants import c
import domain as d

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

def density_step(domain, rho1, rho2):
    """
    A density function which has value rho1 for y>0 and rho2 for y<=0
    """
    density = np.zeros_like(domain.YY)
    density[domain.YY > 0] = rho1
    density[domain.YY <= 0] = rho2
    return density