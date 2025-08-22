
import jax.numpy as jnp
import field_generator.gaussian3D as g3

def distance(x,y,z):
    return jnp.sqrt(x**2 + y**2 + z**2)

def grid(lengths, n_cells):
    """
    lengths should both be an array of length 3
    n_cells can either be an array of length 3 or a float/int
    """
    if isinstance(n_cells, (float, int)):
        n_cells = 3*[n_cells]
    xlen, ylen, zlen = lengths[0], lengths[1], lengths[2]
    x_n, y_n, z_n = n_cells[0], n_cells[1], n_cells[2]
    x = jnp.linspace(-xlen/2, xlen/2, x_n)
    y = jnp.linspace(-ylen/2, ylen/2, y_n)
    z = jnp.linspace(-zlen/2, zlen/2, z_n)
    XX, YY, ZZ = jnp.meshgrid(x, y, z, indexing = 'ij')
    return XX, YY, ZZ


def spherical(radii, ne, rho, lengths, n_cells):
    #I feel like this is probably not very memory efficient. Might be worth optimising in the future
    radii = (0,) + radii
    XX, YY, ZZ = grid(lengths, n_cells)
    ne_grid = jnp.zeros_like(XX)
    rho_grid_list  = len(rho)*[ne_grid]
    distance_grid = distance(XX, YY, ZZ)
    for i in range (0, len(radii)-1):
        ne_grid = ne_grid.at[(radii[i] < distance_grid) & (distance_grid<= radii[i+1])].set(ne[i])
        rho_grid_list[i] = rho_grid_list[i].at[(radii[i] < distance_grid) & (distance_grid<= radii[i+1])].set(rho[i])

    return ne_grid, rho_grid_list

def density_step(rho1, rho2, lengths, n_cells):
    """
    A density function which has value rho1 for y>0 and rho2 for y<=0
    """
    _, YY, _ = grid(lengths, n_cells)
    density = jnp.full_like(YY, rho1)
    density[YY <= 0] = rho2
    return density

def turbulent_box(ne_0, max_pert, l_max, l_min, extent, n_cells, power = -5/3):
    #NB: n_cells must be even
    def k_func(k):
        return k**(power)
    
    field = g3.gaussian3D(k_func)
    noise3D = field.domain_fft(l_max, l_min, extent, n_cells//2, factor = 1.0)

    ne = ne_0 + noise3D*max_pert
    return ne