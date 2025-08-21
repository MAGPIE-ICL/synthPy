
import jax.numpy as jnp



def distance(x,y,z):
    return jnp.sqrt(x**2 + y**2 + z**2)

def grid(lengths, n_cells):
    """
    lengths and n_cells should both be arrays of length 3
    """
    xlen, ylen, zlen = lengths[0], lengths[1], lengths[2]
    x_n, y_n, z_n = n_cells[0], n_cells[1], n_cells[2]
    x = jnp.linspace(-xlen/2, xlen/2, x_n)
    y = jnp.linspace(-ylen/2, ylen/2, y_n)
    z = jnp.linspace(-zlen/2, zlen/2, z_n)
    XX, YY, ZZ = jnp.meshgrid(x, y, z, indexing = 'ij')
    return XX, YY, ZZ


def spherical(radii, ne, rho, lengths, n_cells):
    radii = (0,) + radii
    ne_grid = jnp.zeros(n_cells, n_cells, n_cells)
    rho_grid_list  = len(rho)*[ne_grid]
    distance_grid = distance(domain.XX, domain.YY, domain.ZZ)
    for i in range (0, len(radii)-1):
        ne_grid[(radii[i] < distance_grid) & (distance_grid<= radii[i+1])] = ne[i]
        rho_grid_list[i][(radii[i] < distance_grid) & (distance_grid<= radii[i+1])] = rho[i]

    return ne_grid, rho_grid_list

def density_step(rho1, rho2, lengths, n_cells):
    """
    A density function which has value rho1 for y>0 and rho2 for y<=0
    """
    density = np.zeros_like(domain.YY)
    density[domain.YY > 0] = rho1
    density[domain.YY <= 0] = rho2
    return density