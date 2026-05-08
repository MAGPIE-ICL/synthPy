
import jax.numpy as jnp
import field_generator.gaussian3D as g3

def distance(x,y,z):
    """
    Calculates the Euclidean distance from the origin for given coordinates.
    
    :param x: x coordinate or array
    :type x: float or jax.Array
    
    :param y: y coordinate or array
    :type y: float or jax.Array
    
    :param z: z coordinate or array
    :type z: float or jax.Array
    
    :return: Euclidean distance
    :rtype: float or jax.Array
    """
    return jnp.sqrt(x**2 + y**2 + z**2)

#lengths should both be an array of length 3
#n_cells can either be an array of length 3 or a float/int

def grid(lengths, n_cells):
    """
    Generates a 3D meshgrid based on lengths and cell counts.
    
    :param lengths: Array of length 3 specifying the dimensions
    :type lengths: array-like
    
    :param n_cells: Number of cells per dimension, can be scalar or array of length 3
    :type n_cells: float, int, or array-like
    
    :return: A tuple of 3D arrays (XX, YY, ZZ) representing the grid
    :rtype: tuple of jax.Array
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
    """
    Creates a spherical density distribution on a grid.
    
    :param radii: Array of radii for the spherical shells
    :type radii: array-like
    
    :param ne: Array of electron densities for each shell
    :type ne: array-like
    
    :param rho: Array of mass densities for each shell
    :type rho: array-like
    
    :param lengths: Domain lengths
    :type lengths: array-like
    
    :param n_cells: Number of cells in each dimension
    :type n_cells: array-like or int
    
    :return: Electron density grid and a list of mass density grids
    :rtype: tuple
    """
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
    
    :param rho1: Density for y > 0
    :type rho1: float
    
    :param rho2: Density for y <= 0
    :type rho2: float
    
    :param lengths: Domain lengths
    :type lengths: array-like
    
    :param n_cells: Number of cells in each dimension
    :type n_cells: array-like or int
    
    :return: Density grid
    :rtype: jax.Array
    """
    _, YY, _ = grid(lengths, n_cells)
    density = jnp.full_like(YY, rho1)
    density = density.at[YY <= 0].set(rho2)
    return density

def turbulent_box(ne_0, max_pert, l_max, l_min, extent, n_cells, power = -5/3):
    """
    Generates a 3D turbulent density box with a specified power spectrum.
    
    :param ne_0: Base electron density
    :type ne_0: float
    
    :param max_pert: Maximum perturbation amplitude
    :type max_pert: float
    
    :param l_max: Maximum wavelength
    :type l_max: float
    
    :param l_min: Minimum wavelength
    :type l_min: float
    
    :param extent: Extent of the domain
    :type extent: float
    
    :param n_cells: Number of cells (must be even)
    :type n_cells: int
    
    :param power: Power law exponent for the turbulence spectrum
    :type power: float, default: -5/3
    
    :return: 3D electron density grid with turbulence
    :rtype: jax.Array
    """
    #NB: n_cells must be even
    def k_func(k):
        """
        Calculates k-space function value based on power law.
        
        :param k: Wavenumber
        :type k: float or jax.Array
        
        :return: Scaled wavenumber
        :rtype: float or jax.Array
        """
        return k**(power)
    
    field = g3.gaussian3D(k_func)
    noise3D = field.domain_fft(l_max, l_min, extent, n_cells//2, factor = 1.0)

    ne = ne_0 + noise3D*max_pert
    return ne