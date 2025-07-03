import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import jax
import jax.numpy as jnp

from propagator import ray_to_Jonesvector

#jax.tree_util.tree_leaves(x, is_leaf = lambda x: x is None)

'''
(rtm_solver)
Ray Transfer Matrix Solver - Modified from Jack Hare's Version
Example:

###INITIALISE RAYS###
#Rays are a 4 vector of x, theta, y, phi - 6 vector (E_x and E_y added) if E field is taken into account in solver
#here we initialise 10*7 randomly distributed rays
rr0=jnp.random.rand(6,int(1e7))
rr0[0,:]-= 0.5 #rand generates [0,1], so we recentre [-0.5,0.5]
rr0[2,:]-= 0.5

rr0[4,:]-= 0.5 #rand generates [0,1], so we recentre [-0.5,0.5]
rr0[5,:]-= 0.5

#x, θ, y, ϕ
scales=jnp.diag(jnp.array([10, 0, 10, 0, 1, 1j])) #set angles to 0, collimated beam. x, y in [-5,5]. Circularly polarised beam, E_x = iE_y
rr0=jnp.matmul(scales, rr0)
r0=circular_aperture(5, rr0) #cut out a circle

### Shadowgraphy, no polarisation
## object_length: determines where the focal plane is. If you object is 10 mm long, object length = 5 will
## make the focal plane in the middle of the object. Yes, it's a bad variable name.
s = Shadowgraphy(rr0, L = 400, R = 25, object_length=5)
s.solve()
s.histogram(bin_scale = 25)
fig, axs = plt.subplots(figsize=(6.67, 6))

cm='gray'
clim=[0,100]

s.plot(axs, clim=clim, cmap=cm)

###CREATE A SHOCK PAIR FOR TESTING###
def α(x, n_e0, w, x0, Dx, l=10):
    dn_e = n_e0*(jnp.tanh((x+Dx+x0)/w)**2-jnp.tanh((x-Dx+x0)/w)**2)
    n_c=1e21
    a = 0.5* l/n_c * dn_e
    return a

def ne(x,n_e0, w, Dx, x0):
    return n_e0*(jnp.tanh((x+Dx+x0)/w)-jnp.tanh((x-Dx+x0)/w))

def ne_ramp(y, ne_0, scale):
    return ne_0*10**(y/scale)

# Parameters for shock pair
w=0.1
Dx=1
x0=0
ne0=1e18
s=5

x=jnp.linspace(-5,5,1000)
y=jnp.linspace(-5,5,1000)

a=α(x, n_e0=ne0, w=w, Dx=Dx, x0=x0)
n=ne(x, n_e0=ne0, w=w, Dx=Dx, x0=x0)
ne0s=ne_ramp(y, ne_0=ne0, scale=s)

nn=jnp.array([ne(x, n_e0=n0, w=w, Dx=Dx, x0=x0) for n0 in ne0s])
nn=jnp.rot90(nn)

###PLOT SHOCKS###
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(6.67/2, 2))

ax1.imshow(nn, clim=[1e16,1e19], cmap='inferno')
ax1.axis('off')
ax2.plot(x, n/5e18, label=r'$n_e$')
ax2.plot(x, a*57, label=r'$\alpha$')

ax2.set_xlim([-5,5])
ax2.set_xticks([])
ax2.set_yticks([])
ax2.legend(borderpad=0.5, handlelength=1, handletextpad=0.2, labelspacing=0.2)
fig.subplots_adjust(left=0, bottom=0.14, right=0.98, top=0.89, wspace=0.1, hspace=None)

###DEFLECT RAYS###
r0[3,:]=α(r0[2,:],n_e0=ne_ramp(r0[0,:], ne0, s), w=w, Dx=Dx, x0=x0)

###SOLVE FOR RAYS###
b=refractometerRays(r0)
sh=ShadowgraphyRays(r0)
sc=SchlierenRays(r0)

sh.solve(displacement=10)
sh.histogram(bin_scale=10)
sc.solve()
sc.histogram(bin_scale=10)
b.solve()
b.histogram(bin_scale=10)

###PLOT DATA###
fig, axs = plt.subplots(1,3,figsize=(6.67, 1.8))

cm='gray'
clim=[0,100]

sh.plot(axs[1], clim=clim, cmap=cm)
#axs[0].imshow(nn.T, extent=[-5,5,-5,5])
sc.plot(axs[0], clim=clim, cmap=cm)
b.plot(axs[2], clim=clim, cmap=cm)

for ax in axs:
    ax.axis('off')
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=None)
'''

def m_to_mm(r):
    rr = jnp.copy(r)
    rr = rr.at[0::2, :].set(rr[0::2, :] * 1e3)

    return rr

def mm_to_m(r):
    rr = jnp.copy(r)
    rr = rr.at[0::2, :].set(rr[0::2, :] * 1e-3)

    return rr

def lens(r, f1, f2):
    '''
    4x4 matrix for a thin lens, focal lengths f1 and f2 in orthogonal axes
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    '''

    l1 = np.array([[1, 0],
            [-1 / f1, 1]])
    l2 = np.array([[1, 0],
            [-1 / f2, 1]])

    L = np.zeros((4, 4))
    L[:2, :2] = l1
    L[2:, 2:] = l2

    return jnp.matmul(L, r)

def sym_lens(r, f):
    '''
    Helper function to create an axisymmetryic lens
    '''

    return lens(r, f, f)

def distance(r, d):
    '''4x4 matrix  matrix for travelling a distance d
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    '''

    d = np.array([[1, d],
                  [0, 1]])

    L = np.zeros((4, 4))

    L[:2, :2] = d
    L[2:, 2:] = d

    return jnp.matmul(L, r)

def circular_aperture(r, R, E = None):
    '''
    Rejects rays outside radius R
    '''

    filt = r[0, :] ** 2 + r[2, :] ** 2 > R ** 2
    # if you want to reject rays outside of the radius, then when filt is true you should set equal to None
    r = r.at[:, filt].set(jnp.nan)

    if E is not None:
        # double checks it is a jnp array (matrix in this case, 9*Np) and converts if not
        # - had issues in the past, is just a 'just in case' thing - likely useless
        E = jnp.asarray(E).at[:, filt].set(jnp.nan) # filters out all in [:, i] if i contains any jnp.nan values in [:, i] axis

        return r, E

    return r

def circular_stop(r, R):
    '''
    Rejects rays inside a radius R
    '''

    filt = r[0, :] ** 2 + r[2,:] ** 2 < R ** 2
    r = r.at[:, filt].set(jnp.nan)

    return r

def annular_stop(r, R1, R2):
    '''
    Rejects rays which fall between R1 and R2
    '''

    filt1 = (r[0,:]**2+r[2,:]**2 > R1**2)
    filt2 = (r[0,:]**2+r[2,:]**2 < R2**2)
    filt = (filt1 & filt2)

    return filt

def rect_aperture(r, Lx, Ly):
    '''
    Rejects rays outside a rectangular aperture, total size 2*Lx x 2*Ly
    '''

    filt1 = (r[0, :] ** 2 > Lx ** 2)
    filt2 = (r[2, :] ** 2 > Ly ** 2)

    filt = filt1 * filt2
    r = r.at[:, filt].set(jnp.nan)

    return r

def knife_edge(r, offset, axis, direction):
    '''
    Filters rays using a knife edge.
    Default is a knife edge in y, can also do a knife edge in x.
    '''

    if axis == 'y':
        a = 2
    if axis == 'x':
        a = 0

    if direction > 0:
        filt = r[a,:] > offset
    if direction < 0:
        filt = r[a,:] < offset
    if direction == 0:
        print('Direction must be < 0 or > 0')

    r = r.at[:, filt].set(jnp.nan)

    return r

def clear_rays(self):
    '''
    Clears the r0, rf and Jf variables to save memory
    '''
    # does this actually save memory in the best way?
    # would it be better to del self.r_ instead?

    self.r0 = None
    self.rf = None
    self.Jf = None

def ray(x, θ, y, ϕ):
    '''
    Returns a 4x1 matrix representing a ray. Spatial units must be consistent, angular units in radians.
    '''

    return sym.Matrix([x, θ, y, ϕ])

def d2r(d):
    # helper function, degrees to radians
    return d * jnp.pi / 180

class Diagnostic:
    """
    Inheritable class for ray diagnostics.
    """

    # this is in mm's not metres - self.rf is converted to mm's (not sure if everything else is covered though)
    def __init__(self, wavelength, focal_plane = 0, L = 400, R = 25, Lx = 18, Ly = 13.5):
        """
        Initialise ray diagnostic.

        Args:
            r0 (4xN float array): N rays, [x, theta, y, phi]

            L (int, optional): Length scale L. First lens is at L. Defaults to 400.
            R (int, optional): Radius of lenses. Defaults to 25.
            Lx (int, optional): Detector size in x. Defaults to 18.
            Ly (float, optional): Detector size in y. Defaults to 13.5.
        """     

        self.wavelength, self.focal_plane, self.L, self.R, self.Lx, self.Ly = wavelength, focal_plane, L, R, Lx, Ly

        # these HAVE to stay... for some reason - not entirely sure why you can't just reference self.Beam.r_ directly
        # if you can make it without the memory duplication work please do, else DON'T REMOVE!

        #self.rf = self.Beam.rf
        #self.Jf = self.Beam.Jf

        # however, doesn't have to be done manually now as already sorted in propagator.py, therefore no more duplication
        # still odd though... (hence the keeping of the comment)

        self.r0 = m_to_mm(self.rf)

    def propagate_E(self, r1, r0):
        dx = r1[0, :] - r0[0, :]
        dy = r1[2, :] - r0[2, :]

        k = 2 * jnp.pi / self.wavelength

        self.Jf = self.Jf.at[:, :].set(self.Jf[:, :] * jnp.exp(1.0j * k * jnp.sqrt(dx ** 2 + dy ** 2)))

    def histogram(self, bin_scale = 1, pix_x = 3448, pix_y = 2574, clear_mem = False):
        '''
        Bin data into a histogram. Defaults are for a KAF-8300.
        Outputs are H, the histogram, and xedges and yedges, the bin edges.

        Args:
            bin_scale (int, optional): bin size, same in x and y. Defaults to 1.
            pix_x (int, optional): number of x pixels in detector plane. Defaults to 3448.
            pix_y (int, optional): number of y pixels in detector plane. Defaults to 2574.
        '''
    
        x = self.rf[0, :]
        y = self.rf[2, :]

        print("\nrf size expected: (", len(x), ", ", len(y), ")", sep='')

        # means that jnp.isnan(a) returns True when a is not Nan
        # ensures that x & y are the same length, if output of either is Nan then will not try to render ray in histogram
        mask = ~jnp.isnan(x) & ~jnp.isnan(y)

        x = x[mask]
        y = y[mask]

        print("rf after clearing nan's: (", len(x), ", ", len(y), ")", sep='')

        self.H, self.xedges, self.yedges = jnp.histogram2d(x, y, bins=[pix_x // bin_scale, pix_y // bin_scale], range=[[-self.Lx / 2, self.Lx / 2],[-self.Ly / 2, self.Ly / 2]])
        self.H = self.H.T

        #Optional - clear ray attributes to save memory
        if clear_mem:
            clear_rays(self)

    def plot(self, ax, clim = None, cmap = None):
        ax.imshow(self.H, interpolation='nearest', origin='lower', clim=clim, cmap=cmap, extent = [self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])

    def histogram_legacy(self, bin_scale = 1, pix_x = 3448, pix_y = 2574, clear_mem = False):
        # repeated across many functions, have made a wrapper for it instead of repeats to preserve backwards compatability
        # was this replaced by jnp.histogram2d function?
        # this function is far slower for a general histogram than the new function - yet is used for refractogram and interferogram so kept to be wrapped for those
        x_bins = jnp.linspace(-self.Lx // 2, self.Lx // 2, pix_x // bin_scale)
        y_bins = jnp.linspace(-self.Ly // 2, self.Ly // 2, pix_y // bin_scale)

        amplitude_x = jnp.zeros((len(y_bins) - 1, len(x_bins) - 1), dtype = complex)
        amplitude_y = jnp.zeros((len(y_bins) - 1, len(x_bins) - 1), dtype = complex)

        x_indices = jnp.digitize(self.rf[0, :], x_bins) - 1
        y_indices = jnp.digitize(self.rf[2, :], y_bins) - 1

        for i in range(self.rf.shape[1]):
            if 0 <= x_indices[i] < amplitude_x.shape[1] and 0 <= y_indices[i] < amplitude_x.shape[0]:
                # jax arrays are immutable - fix later
                amplitude_x = amplitude_x.at[y_indices[i], x_indices[i]].set(amplitude_x[y_indices[i], x_indices[i]] + self.Jf[0, i])
                amplitude_y = amplitude_y.at[y_indices[i], x_indices[i]].set(amplitude_y[y_indices[i], x_indices[i]] + self.Jf[1, i])

        amplitude = jnp.sqrt(jnp.real(amplitude_x) ** 2 + jnp.real(amplitude_y) ** 2)
        # amplitude_normalised = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min()) # this line needs work and is currently causing problems
        self.H = amplitude

class Shadowgraphy(Diagnostic):
    """
    Example shadowgraphy diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1 and a single lens system with M = 2. Both lenses have a f = L/2 focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    """

    def single_lens_solve(self):
        ## single lens - M = Variable (around ~2) (based on Detector position. Real experimental setup)
        r1 = distance(self.r0, 3 * self.L / 4 - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)      # cut off
        r3 = sym_lens(r2, self.L / 2)             # lens 1
        r4 = distance(r3, 3*self.L / 2)           # detector
        self.rf = r4

    def two_lens_solve(self):
        ## 2 lens telescope, M = 1
        r1 = distance(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)    # cut off
        r3 = sym_lens(r2, self.L / 2)           # lens 1
        r4 = distance(r3, self.L * 2)           # displace rays to lens 2.
        r5 = circular_aperture(r4, self.R)    # cut off
        r6 = sym_lens(r5, self.L / 2)           # lens 2
        r7 = distance(r6, self.L)             # displace rays to detector
        self.rf = r7
    
class Schlieren(Diagnostic):
    """
    Example dark field schlieren diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1. Both lenses have a f = L focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    There is a circular stop placed at the focal point after the first lens which rejects rays which hit the focal planes at distance less than R [mm] from the optical axis.
    """

    def DF_solve(self, R = 1):
        ## 2 lens telescope, M = 1
        r1 = distance(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R) # cut off
    
        r3 = sym_lens(r2, self.L) #lens 1

        r4 = distance(r3, self.L) #displace rays to stop

        # this and positioning of lenses means schlieren ends up with less usable rays than other methods
        r5 = circular_stop(r4, R = R) # stop - blocker at focal point after the first lens of size R (1 mm?)

        r6 = distance(r5, self.L) #displace rays to lens 2

        r7 = circular_aperture(r6, self.R) # cut off

        r8 = sym_lens(r7, self.L) #lens 2

        r9 = distance(r8, self.L) #displace rays to detector

        self.rf = r9
    
    """
    Example light field schlieren diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1. Both lenses have a f = L/2 focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    There is a circular stop placed at the focal point afte rthe first lens which accepts only rays which hit the focal planes at distance less than R [mm] from the optical axis.
    """

    def LF_solve(self, R = 1):
        ## 2 lens telescope, M = 1
        r1 = distance(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R) # cut off
        r3 = sym_lens(r2, self.L) #lens 1

        r4 = distance(r3, self.L) #displace rays to stop
        r5 = circular_aperture(r4, R = R) # stop

        r6 = distance(r5, self.L) #displace rays to lens 2
        r7 = circular_aperture(r6, self.R) # cut off
        r8 = sym_lens(r7, self.L) #lens 2

        r9 = distance(r8, self.L) #displace rays to detector
        self.rf = r9
        
class Refractometry(Diagnostic):
    '''
    Example of Imaging Refractometer. Inherits from Rays, has custom solve method.
    Implements a spherical lens with focal length f1 = L/2 and M = 2 for the spatial axis and a cylindrical lens
    with focal length f1 and f2.
    '''

    def incoherent_solve(self):
        ##
        ## Is there an efficient way to chain these so needlessly variables are not used without having 1 really long line
        ##

        ## Imaging the spatial axis - M = 2
        r1 = distance(self.r0, 3 * self.L / 4 - self.focal_plane) #displace rays to lens 1. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)      # cut off
        r3 = sym_lens(r2, self.L/2)             # lens 1 - spherical
        r4 = distance(r3, 3*self.L/2)           # displace rays to lens 2 - hybrid
        r5 = rect_aperture(r4, 15, 30)          # rectangular lens cut-off
        r6 = circular_aperture(r5, self.R)      # cut off
        r7 = lens(r6, self.L/3, self.L/2)       # lens 2 - hybrid lens
        r8 = distance(r7, self.L)               # displace rays to detector
        self.rf = r8

    def coherent_solve(self):
        ## Imaging the spatial axis - M = 2 - Coherent Implementation of the Refractometer
        r1 = distance(self.r0, 3 * self.L / 4 - self.focal_plane)
        # propagate E field
        self.propagate_E(r1, self.r0)
        r2, self.Jf = circular_aperture(self.r0, self.R, E = self.Jf)      # cut off
        r3 = sym_lens(r2, self.L/2)          # lens 1 - spherical
        self.propagate_E(r3, r2)
        r4 = distance(r3, 3*self.L/2)
        self.propagate_E(r4, r3)                 # displace rays to lens 2 - hybrid
        r5, self.Jf = circular_aperture(r4, self.R, E = self.Jf)      # cut off
        r6 = lens(r5, self.L/3, self.L/2)       # lens 2 - hybrid lens
        self.propagate_E(r6, r5)

        r7 = distance(r6, self.L)               # displace rays to detector
        self.propagate_E(r7, r6)
        self.rf = r7
    
    def refactogram(self, bin_scale = 1, pix_x = 3448, pix_y = 2574, clear_mem = False):
        self.histogram_legacy(bin_scale = bin_scale, pix_x = pix_x, pix_y = pix_y, clear_mem = clear_mem)

class Interferometry(Diagnostic):
    '''
    Simple class to keep all the ray properties together
    '''

    def interfere_ref_beam(self, n_fringes, deg):
        '''
        Input beam ray positions and electric field component, and desired angle of evenly spaced background fringes. 
        Deg is angle in degrees from the vertical axis

        Returns:
            'Interfered with' E field
        '''

        if deg >= 45:
            deg = - jnp.abs(deg - 90)

        rad = deg* jnp.pi /180 #deg to rad
        y_weight = jnp.arctan(rad) #take x_weight is 1
        x_weight = jnp.sqrt(1-y_weight**2)

        ref_beam = jnp.exp(2 * n_fringes / 3 * 1.0j * (x_weight * self.rf[0,:] + y_weight * self.rf[2,:]))

        self.Jf = self.Jf.at[1,:].set(self.Jf[1,:] + ref_beam) # assume ref_beam is polarised in y
    
    def bkg(self, domain_length, n_fringes, deg):
        rr0, E0 = ray_to_Jonesvector(self.Beam)
        E = self.Jf.copy() #temporarily store E field in another variable
        self.Jf = E0

        # assuming reference is recombined with the probe beam at the exit of the domain (should be changed)
        self.interfere_ref_beam(n_fringes, deg)
        ## 2 lens telescope, M = 1
        r1 = distance(rr0, self.L + domain_length) #displace rays to lens. Accounts for object with depth
        # propagate E field
        self.propagate_E(r1, rr0)
        r2, self.Jf = circular_aperture(r1, self.R, E = self.Jf)    # cut off
        r3 = sym_lens(r2, self.L/2)           # lens 1
        self.propagate_E(r3, r2)

        r4 = distance(r3, self.L*2)           # displace rays to lens 2.
        self.propagate_E(r4, r3)
        r5, self.Jf = circular_aperture(r4, self.R, E = self.Jf)    # cut off
        r6 = sym_lens(r5, self.L/2)                             # lens 2
        self.propagate_E(r6, r5)
        
        r7 = distance(r6, self.L)             # displace rays to detector
        self.propagate_E(r7, r6)
        rf = r7

        self.histogram(self)
        self.bkg_signal = self.H

        self.Jf = E #restore E field

    def two_lens_solve(self):
        # assuming reference is recombined with the probe beam at the exit of the domain (should be changed)
        self.interfere_ref_beam(10, 20)
        ## 2 lens telescope, M = 1
        r1 = distance(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth

        # propagate E field
        self.propagate_E(r1, self.r0)
        r2, self.Jf = circular_aperture(r1, self.R, E = self.Jf)    # cut off

        r3 = sym_lens(r2, self.L/2)           # lens 1
        self.propagate_E(r3, r2)

        r4 = distance(r3, self.L*2)           # displace rays to lens 2.
        self.propagate_E(r4, r3)

        r5, self.Jf = circular_aperture(r4, self.R, E = self.Jf)    # cut off

        r6 = sym_lens(r5, self.L/2)                             # lens 2
        self.propagate_E(r6, r5)
        
        r7 = distance(r6, self.L)             # displace rays to detector
        self.propagate_E(r7, r6)

        self.rf = r7

    def interferogram(self, bin_scale = 1, pix_x = 3448, pix_y = 2574, clear_mem = False):
        self.histogram_legacy(bin_scale = bin_scale, pix_x = pix_x, pix_y = pix_y, clear_mem = clear_mem)