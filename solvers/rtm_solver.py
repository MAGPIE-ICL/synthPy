import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
Ray Transfer Matrix Solver - Modified from Jack Hare's Version
Example:

###INITIALISE RAYS###
#Rays are a 4 vector of x, theta, y, phi,
#here we initialise 10*7 randomly distributed rays
rr0=np.random.rand(6,int(1e7))
rr0[0,:]-=0.5 #rand generates [0,1], so we recentre [-0.5,0.5]
rr0[2,:]-=0.5

rr0[4,:]-=0.5 #rand generates [0,1], so we recentre [-0.5,0.5]
rr0[5,:]-=0.5

#x, θ, y, ϕ
scales=np.diag(np.array([10,0,10,0,1,1j])) #set angles to 0, collimated beam. x, y in [-5,5]. Circularly polarised beam, E_x = iE_y
rr0=np.matmul(scales, rr0)
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
'''

def m_to_mm(r):
    rr = np.ndarray.copy(r)
    rr[0::2,:]*=1e3
    return rr

def lens(r, f1,f2):
    '''4x4 matrix for a thin lens, focal lengths f1 and f2 in orthogonal axes
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    '''
    l1= np.array([[1,    0],
                [-1/f1, 1]])
    l2= np.array([[1,    0],
                [-1/f2, 1]])
    L=np.zeros((4,4))
    L[:2,:2]=l1
    L[2:,2:]=l2

    return np.matmul(L, r)

def sym_lens(r, f):
    '''
    helper function to create an axisymmetryic lens
    '''
    return lens(r, f, f)

def distance(r, d):
    '''4x4 matrix  matrix for travelling a distance d
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    '''
    d = np.array([[1, d],
                    [0, 1]])
    L=np.zeros((4,4))
    L[:2,:2]=d
    L[2:,2:]=d
    return np.matmul(L, r)

def circular_aperture(r, R):
    '''
    Rejects rays outside radius R
    '''
    filt = r[0,:]**2+r[2,:]**2 > R**2
    r[:,filt]=None
    return r

def circular_stop(r, R):
    '''
    Rjects rays inside a radius R
    '''
    filt = r[0,:]**2+r[2,:]**2 < R**2
    r[:,filt]=None
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
    filt1 = (r[0,:]**2 > Lx**2)
    filt2 = (r[2,:]**2 > Ly**2)
    filt=filt1*filt2
    r[:,filt]=None
    return r

def knife_edge(r, offset, axis, direction):
    '''
    Filters rays using a knife edge.
    Default is a knife edge in y, can also do a knife edge in x.
    '''
    if axis == 'y':
        a=2
    if axis == 'x':
        a=0
    if direction > 0:
        filt = r[a,:] > offset
    if direction < 0:
        filt = r[a,:] < offset
    if direction == 0:
        print('Direction must be <0 or >0')
    r[:,filt]=None
    return r

class Rays:
    """
    Inheritable class for ray diagnostics.
    """
    def __init__(self, r0, E = None, focal_plane = 0, L=400, R=25, Lx=18, Ly=13.5):
        """Initialise ray diagnostic.

        Args:
            r0 (4xN float array): N rays, [x, theta, y, phi]

            L (int, optional): Length scale L. First lens is at L. Defaults to 400.
            R (int, optional): Radius of lenses. Defaults to 25.
            Lx (int, optional): Detector size in x. Defaults to 18.
            Ly (float, optional): Detector size in y. Defaults to 13.5.
        """        
        self.E, self.focal_plane, self.L, self.R, self.Lx, self.Ly = E, focal_plane, L, R, Lx, Ly
        self.r0 = m_to_mm(r0)

    def histogram(self, bin_scale=10, pix_x=3448, pix_y=2574, clear_mem=False):
        """Bin data into a histogram. Defaults are for a KAF-8300.
        Outputs are H, the histogram, and xedges and yedges, the bin edges.

        Args:
            bin_scale (int, optional): bin size, same in x and y. Defaults to 10.
            pix_x (int, optional): number of x pixels in detector plane. Defaults to 3448.
            pix_y (int, optional): number of y pixels in detector plane. Defaults to 2574.
        """        
        x=self.rf[0,:]
        y=self.rf[2,:]

        x=x[~np.isnan(x)]
        y=y[~np.isnan(y)]

        self.H, self.xedges, self.yedges = np.histogram2d(x, y, 
                                           bins=[pix_x//bin_scale, pix_y//bin_scale], 
                                           range=[[-self.Lx/2, self.Lx/2],[-self.Ly/2,self.Ly/2]])
        self.H = self.H.T

        # Optional - clear ray attributes to save memory
        if(clear_mem):
            self.clear_rays()

    def plot(self, ax, clim=None, cmap=None):
        ax.imshow(self.H, interpolation='nearest', origin='lower', clim=clim, cmap=cmap,
                extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])

    def clear_rays(self):
        '''
        Clears the r0 and rf variables to save memory
        '''
        self.r0 = None
        self.rf = None
        
class Shadowgraphy(Rays):
    """
    Example shadowgraphy diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1 and a single lens system with M = 2. Both lenses have a f = L/2 focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    """
    def single_lens_solve(self):
        ## single lens - M = Variable (around ~2) (based on Detector position. Real experimental setup)
        r1 = distance(self.r0, 3*self.L/4 - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)      # cut off
        r3 = sym_lens(r2, self.L/2)             # lens 1
        r4 = distance(r3, 3*self.L/2)           # detector
        self.rf=r4
        
    def two_lens_solve(self):
        ## 2 lens telescope, M = 1
        r1 = distance(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)    # cut off
        r3 = sym_lens(r2, self.L/2)           # lens 1
        r4 = distance(r3, self.L*2)           # displace rays to lens 2.
        r5 = circular_aperture(r4, self.R)    # cut off
        r6 = sym_lens(r5, self.L/2)           # lens 2
        r7 = distance(r6, self.L)             # displace rays to detector
        self.rf = r7
    
class Schlieren(Rays):
    """
    Example dark field schlieren diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1. Both lenses have a f = L focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    There is a circular stop placed at the focal point afte rthe first lens which rejects rays which hit the focal planes at distance less than R [mm] from the optical axis.
    """
    def DF_solve(self, R = 1):
        ## 2 lens telescope, M = 1
        r1=distance(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2=circular_aperture(r1, self.R) # cut off
        r3=sym_lens(r2, self.L) #lens 1

        r4=distance(r3, self.L) #displace rays to stop
        r5=circular_stop(r4, R = R) # stop

        r6=distance(r5, self.L) #displace rays to lens 2
        r7=circular_aperture(r6, self.R) # cut off
        r8=sym_lens(r7, self.L) #lens 2

        r9=distance(r8, self.L) #displace rays to detector
        self.rf = r9
    
    """
    Example light field schlieren diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1. Both lenses have a f = L/2 focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    There is a circular stop placed at the focal point afte rthe first lens which accepts only rays which hit the focal planes at distance less than R [mm] from the optical axis.
    """
    def LF_solve(self, R = 1):
        ## 2 lens telescope, M = 1
        r1=distance(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2=circular_aperture(r1, self.R) # cut off
        r3=sym_lens(r2, self.L) #lens 1

        r4=distance(r3, self.L) #displace rays to stop
        r5=circular_aperture(r4, R = R) # stop

        r6=distance(r5, self.L) #displace rays to lens 2
        r7=circular_aperture(r6, self.R) # cut off
        r8=sym_lens(r7, self.L) #lens 2

        r9=distance(r8, self.L) #displace rays to detector
        self.rf = r9
        
class Refractometry(Rays):
    """
    Example of Imaging Refractometer. Inherits from Rays, has custom solve method.
    Implements a spherical lens with focal length f1 = L/2 and M = 2 for the spatial axis and a cylindrical lens
    with focal length f1 and f2.
    """

    def incoherent_solve(self):
        ## Imaging the spatial axis - M = 2
        r1 = distance(self.r0, 3*self.L/4 - self.focal_plane) #displace rays to lens 1. Accounts for object with depth
        r2 = circular_aperture(r1, self.R)      # cut off
        r3 = sym_lens(r2, self.L/2)             # lens 1 - spherical
        r4 = distance(r3, 3*self.L/2)           # displace rays to lens 2 - hybrid
        r5 = rect_aperture(r4, 15, 30)          # rectangular lens cut-off
        r6 = circular_aperture(r5, self.R)      # cut off
        r7 = lens(r6, self.L/3, self.L/2)       # lens 2 - hybrid lens
        r8 = distance(r7, self.L)               # displace rays to detector
        self.rf = r8

    def coherent_solve(self, wl = 1064e-9):
        ## Imaging the spatial axis - M = 2 - Coherent Implementation of the Refractometer
        r1 = distance(self.r0, 3*self.L/4 - self.focal_plane)
        # propagate E field
        dx = r1[0,:] - self.r0[0,:]
        dy = r1[2,:] - self.r0[2,:]
        lwl = wl
        k = 2 * np.pi / lwl
        E0 = self.E*np.exp(1.0j * k * (np.sqrt(dx**2 + dy**2)))
        del dx
        del dy
        
        r2 = circular_aperture(r1, self.R)      # cut off
        r3 = sym_lens(r2, self.L/2)             # lens 1 - spherical
        dx = r3[0,:] - r2[0,:]
        dy = r3[2,:] - r2[2,:]
        k = 2* np.pi / lwl
        E1 = E0*np.exp(1.0j * k * (np.sqrt(dx**2 + dy**2)))
        del dx
        del dy
        
        r4 = distance(r3, 3*self.L/2)           # displace rays to lens 2 - hybrid
        r5 = circular_aperture(r4, self.R)      # cut off
        dx = r5[0,:] - r4[0,:]
        dy = r5[2,:] - r4[2,:]
        k = 2* np.pi / lwl
        E2 = E1*np.exp(1.0j * k * (np.sqrt(dx**2 + dy**2)))
        del dx
        del dy

        r6 = lens(r5, self.L/3, self.L/2)       # lens 2 - hybrid lens
        dx = r6[0,:] - r5[0,:]
        dy = r6[2,:] - r5[2,:]
        k = 2* np.pi / lwl
        E3 = E2*np.exp(1.0j * k * (np.sqrt(dx**2 + dy**2)))

        r7 = distance(r6, self.L)               # displace rays to detector
        #r8=rect_aperture(self.Lx/2,self.Ly/2,r7) # detector cutoff
        dx = r7[0,:] - r6[0,:]
        dy = r7[2,:] - r6[2,:]
        k = 2* np.pi / lwl
        E4 = E3*np.exp(1.0j * k * (np.sqrt(dx**2 + dy**2)))
        self.rE = E4
        self.rf = r7
    
    def refractogram(self, bin_scale=1, pix_x=3448, pix_y=2574, clear_mem=False):
        """Bin data into a histogram. Defaults are for a KAF-8300.
        Outputs are H, the histogram, and xedges and yedges, the bin edges.

        Args:
            bin_scale (int, optional): bin size, same in x and y. Defaults to 1.
            pix_x (int, optional): number of x pixels in detector plane. Defaults to 3448.
            pix_y (int, optional): number of y pixels in detector plane. Defaults to 2574.
        """        
        x=self.rf[0,:]
        y=self.rf[2,:]

        x_bins = np.linspace(-self.Lx//2,self.Lx//2, pix_x // bin_scale)
        y_bins = np.linspace(-self.Ly//2, self.Ly//2 , pix_y // bin_scale)
        
        amplitude_x = np.zeros((len(y_bins)-1, len(x_bins)-1), dtype=complex)
        amplitude_y = np.zeros((len(y_bins)-1, len(x_bins)-1), dtype=complex)

        x_indices = np.digitize(self.rf[0,:], x_bins) - 1
        y_indices = np.digitize(self.rf[2,:], y_bins) - 1

        for i in range(self.rf.shape[1]):
            if 0 <= x_indices[i] < amplitude_x.shape[1] and 0 <= y_indices[i] < amplitude_x.shape[0]:
                amplitude_x[y_indices[i], x_indices[i]] += self.rE[0, i]
                amplitude_y[y_indices[i], x_indices[i]] += self.rE[1, i]

        amplitude = np.sqrt(np.real(amplitude_x)**2 + np.real(amplitude_y)**2)
        # amplitude_normalised = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min()) # this line needs work and is currently causing problems
        self.H = amplitude


class Interferometry(Rays):
    '''
    Simple class to keep all the ray properties together
    '''           
    def two_lens_solve(self, wl = 532e-9):
        ## 2 lens telescope, M = 1
        r1 = distance(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        # propagate E field
        dx = r1[0,:] - self.r0[0,:]
        dy = r1[2,:] - self.r0[2,:]
        lwl = wl
        k = 2* np.pi / lwl
        E0 = self.E*np.exp(1.0j * k * (np.sqrt(dx**2 + dy**2)))
        del dx
        del dy

        r2 = circular_aperture(r1, self.R)    # cut off
        r3 = sym_lens(r2, self.L/2)           # lens 1
        dx = r3[0,:] - r2[0,:]
        dy = r3[2,:] - r2[2,:]
        k = 2* np.pi / lwl
        E1 = E0*np.exp(1.0j * k * (np.sqrt(dx**2 + dy**2)))
        del dx
        del dy

        r4 = distance(r3, self.L*2)           # displace rays to lens 2.
        dx = r4[0,:] - r3[0,:]
        dy = r4[2,:] - r3[2,:]
        k = 2* np.pi / lwl
        E2 = E1*np.exp(1.0j * k * (np.sqrt(dx**2 + dy**2)))
        del dx
        del dy

        r5 = circular_aperture(r4, self.R)    # cut off
        r6 = sym_lens(r5, self.L/2)           # lens 2
        dx = r6[0,:] - r5[0,:]
        dy = r6[2,:] - r5[2,:]
        k = 2* np.pi / lwl
        E3 = E2*np.exp(1.0j * k * (np.sqrt(dx**2 + dy**2)))
        del dx
        del dy
        
        r7 = distance(r6, self.L)             # displace rays to detector
        dx = r7[0,:] - r6[0,:]
        dy = r7[2,:] - r6[2,:]
        k = 2* np.pi / lwl
        E4 = E3*np.exp(1.0j * k * (np.sqrt(dx**2 + dy**2)))
        del dx
        del dy
        self.rE = E4
        self.rf = r7
    
    def interferogram(self, bin_scale=1, pix_x=3448, pix_y=2574, clear_mem=False):
        """Bin data into a histogram. Defaults are for a KAF-8300.
        Outputs are H, the histogram, and xedges and yedges, the bin edges.

        Args:
            bin_scale (int, optional): bin size, same in x and y. Defaults to 1.
            pix_x (int, optional): number of x pixels in detector plane. Defaults to 3448.
            pix_y (int, optional): number of y pixels in detector plane. Defaults to 2574.
        """        
        x=self.rf[0,:]
        y=self.rf[2,:]

        x_bins = np.linspace(-self.Lx//2,self.Lx//2, pix_x // bin_scale)
        y_bins = np.linspace(-self.Ly//2, self.Ly //2 , pix_y // bin_scale)
        
        amplitude_x = np.zeros((len(y_bins)-1, len(x_bins)-1), dtype=complex)
        amplitude_y = np.zeros((len(y_bins)-1, len(x_bins)-1), dtype=complex)

        x_indices = np.digitize(self.rf[0,:], x_bins) - 1
        y_indices = np.digitize(self.rf[2,:], y_bins) - 1

        for i in range(self.rf.shape[1]):
            if 0 <= x_indices[i] < amplitude_x.shape[1] and 0 <= y_indices[i] < amplitude_x.shape[0]:
                amplitude_x[y_indices[i], x_indices[i]] += self.rE[0, i]
                amplitude_y[y_indices[i], x_indices[i]] += self.rE[1, i]

        amplitude = np.sqrt(np.real(amplitude_x)**2 + np.real(amplitude_y)**2)
        
        # amplitude_normalised = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min()) # this line needs work and is currently causing problems
        self.H = amplitude