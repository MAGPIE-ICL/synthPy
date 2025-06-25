import numpy as np

class Propagator:
    
    def __init__(self, ScalarDomain, Beam, inv_brems = False, phaseshift = False):
        self.ScalarDomain = ScalarDomain
        self.Beam = Beam
        self.inv_brems = inv_brems
        self.phaseshift = phaseshift
        # finish initialising the beam position using the scalardomain edge position
        axes = ['x', 'y', 'z']
        index = np.where(axes == Beam.probing_direction)[0][0]
        self.integration_length = ScalarDomain.lengths[index]
        self.extent = integration_length/2

        Beam.init_beam(ne_extent)       #is this a second call instance for init_beam()?

# The following functions are methods to be called by the solve()
    def calc_dndr(self):
        """Generate interpolators for derivatives.

        Args:
            lwl (float, optional): laser wavelength. Defaults to 1053e-9 m.
        """

        lwl = self.Beam.wavelength

        self.omega = 2*np.pi*(c/lwl)
        nc = 3.14207787e-4*self.omega**2

        # Find Faraday rotation constant http://farside.ph.utexas.edu/teaching/em/lectures/node101.html
        if (self.ScalarDomain.B_on):
            self.VerdetConst = 2.62e-13*lwl**2 # radians per Tesla per m^2

        self.ne_nc = np.array(self.ScalarDomain.ne/nc, dtype = np.float32) #normalise to critical density
        
        #More compact notation is possible here, but we are explicit
        self.dndx = -0.5*c**2*np.gradient(self.ne_nc,self.ScalarDomain.x,axis=0)
        self.dndy = -0.5*c**2*np.gradient(self.ne_nc,self.ScalarDomain.y,axis=1)
        self.dndz = -0.5*c**2*np.gradient(self.ne_nc,self.ScalarDomain.z,axis=2)
        
        self.dndx_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.dndx, bounds_error = False, fill_value = 0.0)
        self.dndy_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.dndy, bounds_error = False, fill_value = 0.0)
        self.dndz_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.dndz, bounds_error = False, fill_value = 0.0)


    def omega_pe(ne):
        '''Calculate electron plasma freq. Output units are rad/sec. From nrl pp 28'''

        return 5.64e4*np.sqrt(ne)

    # NRL formulary inverse brems - cheers Jack Halliday for coding in Python
    # Converted to rate coefficient by multiplying by group velocity in plasma
    def kappa(self):
        # Useful subroutines
        def v_the(Te):
            '''Calculate electron thermal speed. Provide Te in eV. Retrurns result in m/s'''

            return 4.19e5*np.sqrt(Te)

        def V(ne, Te, Z, omega):
            o_pe  = omega_pe(ne)
            o_max = np.copy(o_pe)
            o_max[o_pe < omega] = omega
            L_classical = Z*sc.e/Te
            L_quantum = 2.760428269727312e-10/np.sqrt(Te) # sc.hbar/np.sqrt(sc.m_e*sc.e*Te)
            L_max = np.maximum(L_classical, L_quantum)

            return o_max*L_max

        def coloumbLog(ne, Te, Z, omega):
            return np.maximum(2.0,np.log(v_the(Te)/V(ne, Te, Z, omega)))

        ne_cc = self.ScalarDomain.ne*1e-6
        o_pe  = omega_pe(ne_cc)
        CL    = coloumbLog(ne_cc, self.ScalarDomain.Te, self.ScalarDomain.Z, self.omega)

        return 3.1e-5*self.ScalarDomain.Z*c*np.power(ne_cc/self.omega,2)*CL*np.power(self.ScalarDomain.Te, -1.5) # 1/s

    # Plasma refractive index
    def n_refrac(self):

        ne_cc = self.ScalarDomain.ne*1e-6
        o_pe  = omega_pe(ne_cc)
        return np.sqrt(1.0-(o_pe/self.omega)**2)

    def set_up_interps(self):
        # Electron density
        self.ne_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.ne, bounds_error = False, fill_value = 0.0)
        # Magnetic field
        if(self.B_on):
            self.Bx_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.B[:,:,:,0], bounds_error = False, fill_value = 0.0)
            self.By_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.B[:,:,:,1], bounds_error = False, fill_value = 0.0)
            self.Bz_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.B[:,:,:,2], bounds_error = False, fill_value = 0.0)
        # Inverse Bremsstrahlung
        if(self.inv_brems):
            self.kappa_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.kappa(), bounds_error = False, fill_value = 0.0)
        # Phase shift
        if(self.phaseshift):
            self.refractive_index_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.n_refrac(), bounds_error = False, fill_value = 1.0)
    
    def dndr(self,x):
        """returns the gradient at the locations x

        Args:
            x (3xN float): N [x,y,z] locations

        Returns:
            3 x N float: N [dx,dy,dz] electron density gradients
        """

        grad = np.zeros_like(x)
        grad[0,:] = self.dndx_interp(x.T)
        grad[1,:] = self.dndy_interp(x.T)
        grad[2,:] = self.dndz_interp(x.T)

        return grad

    # Attenuation due to inverse bremsstrahlung
    def atten(self,x):
        if(self.inv_brems):
            return self.kappa_interp(x.T)
        else:
            return 0.0

    # Phase shift introduced by refractive index
    def phase(self,x):
        if(self.phaseshift):
            self.refractive_index_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.n_refrac(), bounds_error = False, fill_value = 1.0)
            return self.omega*(self.refractive_index_interp(x.T)-1.0)
        else:
            return 0.0
    
    def get_ne(self,x):
        return self.ne_interp(x.T)

    def get_B(self,x):
        return np.array([self.Bx_interp(x.T),self.By_interp(x.T),self.Bz_interp(x.T)])

    def neB(self,x,v):
        """returns the VerdetConst ne B.v

        Args:
            x (3xN float): N [x,y,z] locations
            v (3xN float): N [vx,vy,vz] velocities

        Returns:
            N float: N values of ne B.v
        """

        if(self.ScalarDomain.B_on):
            ne_N = self.get_ne(x)
            Bv_N = np.sum(self.get_B(x)*v,axis=0)
            pol  = self.VerdetConst*ne_N*Bv_N
        else:
            pol = 0.0

        return pol

    def solve(self):
        # Need to make sure all rays have left volume
        # Conservative estimate of diagonal across volume
        # Then can backproject to surface of volume
        s0 = self.Beam.s0

        t = np.linspace(0.0,np.sqrt(8.0)*self.extent/c,2)

        print(s0.shape)
        print(s0.size)
        s0 = s0.flatten() #odeint insists
        print(s0.shape)
        print(s0.size)

        start = time()

        dsdt_ODE = lambda t, y: dsdt(t, y, self)
        sol = solve_ivp(dsdt_ODE, [0,t[-1]], s0, t_eval=t)
        finish = time()
        self.duration = finish - start

        Np = s0.size//9
        self.Beam.rf = sol.y[:,-1].reshape(9,Np)

        self.Beam.rf, self.Beam.Jf = ray_to_Jonesvector(self.Beam.rf, self.extent, probing_direction = self.Beam.probing_direction)
        
    
    def solve_at_depth(self, z):
        '''
        Solve intial rays up until a given depth, z
        '''
        # Need to make sure all rays have left volume
        # Conservative estimate of diagonal across volume
        # Then can backproject to surface of volume
        length = self.extent + z
        t  = np.linspace(0.0,length/c,2)
        s0 = s0.flatten() #odeint insists

        start = time()
        dsdt_ODE = lambda t, y: dsdt(t, y, self)
        #try converting to this to jax based diffrax (also pure jax to see if abstraction is significant to performance) and compare solution times
        sol = solve_ivp(dsdt_ODE, [0,t[-1]], s0, t_eval=t)
        finish = time()
        self.duration = finish - start

        Np = s0.size//9
        self.Beam.sf = sol.y[:,-1].reshape(9,Np)

        self.Beam.rf, self.Beam.Jf = ray_to_Jonesvector(self.Beam.sf, self.extent, probing_direction = self.Beam.probing_direction)
        del self.Beam.Jf

    def clear_memory(self):
        """
        Clears variables not needed by solve method, saving memory

        Can also use after calling solve to clear ray positions - important when running large number of rays

        """
        self.dndx = None
        self.dndy = None
        self.dndz = None
        self.ScalarDomain.ne = None
        self.ne_nc = None
        self.Beam.sf = None
        self.Beam.rf = None

# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, Propagator):
    """Returns an array with the gradients and velocity per ray for ode_int

    Args:
        t (float array): I think this is a dummy variable for ode_int - our problem is time invarient
        s (9N float array): flattened 9xN array of rays used by ode_int
        ScalarDomain (ScalarDomain): an ScalarDomain object which can calculate gradients

    Returns:
        9N float array: flattened array for ode_int
    """

    Np     = s.size//9
    s      = s.reshape(9,Np)
    sprime = np.zeros_like(s)
    # Velocity and position
    v = s[3:6,:]
    x = s[:3,:]
    # Amplitude, phase and polarisation
    a = s[6,:]
    p = s[7,:]
    r = s[8,:]

    sprime[3:6,:] = Propagator.dndr(x)
    sprime[:3,:]  = v
    sprime[6,:]   = Propagator.atten(x)*a
    sprime[7,:]   = Propagator.phase(x)
    sprime[8,:]   = Propagator.neB(x,v)

    return sprime.flatten()

# Need to backproject to ne volume, then find angles
def ray_to_Jonesvector(ode_sol, ne_extent, probing_direction):
    """Takes the output from the 9D solver and returns 6D rays for ray-transfer matrix techniques.
    Effectively finds how far the ray is from the end of the volume, returns it to the end of the volume.
    Args:
        ode_sol (6xN float): N rays in (x,y,z,vx,vy,vz) format, m and m/s and amplitude, phase and polarisation
        ne_extent (float): edge length of cube, m
        probing_direction (str): x, y or z.
    Returns:
        [type]: [description]
    """

    Np = ode_sol.shape[1] # number of photons
    ray_p = np.zeros((4,Np))
    ray_J = np.zeros((2,Np),dtype=complex)

    x, y, z, vx, vy, vz = ode_sol[0], ode_sol[1], ode_sol[2], ode_sol[3], ode_sol[4], ode_sol[5]

    # Resolve distances and angles
    # YZ plane
    if(probing_direction == 'x'):
        t_bp = (x-ne_extent)/vx
        # Positions on plane
        ray_p[0] = y-vy*t_bp
        ray_p[2] = z-vz*t_bp
        # Angles to plane
        ray_p[1] = np.arctan(vy/vx)
        ray_p[3] = np.arctan(vz/vx)
    # XZ plane
    elif(probing_direction == 'y'):
        t_bp = (y-ne_extent)/vy
        # Positions on plane
        ray_p[0] = x-vx*t_bp
        ray_p[2] = z-vz*t_bp
        # Angles to plane
        ray_p[1] = np.arctan(vx/vy)
        ray_p[3] = np.arctan(vz/vy)
    # XY plane
    elif(probing_direction == 'z'):
        t_bp = (z-ne_extent)/vz
        # Positions on plane
        ray_p[0] = x-vx*t_bp
        ray_p[2] = y-vy*t_bp
        # Angles to plane
        ray_p[1] = np.arctan(vx/vz)
        ray_p[3] = np.arctan(vy/vz)

    # Resolve Jones vectors
    amp,phase,pol = ode_sol[6], ode_sol[7], ode_sol[8]
    # Assume initially polarised along y
    E_x_init = np.zeros(Np)
    E_y_init = np.ones(Np)
    # Perform rotation for polarisation, multiplication for amplitude, and complex rotation for phase
    ray_J[0] = amp*(np.cos(phase)+1.0j*np.sin(phase))*(np.cos(pol)*E_x_init-np.sin(pol)*E_y_init)
    ray_J[1] = amp*(np.cos(phase)+1.0j*np.sin(phase))*(np.sin(pol)*E_x_init+np.cos(pol)*E_y_init)

    # ray_p [x,phi,y,theta], ray_J [E_x,E_y]

    return ray_p,ray_J