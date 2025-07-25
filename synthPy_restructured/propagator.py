import numpy as np
import diffrax
import optax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import sys
import os

sys.path.append('../../utils')

from scipy.integrate import odeint, solve_ivp
from time import time
from jax.scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RegularGridInterpolator as RGI
from equinox import filter_jit
from SpK_reader import open_emi_files
from scipy.constants import c, e



def omega_pe(ne):
    '''Calculate electron plasma freq. Output units are rad/sec. From nrl pp 28'''

    return 56.35*np.sqrt(ne)

class Propagator:
    def __init__(self, ScalarDomain, Beam, inv_brems = False, x_ray = False, phaseshift = False, refrac_field = False, elec_density = True):
        self.ScalarDomain = ScalarDomain
        self.Beam = Beam
        self.inv_brems = inv_brems
        self.x_ray = x_ray
        self.phaseshift = phaseshift
        self.prev_x = None
        self.phase_integral = 0
        self.refrac_field = refrac_field
        self.elec_density = elec_density

        #Opacity takes into account inverse bremstrahlung. Therefore, if both x_ray and inv_brems are True,
        # then x_ray should remain True and inv_brems should be set to False. 
        if self.x_ray:
            self.inv_brems = False
        

        # finish initialising the beam position using the scalardomain edge position

        #axes = ['x', 'y', 'z']
        #print(np.asarray(axes == Beam.probing_direction).nonzero())
        #print(np.where(axes == assert isinstance(Beam.probing_direction, str)))
        #index = np.where(axes == Beam.probing_direction)[0]

        index = ['x', 'y', 'z'].index(Beam.probing_direction)
        self.integration_length = ScalarDomain.lengths[index]
        self.extent = self.integration_length / 2
        self.energy= 6.63e-34*c/(self.Beam.wavelength*1.6e-19)

        #Beam.init_beam(ne_extent)       # this is the second call instance for init_beam() stefano was referring too

# The following functions are methods to be called by the solve()
    def calc_dndr(self):
        """Generate interpolators for derivatives.

        Args:
            lwl (float, optional): laser wavelength. Defaults to 1053e-9 m.
        """

        lwl = self.Beam.wavelength

        self.omega = 2*np.pi*(c/lwl)
        nc = 3.14207787e-4*self.omega**2
        self.nc = nc

        # Find Faraday rotation constant http://farside.ph.utexas.edu/teaching/em/lectures/node101.html
        if (self.ScalarDomain.B_on):
            self.VerdetConst = 2.62e-13*lwl**2 # radians per Tesla per m^2

        
        if self.refrac_field is not True:
            self.ne_nc = np.array(self.ScalarDomain.ne / nc, dtype = np.float32) #normalise to critical density
            gradient_term = -0.5 * c**2 * self.ne_nc
        else:
            gradient_term = 0.5 * c**2 * self.ScalarDomain.refrac_field**2
            

        #More compact notation is possible here, but we are explicit
        # can we find a way to reduce ram allocation
        self.dndx = np.gradient(gradient_term, self.ScalarDomain.x, axis=0)
        self.dndy = np.gradient(gradient_term, self.ScalarDomain.y, axis=1)
        self.dndz = np.gradient(gradient_term, self.ScalarDomain.z, axis=2)
        
        self.dndx_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.dndx, bounds_error = False, fill_value = 0.0)
        self.dndy_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.dndy, bounds_error = False, fill_value = 0.0)
        self.dndz_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.dndz, bounds_error = False, fill_value = 0.0)

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
            L_classical = Z*e/Te
            L_quantum = 2.760428269727312e-10/np.sqrt(Te) # sc.hbar/np.sqrt(sc.m_e*sc.e*Te)
            L_max = np.maximum(L_classical, L_quantum)

            return o_max*L_max

        def coloumbLog(ne, Te, Z, omega):
            return np.maximum(2.0,np.log(v_the(Te)/V(ne, Te, Z, omega)))

        ne_cc = self.ScalarDomain.ne*1e-6
        o_pe  = omega_pe(ne_cc)
        CL    = coloumbLog(ne_cc, self.ScalarDomain.Te, self.ScalarDomain.z, self.omega)

        return 3.1e-5*self.ScalarDomain.z*c*np.power(ne_cc/self.omega,2)*CL*np.power(self.ScalarDomain.Te, -1.5) # 1/s

    # Plasma refractive index
    def n_refrac(self):

        #ne_cc = self.ScalarDomain.ne*1e-6
        o_pe  = omega_pe(self.ScalarDomain.ne)
        return np.sqrt(1.0-(o_pe/self.omega)**2)

    def set_up_interps(self):
        # Electron density
        if (self.elec_density):
            self.ne_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.ne, bounds_error = False, fill_value = 0.0)
            # Phase shift
            if(self.phaseshift):
                self.refractive_index_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.n_refrac(), bounds_error = False, fill_value = 1.0)
        # Magnetic field
        if(self.ScalarDomain.B_on):
            self.Bx_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.B[:,:,:,0], bounds_error = False, fill_value = 0.0)
            self.By_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.B[:,:,:,1], bounds_error = False, fill_value = 0.0)
            self.Bz_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.B[:,:,:,2], bounds_error = False, fill_value = 0.0)

        # Inverse Bremsstrahlung
        if(self.inv_brems):
            self.kappa_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.kappa(), bounds_error = False, fill_value = 0.0)
        
        #Opacity, Temperature, and Mass Density
        if(self.x_ray):
            grp_centres, grps, rho, Te, opa_data = open_emi_files("../../opa_multi_planck_CH_LTE_210506_Hydra_ColdOpa.spk")
            opa_max=self.ScalarDomain.x_n/(self.ScalarDomain.x_length)
            opa_data_capped=np.minimum(opa_max, opa_data)
            self.opacity_interp = RegularGridInterpolator((grp_centres, rho, Te), opa_data_capped, bounds_error = False, fill_value = 0.0)
            self.Te_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.Te, bounds_error = False, fill_value = 0.0)
            self.rho_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.rho, bounds_error = False, fill_value = 0.0)
            opacity_grid = self.opacity_interp((self.energy, self.ScalarDomain.rho, self.ScalarDomain.Te))
            self.opacity_spatial_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), opacity_grid, bounds_error = False, fill_value = 0.0)
        
        
        if(self.refrac_field):
            self.refractive_index_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.refrac_field, bounds_error = False, fill_value = 1.0)
    def dndr(self, r):
        """returns the gradient at the locations r

        Args:
            r (3xN float): N [x, y, z] locations

        Returns:
            3 x N float: N [dx, dy, dz] electron density gradients
        """

        grad = jnp.zeros_like(r)

        grad = grad.at[0, :].set(self.dndx_interp(r.T))
        grad = grad.at[1, :].set(self.dndy_interp(r.T))
        grad = grad.at[2, :].set(self.dndz_interp(r.T))

        return grad
        

    # Attenuation due to inverse bremsstrahlung
    def atten(self,x):
        if(self.inv_brems):
            return self.kappa_interp(x.T)
        else:
            return 0.0
    

    def atten_x_ray(self, x):
        if(self.x_ray):
            # rho = self.rho_interp(x.T)
            # Te = self.Te_interp(x.T)
            opacity = self.opacity_spatial_interp(x.T)
            return -1*opacity*c
        else:
            return 0.0


    # Phase shift introduced by refractive index
    def phase(self,x):
        if(self.phaseshift):
            #self.refractive_index_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.n_refrac(), bounds_error = False, fill_value = 1.0)
            # return self.omega*(self.refractive_index_interp(x.T)-1.0)
            return 0.0
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

    def solve(self, return_E = False, parallelise = True, jitted = True, Nt = 2):
        # Need to make sure all rays have left volume
        # Conservative estimate of diagonal across volume
        # Then can backproject to surface of volume
        self.prev_x = None
        self.phase_integral = 0
        s0 = self.Beam.s0

        # 8.0^0.5 is an arbritrary factor to ensure rays have enough time to escape the box
        t = np.linspace(0.0, np.sqrt(8.0) * self.extent / c, Nt)
        self.t = t

        start = time()

        if not parallelise:
            s0 = s0.flatten() #odeint insists
            #s0 = jnp.array(s0) #for diffrax

            # wrapper allows dummy variables t & y to be used by solve_ivp(), self is required by dsdt
            dsdt_ODE = lambda t, y: dsdt(t, y, self, parallelise)
            sol = solve_ivp(dsdt_ODE, [0, t[-1]], s0, t_eval = t)
            print(sol)
            
            
        else:
            norm_factor = t[-1]
            # wrapper for same reason, diffrax.ODETerm instantiaties this and passes args (this will contain self)
            def dsdt_ODE(t, y, args):
                return dsdt(t, y, args[0], args[1]) * norm_factor

            def diffrax_solve(dydt, t0, t1, Nt, rtol=1e-4, atol=1e-5):
                """
                Here we wrap the diffrax diffeqsolve function such that we can easily parallelise it
                """

                # We convert our python function to a diffrax ODETerm
                term = diffrax.ODETerm(dydt)
                # We chose a solver (time-stepping) method from within diffrax library
                solver = diffrax.Tsit5() # (RK45 - closest I could find to solve_ivp's default method)

                # At what time points you want to save the solution
                saveat = diffrax.SaveAt(ts = jnp.linspace(t0, t1, Nt))
                # Diffrax uses adaptive time stepping to gain accuracy within certain tolerances
                # had to reduce relative tolerance to 1 to get it to run, need to compare to see the consequences of this
                dtmax = 0.5*self.ScalarDomain.x_length/self.ScalarDomain.x_n/(c*norm_factor)
                stepsize_controller = diffrax.PIDController(rtol = rtol, atol = atol, dtmax = dtmax)

                return lambda s0, args : diffrax.diffeqsolve(
                    term,
                    solver,
                    y0 = jnp.array(s0),
                    args = args,
                    t0 = t0,
                    t1 = t1,
                    #dt0 = (t1 - t0) * norm_factor**2 / Nt,
                    dt0 = None,
                    saveat = saveat,
                    stepsize_controller = stepsize_controller,
                    max_steps = (self.ScalarDomain.x_n ** 3) * 100
                )
            # def ODE_solve_2(s, mask):
                
            #     return jax.lax.cond(mask, 
            #                         lambda s : diffrax_solve(dsdt_ODE, 0, 1, Nt)(s, args1),
            #                         lambda s : diffrax_solve(dsdt_ODE, 0, 1, 2)(s, args1), s)
            
            # key = jax.random.PRNGKey(100)
            
            # # randomly choose k unique indices from N
            # chosen_indices = jax.random.choice(key, self.Beam.Np, shape=(Ntrack,), replace=False)

            # # create a boolean mask: True for f1, False for f2
            # mask = jnp.zeros(self.Beam.Np, dtype=bool).at[chosen_indices].set(True)

            ODE_solve = diffrax_solve(dsdt_ODE, 0, 1, Nt)

            if jitted:
                start_comp = time()

                # equinox.filter_jit() (imported as filter_jit()) provides debugging info unlike jax.jit() - it does not like static args though so sticking with jit for now
                #ODE_solve = jax.jit(ODE_solve, static_argnums = 1)
                ODE_solve = filter_jit(ODE_solve)

                finish_comp = time()
                print("jax compilation of solver took:", finish_comp - start_comp)

            # Solve for specific s0 intial values
            #args = {'self': self, 'parallelise': parallelise}
            #args = [self, parallelise]
            args1 = (self, parallelise) # passed args must be hashable to be made static for jax.jit, tuple is hashable, array & dict are not

            # pass s0[:, i] for each ray via a jax.vmap for parallelisation
            # transposed as jax.vmap() expects form of [batch_idx, items] not [items, batch_idx]
            sol = jax.vmap(lambda s: ODE_solve(s, args1))(s0.T)
            #sol = jax.vmap(ODE_solve_2)(s0.T, mask)
            #print(sol.ys)
            
        print("phase shift from line integral:", self.phase_integral)
        

        finish = time()
        self.duration = finish - start

        Np = s0.size // 9
        if not parallelise:
            self.Beam.rf = sol.y[:,-1].reshape(9, Np)
            self.Beam.positions = np.transpose(sol.y.reshape(9, Np, Nt), (1, 2, 0))[:, : ,:3]
            self.Beam.amplitudes = np.transpose(sol.y.reshape(9, Np, Nt), (1, 2, 0))[:, :, 6]
            self.Beam.phases = np.transpose(sol.y.reshape(9, Np, Nt), (1, 2, 0))[:, :, 7]
        else:
            '''
            #for i in enumerate(sol.result):
            #    print(i)
            for idx, result in enumerate(sol.result):
                # Check if each result is successful
                if result.success:
                    print(f"Solution at index {idx} succeeded.")
                else:
                    print(f"Solution at index {idx} failed.")
        
            #print(next(sol.result))
            #print(next(sol.result))
            #print(type(sol.result[0]))  # Check the type of results
            '''

            #if sol.result == RESULTS.successful:
            self.Beam.rf = sol.ys[:, -1, :].T
            self.Beam.positions = sol.ys[:, :, :3]
            self.Beam.amplitudes = sol.ys[:, :, 6]
            self.Beam.phases = sol.ys[:, :, 7]
            

            print("\nParallelised output has resulting 3D matrix of form: [batch_count, 2, 9]:", sol.ys.shape)
            print("\t2 to account the start and end results")
            print("\t9 containing the 3 position and velocity components, amplitude, phase and polarisation")
            print("\nWe reshape into the form:", sol.ys[:, -1, :].reshape(9, Np).shape)
            #else:
            #    print("Ray tracer failed. This could be a case of diffrax exceeding max steps again due to apparent 'strictness' compared to solve_ivp, check error log.")

        self.Beam.rf, self.Beam.Jf = ray_to_Jonesvector(self.Beam.rf, self.extent, probing_direction = self.Beam.probing_direction)
        #print("\n", self.Beam.rf)
        
        if return_E:
            return self.Beam.rf, self.Beam.Jf
        else:
            return self.Beam.rf

    def solve_at_depth(self, z):
        '''
        Solve intial rays up until a given depth, z
        '''

        # Need to make sure all rays have left volume
        # Conservative estimate of diagonal across volume
        # Then can backproject to surface of volume

        length = self.extent + z
        t = np.linspace(0.0, length / c, 2)

        s0 = self.Beam.s0
        s0 = s0.flatten() #odeint insists

        print("\nStarting ray trace.")

        start = time()

        dsdt_ODE = lambda t, y: dsdt(t, y, self)
        sol = solve_ivp(dsdt_ODE, [0,t[-1]], s0, t_eval=t)

        finish = time()
        self.duration = finish - start

        print("\nRay trace completed in:\t", self.duration, "s")

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
        prev_x = None
        phase_integral = 0

def distance(x2, x1):
    return jnp.sqrt(jnp.sum((x2-x1)*(x2-x1), axis=0))


# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, Propagator, parallelise):
    """Returns an array with the gradients and velocity per ray for ode_int

    Args:
        t (float array): I think this is a dummy variable for ode_int - our problem is time invarient
        s (9N float array): flattened 9xN array of rays used by ode_int
        ScalarDomain (ScalarDomain): an ScalarDomain object which can calculate gradients

    Returns:
        9N float array: flattened array for ode_int
    """

    if not parallelise:
        # jnp.reshape() auto converts to a jax array rather than having to do after a numpy reshape
        s = jnp.reshape(s, (9, s.size // 9))
    else:
        # forces s to be a matrix even if has the indexes of a 1d array such that dsdt() can be generalised
        s = jnp.reshape(s, (9, 1))  # one ray per vmap iteration if parallelised

    #sprime = np.zeros_like(s.reshape(9, s.size // 9))
    sprime = jnp.zeros_like(s)
    
    # Position and velocity
    # needs to be before the reshape to avoid indexing errors
    x = s[:3, :]
    v = s[3:6, :]

    if Propagator.phaseshift is True:
        if Propagator.prev_x is not None:
            dr = distance(x, Propagator.prev_x)
            Propagator.phase_integral -= Propagator.ne_interp(x.T)*dr/Propagator.nc*np.pi/Propagator.Beam.wavelength

        Propagator.prev_x = x

    # Amplitude, phase and polarisation
    a = s[6, :]
    p = s[7,:]
    #r = s[8,:]
    
    sprime = sprime.at[3:6, :].set(Propagator.dndr(x))
    sprime = sprime.at[:3, :].set(v)
    #speed = jnp.sqrt(jnp.sum(v*v, axis=0))
    #print (a)
    #print(-Propagator.atten_x_ray(x)/c)
    #print(p)
    sprime = sprime.at[6, :].set((Propagator.atten(x) + Propagator.atten_x_ray(x))*a) #add inverse bremsstrahlung and opacity
    sprime = sprime.at[7, :].set(Propagator.phase(x))
    sprime = sprime.at[8, :].set(Propagator.neB(x, v))

    return sprime.flatten()

# Need to backproject to ne volume, then find angles
def ray_to_Jonesvector(ode_sol, ne_extent, probing_direction):
    """
    Takes the output from the 9D solver and returns 6D rays for ray-transfer matrix techniques.
    Effectively finds how far the ray is from the end of the volume, returns it to the end of the volume.

    Gives position (and angles) in other axes at point where ray is in end plane of its extent in the probing axis

    Args:
        ode_sol (6xN float): N rays in (x,y,z,vx,vy,vz) format, m and m/s and amplitude, phase and polarisation
        ne_extent (float): edge length of shape (cuboid) in probing direction, m
        probing_direction (str): x, y or z.

    Returns:
        [type]: [description]
    """

    Np = ode_sol.shape[1] # number of photons

    ray_p = np.zeros((4, Np))
    ray_J = np.zeros((2, Np), dtype=complex)

    x, y, z, vx, vy, vz = ode_sol[0], ode_sol[1], ode_sol[2], ode_sol[3], ode_sol[4], ode_sol[5]

    # Resolve distances and angles
    # YZ plane
    if(probing_direction == 'x'):
        t_bp = (x - ne_extent) / vx

        # Positions on plane
        ray_p[0] = y - vy * t_bp
        ray_p[2] = z - vz * t_bp

        # Angles to plane
        ray_p[1] = np.arctan(vy / vx)
        ray_p[3] = np.arctan(vz / vx)
    # XZ plane
    elif(probing_direction == 'y'):
        t_bp = (y - ne_extent) / vy
 
        # Positions on plane
        ray_p[0] = x - vx * t_bp
        ray_p[2] = z - vz * t_bp

        # Angles to plane
        ray_p[1] = np.arctan(vx / vy)
        ray_p[3] = np.arctan(vz / vy)
    # XY plane
    elif(probing_direction == 'z'):
        t_bp = (z - ne_extent) / vz

        # Positions on plane
        ray_p[0] = x - vx * t_bp
        ray_p[2] = y - vy * t_bp

        # Angles to plane
        ray_p[1] = np.arctan(vx / vz)
        ray_p[3] = np.arctan(vy / vz)
    else:
        print("\nIncorrect probing direction. Use: x, y or z.")

    # Resolve Jones vectors
    amp,phase,pol = ode_sol[6], ode_sol[7], ode_sol[8]

    # Assume initially polarised along y
    E_x_init = np.zeros(Np)
    E_y_init = np.ones(Np)

    # Perform rotation for polarisation, multiplication for amplitude, and complex rotation for phase
    ray_J[0] = amp*(np.cos(phase)+1.0j*np.sin(phase))*(np.cos(pol)*E_x_init-np.sin(pol)*E_y_init)
    ray_J[1] = amp*(np.cos(phase)+1.0j*np.sin(phase))*(np.sin(pol)*E_x_init+np.cos(pol)*E_y_init)

    # ray_p [x,phi,y,theta], ray_J [E_x,E_y]

    return ray_p, ray_J