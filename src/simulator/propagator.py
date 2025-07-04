import numpy as np
import diffrax
import optax
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# defaults float data types to 64-bit instead of 32 for greater precision
jax.config.update('jax_enable_x64', True)
#jax.config.update('jax_captured_constants_report_frames', -1)
#jax.config.update('jax_captured_constants_warn_bytes', 128*1024**2)
jax.config.update('jax_traceback_filtering', 'off')

from scipy.integrate import odeint, solve_ivp
from time import time
from jax.scipy.interpolate import RegularGridInterpolator
from equinox import filter_jit
from datetime import datetime
from jax.lib import xla_bridge
from os import system as os_system
from sys import getsizeof

from scipy.constants import c
from scipy.constants import e
#from scipy.constants import hbar
#from scipy.constants import m_e

class Propagator:
    def __init__(self, ScalarDomain, s0, *, probing_direction = 'z', inv_brems = False, phaseshift = False):
        self.ScalarDomain = ScalarDomain

        self.s0 = s0

        self.probing_direction = probing_direction

        self.inv_brems = inv_brems
        self.phaseshift = phaseshift

        self.integration_length = ScalarDomain.lengths[['x', 'y', 'z'].index(self.probing_direction)]
        self.extent = self.integration_length / 2

# The following functions are methods to be called by the solve()
    def calc_dndr(self, lwl = 1064e-9):
        """
        Generate interpolators for derivatives.

        Args:
            lwl (float, optional): laser wavelength. Defaults to 1064e-9 m.
        """

        self.omega = 2 * jnp.pi * c / lwl
        nc = 3.14207787e-4 * self.omega ** 2

        # Find Faraday rotation constant http://farside.ph.utexas.edu/teaching/em/lectures/node101.html
        if (self.ScalarDomain.B_on):
            self.VerdetConst = 2.62e-13 * lwl ** 2 # radians per Tesla per m^2

        self.ne_nc = jnp.array(self.ScalarDomain.ne / nc, dtype = jnp.float32) #normalise to critical density
        
        #More compact notation is possible here, but we are explicit
        # can we find a way to reduce ram allocation
        self.dndx = -0.5 * c ** 2 * jnp.gradient(self.ne_nc, self.ScalarDomain.x, axis=0)
        self.dndy = -0.5 * c ** 2 * jnp.gradient(self.ne_nc, self.ScalarDomain.y, axis=1)
        self.dndz = -0.5 * c ** 2 * jnp.gradient(self.ne_nc, self.ScalarDomain.z, axis=2)

        self.dndx_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.dndx, bounds_error = False, fill_value = 0.0)
        self.dndy_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.dndy, bounds_error = False, fill_value = 0.0)
        self.dndz_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.dndz, bounds_error = False, fill_value = 0.0)

    def omega_pe(ne):
        """Calculate electron plasma freq. Output units are rad/sec. From nrl pp 28"""

        return 5.64e4 * jnp.sqrt(ne)

    # NRL formulary inverse brems - cheers Jack Halliday for coding in Python
    # Converted to rate coefficient by multiplying by group velocity in plasma
    def kappa(self):
        # Useful subroutines
        def v_the(Te):
            """Calculate electron thermal speed. Provide Te in eV. Retrurns result in m/s"""

            return 4.19e5 * jnp.sqrt(Te)

        def V(ne, Te, Z, omega):
            o_pe = self.omega_pe(ne)
            o_max = jnp.copy(o_pe)
            o_max[o_pe < omega] = omega
            L_classical = Z * e / Te
            L_quantum = 2.760428269727312e-10 / jnp.sqrt(Te) # hbar / jnp.sqrt(m_e * e * Te)
            L_max = jnp.maximum(L_classical, L_quantum)

            return o_max * L_max

        def coloumbLog(ne, Te, Z, omega):
            return jnp.maximum(2.0, jnp.log(v_the(Te) / V(ne, Te, Z, omega)))

        ne_cc = self.ScalarDomain.ne * 1e-6
        o_pe = self.omega_pe(ne_cc)
        CL = coloumbLog(ne_cc, self.ScalarDomain.Te, self.ScalarDomain.Z, self.omega)

        return 3.1e-5 * self.ScalarDomain.Z * c * jnp.power(ne_cc / self.omega, 2) * CL * jnp.power(self.ScalarDomain.Te, -1.5) # 1/s

    # Plasma refractive index
    def n_refrac(self):
        ne_cc = self.ScalarDomain.ne * 1e-6
        o_pe = self.omega_pe(ne_cc)

        return jnp.sqrt(1.0-(o_pe/self.omega)**2)

    def set_up_interps(self):
        # Electron density
        self.ne_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.ne, bounds_error = False, fill_value = 0.0)

        # Magnetic field
        if(self.ScalarDomain.B_on):
            self.Bx_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.B[:,:,:,0], bounds_error = False, fill_value = 0.0)
            self.By_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.B[:,:,:,1], bounds_error = False, fill_value = 0.0)
            self.Bz_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.ScalarDomain.B[:,:,:,2], bounds_error = False, fill_value = 0.0)

        # Inverse Bremsstrahlung
        if(self.inv_brems):
            self.kappa_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.kappa(), bounds_error = False, fill_value = 0.0)

        # Phase shift
        if(self.phaseshift):
            self.refractive_index_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.n_refrac(), bounds_error = False, fill_value = 1.0)

    def dndr(self, r):
        """
        Returns the gradient at the locations r

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
    def atten(self, x):
        if(self.inv_brems):
            return self.kappa_interp(x.T)
        else:
            return 0.0

    # Phase shift introduced by refractive index
    def phase(self, x):
        if(self.phaseshift):
            #self.refractive_index_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.n_refrac(), bounds_error = False, fill_value = 1.0)
            return self.omega * (self.refractive_index_interp(x.T) - 1.0)
        else:
            return 0.0
    
    def get_ne(self, x):
        return self.ne_interp(x.T)

    def get_B(self, x):
        return jnp.array([self.Bx_interp(x.T),self.By_interp(x.T),self.Bz_interp(x.T)])

    def neB(self, x, v):
        """
        Returns the VerdetConst ne B.v

        Args:
            x (3xN float): N [x,y,z] locations
            v (3xN float): N [vx,vy,vz] velocities

        Returns:
            N float: N values of ne B.v
        """

        if(self.ScalarDomain.B_on):
            ne_N = self.get_ne(x)
            Bv_N = jnp.sum(self.get_B(x) * v, axis = 0)
            pol = self.VerdetConst * ne_N * Bv_N
        else:
            pol = 0.0

        return pol

    def solve(self, return_E = False, parallelise = True, jitted = True):
        # Need to make sure all rays have left volume
        # Conservative estimate of diagonal across volume
        # Then can backproject to surface of volume

        s0 = self.s0
        print("Size in memory of initial rays:", getsizeof(s0))

        # 8.0^0.5 is an arbritrary factor to ensure rays have enough time to escape the box
        # think we should change this???
        t = jnp.linspace(0.0, np.sqrt(8.0) * self.extent / c, 2)

        start = time()

        if not parallelise:
            s0 = s0.flatten() #odeint insists
            #s0 = jnp.array(s0) #for diffrax

            # wrapper allows dummy variables t & y to be used by solve_ivp(), self is required by dsdt
            dsdt_ODE = lambda t, y: dsdt(t, y, self, parallelise)
            sol = solve_ivp(dsdt_ODE, [0, t[-1]], s0, t_eval = t)
        else:
            available_devices = jax.devices()
            print(f"\nAvailable devices: {available_devices}")

            norm_factor = np.max(t)

            # wrapper for same reason, diffrax.ODETerm instantiaties this and passes args (this will contain self)
            def dsdt_ODE(t, y, args):
                return dsdt(t, y, args[0], args[1]) * norm_factor

            def diffrax_solve(dydt, t0, t1, Nt, rtol = 1e-7, atol = 1e-9):
                """
                Here we wrap the diffrax diffeqsolve function such that we can easily parallelise it
                """

                # We convert our python function to a diffrax ODETerm
                term = diffrax.ODETerm(dsdt_ODE)
                # We chose a solver (time-stepping) method from within diffrax library
                solver = diffrax.Tsit5() # (RK45 - closest I could find to solve_ivp's default method)

                # At what time points you want to save the solution
                saveat = diffrax.SaveAt(ts = jnp.linspace(t0, t1, Nt))
                # Diffrax uses adaptive time stepping to gain accuracy within certain tolerances
                # had to reduce relative tolerance to 1 to get it to run, need to compare to see the consequences of this
                stepsize_controller = diffrax.PIDController(rtol = rtol, atol = atol)

                return lambda s0, args : diffrax.diffeqsolve(
                    term,
                    solver,
                    y0 = jnp.array(s0),
                    args = args,
                    t0 = t0,
                    t1 = t1,
                    dt0 = (t1 - t0) * norm_factor**2 / Nt,
                    saveat = saveat,
                    stepsize_controller = stepsize_controller,
                    # set max steps to no. of cells x100
                    max_steps = self.ScalarDomain.x_n * self.ScalarDomain.y_n * self.ScalarDomain.z_n * 100 #10000 - default for solve_ivp?????
                )

            # hardcode to normalise to 1 due to diffrax bug - how do we un-normalise after solving?, check aidrian's issue
            ODE_solve = diffrax_solve(dsdt_ODE, t[0], t[-1] / norm_factor, len(t))

            if jitted:
                start_comp = time()

                # equinox.filter_jit() (imported as filter_jit()) provides debugging info unlike jax.jit() - it does not like static args though so sticking with jit for now
                #ODE_solve = jax.jit(ODE_solve, static_argnums = 1, device = available_devices[0])
                ODE_solve = filter_jit(ODE_solve, device = available_devices[0])
                # not sure about the performance of non-static specified arguments with filter_jit() - only use for debugging not in 'production'

                finish_comp = time()
                print("\njax compilation of solver took:", finish_comp - start_comp)

            running_device = xla_bridge.get_backend().platform
            print("Running device:", running_device, end='')

            if running_device == 'cpu':
                from multiprocessing import cpu_count
                core_count = cpu_count()
                print(", with:", core_count, "cores.")

                """
                from jax.sharding import PartitionSpec as P, NamedSharding

                # Create a Sharding object to distribute a value across devices:
                mesh = jax.make_mesh((4, 2), ('x', 'y'))

                # Create an array of random values:
                x = jax.random.normal(jax.random.key(0), (8192, 8192))
                # and use jax.device_put to distribute it across devices:
                y = jax.device_put(x, NamedSharding(mesh, P('x', 'y')))
                jax.debug.visualize_array_sharding(y)
                """
            elif running_device == 'gpu':
                print("\n")
            elif running_device == 'tpu':
                print("\n")
            else:
                print("No suitable device detected!")

            # Solve for specific s0 intial values
            #args = {'self': self, 'parallelise': parallelise}
            #args = [self, parallelise]
            args = (self, parallelise) # passed args must be hashable to be made static for jax.jit, tuple is hashable, array & dict are not

            # pass s0[:, i] for each ray via a jax.vmap for parallelisation
            # transposed as jax.vmap() expects form of [batch_idx, items] not [items, batch_idx]
            # remove unnecessary static arguments to increase speed and reduce likelihood of unexpected behaviours
            sol = jax.vmap(lambda s: ODE_solve(s, args))(s0.T)

            memory_debug = True
            if memory_debug:
                jax.debug.visualize_array_sharding(sol.ys[:, -1, :])

                getsizeof("Size in memory of initial rays:", getsizeof(s0))
                getsizeof("Size in memory of solution:", getsizeof(sol))
                getsizeof("Size in memory of propagator class:", getsizeof(sol))

                folder = "abc"
                os.chdir(".")
                print("current dir is: %s" % ())

                folder_name = "memory_benchmarks/"
                rel_path_to_folder = "../../evaluation/"
                if os.path.isdir(getcwd() + folder):
                    path = rel_path_to_folder
                else:
                    os.mkdir(folder_name)
                    path = folder_name

                path += "memory-domain" + str(self.ScalarDomain.dim[0]) + "_rays"+ str(s0.shape[1]) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".prof"
                jax.profiler.save_device_memory_profile(path)

                #os_system(f"~/go/bin/pprof -top {sys.executable} memory_{N}.prof")
                os_system(f"~/go/bin/pprof -top /bin/ls " + path)
                #os_system(f"~/go/bin/pprof --web " + path)

        finish = time()
        self.duration = finish - start

        Np = s0.size // 9
        if not parallelise:
            self.rf = sol.y[:,-1].reshape(9, Np)
        else:
            """
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
            """

            #if sol.result == RESULTS.successful:
            #self.rf = sol.ys[:, -1, :].reshape(9, Np)# / scalar
            self.rf = sol.ys[:, -1, :].T

            #self.rf = self.rf.at[:, :].set(self.rf[:, :] * (1e-3) / c)

            print("\nParallelised output has resulting 3D matrix of form: [batch_count, 2, 9]:", sol.ys.shape)
            print("\t2 to account for the start and end results")
            print("\t9 containing the 3 position and velocity components, amplitude, phase and polarisation")
            print("\nWe slice the end result and transpose into the form:", sol.ys[:, -1, :].reshape(9, Np).shape, "to work with later code.")
            #else:
            #    print("Ray tracer failed. This could be a case of diffrax exceeding max steps again due to apparent 'strictness' compared to solve_ivp, check error log.")

        """
        x = self.rf[0, :]
        y = self.rf[2, :]

        print("\nx-y size expected: (", len(x), ", ", len(y), ")", sep='')
        print('Final rays:', self.rf[:, :])

        # means that jnp.isnan(a) returns True when a is not Nan
        # ensures that x & y are the same length, if output of either is Nan then will not try to render ray in histogram
        mask = ~jnp.isnan(x) & ~jnp.isnan(y)

        x = x[mask]
        y = y[mask]

        print("x-y after clearing nan's: (", len(x), ", ", len(y), ")", sep='')
        """

        self.rf, self.Jf = ray_to_Jonesvector(self.rf, self.extent, probing_direction = self.probing_direction, return_E = return_E)

        """
        print("\nJonesvector's output as 2 array's of form's:", self.rf.shape, end = ',')
        if self.Jf is not None:
            print(self.Jf.shape, end = ' - ')

        if return_E:
            print("self.Jf returned as return_E = True")
            #return self.rf, self.Jf
        else:
            print("2nd array is of this shape as return_E = False")
            #return self.rf
        """

    def solve_at_depth(self, z):
        """
        Solve intial rays up until a given depth, z
        """

        # Need to make sure all rays have left volume
        # Conservative estimate of diagonal across volume
        # Then can backproject to surface of volume

        length = self.extent + z
        t = jnp.linspace(0.0, length / c, 2)

        s0 = self.s0
        s0 = s0.flatten() #odeint insists

        print("\nStarting ray trace.")

        start = time()

        dsdt_ODE = lambda t, y: dsdt(t, y, self, parallelise = False)
        sol = solve_ivp(dsdt_ODE, [0,t[-1]], s0, t_eval=t)

        finish = time()
        self.duration = finish - start

        print("\nRay trace completed in:\t", self.duration, "s")

        self.rf, _ = ray_to_Jonesvector(sol.y[:,-1].reshape(9, s0.size // 9), self.extent, probing_direction = self.probing_direction)

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
        self.s0 = None
        self.rf = None
        self.Jf = None

# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, Propagator, parallelise):
    """
    Returns an array with the gradients and velocity per ray for ode_int

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
    
    # unsure as jax array - also passed not created, should be a copy anyway no?
    #del s

    #sprime = jnp.zeros_like(s.reshape(9, s.size // 9))
    sprime = jnp.zeros_like(s)

    # Position and velocity
    # needs to be before the reshape to avoid indexing errors
    x = s[:3, :]
    v = s[3:6, :]

    # Amplitude, phase and polarisation
    a = s[6, :]
    #p = s[7,:]
    #r = s[8,:]

    sprime = sprime.at[3:6, :].set(Propagator.dndr(x))
    sprime = sprime.at[:3, :].set(v)

    sprime = sprime.at[6, :].set(Propagator.atten(x) * a)
    sprime = sprime.at[7, :].set(Propagator.phase(x))
    sprime = sprime.at[8, :].set(Propagator.neB(x, v))

    del x
    del v
    del a

    return sprime.flatten()

# Need to backproject to ne volume, then find angles
def ray_to_Jonesvector(rays, ne_extent, probing_direction, *, keep_current_plane = False, return_E = False):
    # * forces keep_current_plane and return_E to be keyword-only arguments
    # meaning .. return_E = True (missing out keep_current_plane) will work as it will not rely on position
    """
    Takes the output from the 9D solver and returns 6D rays for ray-transfer matrix techniques.
    Effectively finds how far the ray is from the end of the volume, returns it to the end of the volume.

    Gives position (and angles) in other axes at point where ray is in end plane of its extent in the probing axis
    (if keep_current_plane is set to True, it does not return the rays to the end of volume - just returns current 2D slice position)

    Args:
        rays (6xN float): N rays in (x,y,z,vx,vy,vz) format, m and m/s and amplitude, phase and polarisation
        ne_extent (float): edge length of shape (cuboid) in probing direction, m
        probing_direction (str): x, y or z.
        keep_current_plane (boolean): flag to enable compatability (via True) with use in diagnostics.py, defaults to False

    Returns:
        [type]: [description]
    """

    Np = rays.shape[1] # number of photons

    ray_p = np.zeros((4, Np))
    ray_J = np.zeros((2, Np), dtype = complex)

    x, y, z, vx, vy, vz = rays[0], rays[1], rays[2], rays[3], rays[4], rays[5]

    # Resolve distances and angles
    # YZ plane
    if(probing_direction == 'x'):
        t_bp = (x - ne_extent) / vx

        # Positions on plane
        if not keep_current_plane:
            ray_p[0] = y - vy * t_bp
            ray_p[2] = z - vz * t_bp
        else:
            ray_p[0] = y
            ray_p[2] = z

        # Angles to plane
        ray_p[1] = np.arctan(vy / vx)
        ray_p[3] = np.arctan(vz / vx)
    # XZ plane
    elif(probing_direction == 'y'):
        t_bp = (y - ne_extent) / vy

        #
        # I have switched x & z for the sake of consistent ordering of the axes
        # Standardised in keeping with positive 'forward' notation, etc. x * y = z but don't do y * x = -z
        # If memory is not a concern then will instead create a class to cover directions
        # This would entail both the array and a self.dir parameter of type char - containing 'x', 'y' or 'z'
        #

        # Positions on plane
        if not keep_current_plane:
            ray_p[0] = z - vz * t_bp
            ray_p[2] = x - vx * t_bp
        else:
            ray_p[0] = z
            ray_p[2] = x

        # Angles to plane
        ray_p[1] = np.arctan(vz / vy)
        ray_p[3] = np.arctan(vx / vy)
    # XY plane
    elif(probing_direction == 'z'):
        t_bp = (z - ne_extent) / vz

        # Positions on plane
        if not keep_current_plane:
            ray_p[0] = x - vx * t_bp
            ray_p[2] = y - vy * t_bp
        else:
            ray_p[0] = x
            ray_p[2] = y

        # Angles to plane
        ray_p[1] = np.arctan(vx / vz)
        ray_p[3] = np.arctan(vy / vz)
    else:
        print("\nIncorrect probing direction. Use: x, y or z.")
    
    del x
    del y
    del z
    del vx
    del vy
    del vz

    if return_E:
        # Resolve Jones vectors
        amp, phase, pol = rays[6], rays[7], rays[8]

        # Assume initially polarised along y
        E_x_init = np.zeros(Np)
        E_y_init = np.ones(Np)

        # Perform rotation for polarisation, multiplication for amplitude, and complex rotation for phase
        ray_J[0] = amp * (np.cos(phase) + 1.0j * np.sin(phase)) * (np.cos(pol) * E_x_init - np.sin(pol) * E_y_init)
        ray_J[1] = amp * (np.cos(phase) + 1.0j * np.sin(phase)) * (np.sin(pol) * E_x_init + np.cos(pol) * E_y_init)

        del amp
        del phase
        del pol

        del E_x_init
        del E_y_init

    del Np

    # ray_p [x, phi, y, theta], ray_J [E_x, E_y]
    if return_E:
        return jnp.array(ray_p), jnp.array(ray_J)

    return jnp.array(ray_p), None