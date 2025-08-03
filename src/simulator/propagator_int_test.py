import numpy as np
import diffrax
import optax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

#from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint, solve_ivp
from time import time
from jax.scipy.interpolate import RegularGridInterpolator
from equinox import filter_jit

from scipy.constants import c

class Propagator:
    def __init__(self, ScalarDomain, Beam, inv_brems = False, phaseshift = False):
        self.ScalarDomain = ScalarDomain
        self.Beam = Beam
        self.inv_brems = inv_brems
        self.phaseshift = phaseshift

        index = ['x', 'y', 'z'].index(Beam.probing_direction)
        self.integration_length = ScalarDomain.lengths[index]
        self.extent = self.integration_length / 2

    def trilinearInterpolator(self, x, y, z, lengths, dims, values, query_points, *, fill_value = jnp.nan):
        idr = jnp.clip(
            jnp.floor(
                ((query_points / jnp.asarray(lengths)) + 0.5) * (jnp.asarray(dims) - 1)
            ).astype(jnp.int32),
            0, jnp.asarray(dims) - 2
        )

        wx = (query_points[:, 0] - x[idr[:, 0]]) / (x[idr[:, 0] + 1] - x[idr[:, 0]])
        wy = (query_points[:, 1] - y[idr[:, 1]]) / (y[idr[:, 1] + 1] - y[idr[:, 1]])
        wz = (query_points[:, 2] - z[idr[:, 2]]) / (z[idr[:, 2] + 1] - z[idr[:, 2]])

        return (
            values[idr[:, 0], idr[:, 1], idr[:, 2]] * (1 - wx) * (1 - wy) * (1 - wz) +
            values[idr[:, 0], idr[:, 1], idr[:, 2] + 1] * (1 - wx) * (1 - wy) * wz       +
            values[idr[:, 0], idr[:, 1] + 1, idr[:, 2]] * (1 - wx) * wy       * (1 - wz) +
            values[idr[:, 0], idr[:, 1] + 1, idr[:, 2] + 1] * (1 - wx) * wy       * wz       +
            values[idr[:, 0] + 1, idr[:, 1], idr[:, 2]] * wx       * (1 - wy) * (1 - wz) +
            values[idr[:, 0] + 1, idr[:, 1], idr[:, 2] + 1] * wx       * (1 - wy) * wz       +
            values[idr[:, 0] + 1, idr[:, 1] + 1, idr[:, 2]] * wx       * wy       * (1 - wz) +
            values[idr[:, 0] + 1, idr[:, 1] + 1, idr[:, 2] + 1] * wx       * wy       * wz
        )

    def calc_dndr(self):
        lwl = self.Beam.wavelength

        self.omega = 2*np.pi*(c/lwl)
        nc = 3.14207787e-4*self.omega**2

        self.ne_nc = np.array(self.ScalarDomain.ne / nc, dtype = np.float32) #normalise to critical density

        #More compact notation is possible here, but we are explicit
        # can we find a way to reduce ram allocation
        self.dndx = -0.5 *c ** 2 * np.gradient(self.ne_nc, self.ScalarDomain.x, axis=0)
        self.dndy = -0.5 *c ** 2 * np.gradient(self.ne_nc, self.ScalarDomain.y, axis=1)
        self.dndz = -0.5 *c ** 2 * np.gradient(self.ne_nc, self.ScalarDomain.z, axis=2)

        self.dndx_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.dndx, bounds_error = False, fill_value = 0.0)
        self.dndy_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.dndy, bounds_error = False, fill_value = 0.0)
        self.dndz_interp = RegularGridInterpolator((self.ScalarDomain.x, self.ScalarDomain.y, self.ScalarDomain.z), self.dndz, bounds_error = False, fill_value = 0.0)

    def dndr(self, r):
        grad = jnp.zeros_like(r)

        grad = grad.at[0, :].set(self.dndx_interp(r.T))
        grad = grad.at[1, :].set(self.dndy_interp(r.T))
        grad = grad.at[2, :].set(self.dndz_interp(r.T))

        return grad

    def dndr_test(self, r, x, y, z, dndx, dndy, dndz):
        grad = jnp.zeros_like(r)

        grad = grad.at[0, :].set(
            self.trilinearInterpolator(
                x, y, z,
                self.ScalarDomain.lengths,
                self.ScalarDomain.dims,
                dndx,
                r.T,
                fill_value = 0.0
            )
        )
        
        grad = grad.at[1, :].set(
            self.trilinearInterpolator(
                x, y, z,
                self.ScalarDomain.lengths,
                self.ScalarDomain.dims,
                dndy,
                r.T,
                fill_value = 0.0
            )
        )

        grad = grad.at[2, :].set(
            self.trilinearInterpolator(
                x, y, z,
                self.ScalarDomain.lengths,
                self.ScalarDomain.dims,
                dndz,
                r.T,
                fill_value = 0.0
            )
        )

        return grad

    def solve(self, return_E = False, parallelise = True, jitted = True):
        import os
        os.environ["EQX_ON_ERROR"] = "breakpoint"

        s0 = self.Beam.s0

        t = np.linspace(0.0, np.sqrt(8.0) * self.extent / c, 2)
        norm_factor = jnp.max(t)

        start = time()

        def dsdt_ODE(t, y, args):
            return dsdt(t, y, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]) * norm_factor

        def diffrax_solve(dydt, t0, t1, Nt, rtol = 1e-7, atol = 1e-9):
            term = diffrax.ODETerm(dsdt_ODE)
            solver = diffrax.Tsit5()
            saveat = diffrax.SaveAt(ts = jnp.linspace(t0, t1, Nt))
            stepsize_controller = diffrax.PIDController(rtol = 1, atol = 1e-5)

            return lambda s0, args : diffrax.diffeqsolve(
                term,
                solver,
                y0 = jnp.array(s0),
                args = args,
                t0 = t0,
                t1 = t1,
                dt0 = (t1 - t0) * norm_factor / Nt,
                saveat = saveat,
                stepsize_controller = stepsize_controller
            )

        ODE_solve = diffrax_solve(dsdt_ODE, t[0], t[-1] / norm_factor, len(t))

        from equinox import filter_jit

        if jitted:
            start_comp = time()

            ODE_solve = filter_jit(ODE_solve)

            finish_comp = time()
            print("jax compilation of solver took:", finish_comp - start_comp)

        x = jnp.float32(jnp.linspace(-self.ScalarDomain.x_length/2, self.ScalarDomain.x_length/2, self.ScalarDomain.x_n))
        y = jnp.float32(jnp.linspace(-self.ScalarDomain.y_length/2, self.ScalarDomain.y_length/2, self.ScalarDomain.y_n))
        z = jnp.float32(jnp.linspace(-self.ScalarDomain.z_length/2, self.ScalarDomain.z_length/2, self.ScalarDomain.z_n))

        dndx = -0.5 * c ** 2 * np.gradient(self.ne_nc, x, axis = 0)
        dndy = -0.5 * c ** 2 * np.gradient(self.ne_nc, y, axis = 1)
        dndz = -0.5 * c ** 2 * np.gradient(self.ne_nc, z, axis = 2)

        args = (self, parallelise, x, y, z, dndx, dndy, dndz)
        sol = jax.vmap(lambda s: ODE_solve(s, args))(s0.T)

        print("Time to ray trace:", time() - start)

        return ray_to_Jonesvector(sol.ys[:, -1, :].T, self.extent, probing_direction = self.Beam.probing_direction)

# ODEs of photon paths, standalone function to support the solve()
def dsdt(t, s, Propagator, parallelise, x, y, z, dndx, dndy, dndz):
    # forces s to be a matrix even if has the indexes of a 1d array such that dsdt() can be generalised
    s = jnp.reshape(s, (9, 1))  # one ray per vmap iteration if parallelised

    sprime = jnp.zeros_like(s)

    r = s[:3, :]
    v = s[3:6, :]

    a = s[6, :]

    sprime = sprime.at[3:6, :].set(Propagator.dndr(r))
    #sprime = sprime.at[3:6, :].set(Propagator.dndr_test(r, x, y, z, dndx, dndy, dndz))
    sprime = sprime.at[:3, :].set(v)

    return sprime.flatten()

def ray_to_Jonesvector(ode_sol, ne_extent, probing_direction):
    Np = ode_sol.shape[1] # number of photons

    ray_p = np.zeros((4, Np))

    x, y, z, vx, vy, vz = ode_sol[0], ode_sol[1], ode_sol[2], ode_sol[3], ode_sol[4], ode_sol[5]

    t_bp = (z - ne_extent) / vz

    # Positions on plane
    ray_p[0] = x - vx * t_bp
    ray_p[2] = y - vy * t_bp

    # Angles to plane
    ray_p[1] = np.arctan(vx / vz)
    ray_p[3] = np.arctan(vy / vz)

    return ray_p