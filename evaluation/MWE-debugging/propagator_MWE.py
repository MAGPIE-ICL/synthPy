import jax
import jax.numpy as jnp

from jax.scipy.interpolate import RegularGridInterpolator

from scipy.constants import c
from scipy.constants import e

def calc_dndr(ne, lwl = 1064e-9):
    omega = 2 * jnp.pi * c / lwl
    nc = 3.14207787e-4 * omega ** 2

    return jnp.array(ne / nc, dtype = jnp.float32)

def dndr(r, ne_nc, x, y, z):
    grad = jnp.zeros_like(r)

    dndx = -0.5 * c ** 2 * jnp.gradient(ne_nc, x, axis = 0)
    dndx_interp = RegularGridInterpolator((x, y, z), dndx, bounds_error = False, fill_value = 0.0)
    del dndx

    grad = grad.at[0, :].set(dndx_interp(r.T))
    del dndx_interp

    dndy = -0.5 * c ** 2 * jnp.gradient(ne_nc, y, axis = 1)
    dndy_interp = RegularGridInterpolator((x, y, z), dndy, bounds_error = False, fill_value = 0.0)
    del dndy

    grad = grad.at[1, :].set(dndy_interp(r.T))
    del dndy_interp

    dndz = -0.5 * c ** 2 * jnp.gradient(ne_nc, z, axis = 2)
    dndz_interp = RegularGridInterpolator((x, y, z), dndz, bounds_error = False, fill_value = 0.0)
    del dndz

    grad = grad.at[2, :].set(dndz_interp(r.T))
    del dndz_interp

    return grad

def dsdt(t, s, ne_nc, x, y, z):
    s = jnp.reshape(s, (9, 1))
    sprime = jnp.zeros_like(s)

    r = s[:3, :]
    v = s[3:6, :]

    a = s[6, :]

    sprime = sprime.at[3:6, :].set(dndr(r, ne_nc, x, y, z))
    sprime = sprime.at[:3, :].set(v)

    return sprime.flatten()

def solve(s0_import, ne_nc, x, y, z, x_n, y_n, z_n, extent):
    Np = s0_import.shape[1]
    s0 = s0_import.T
    del s0_import

    t = jnp.linspace(0.0, jnp.sqrt(8.0) * extent / c, 2)
    norm_factor = jnp.max(t)

    def dsdt_ODE(t, y, args):
        return dsdt(t, y, *args) * norm_factor

    import diffrax

    def diffrax_solve(dydt, t0, t1, Nt, rtol = 1e-7, atol = 1e-9):
        term = diffrax.ODETerm(dydt)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts = jnp.linspace(t0, t1, Nt))
        stepsize_controller = diffrax.PIDController(rtol = rtol, atol = atol)

        return lambda s0, args : diffrax.diffeqsolve(
            term,
            solver,
            y0 = jnp.array(s0),
            args = args,
            t0 = t0,
            t1 = t1,
            dt0 = (t1 - t0) * norm_factor / Nt,
            saveat = saveat,
            stepsize_controller = stepsize_controller,
            # set max steps to no. of cells x100
            max_steps = x_n * y_n * z_n * 100 #10000 - default for solve_ivp?????
        )

    from equinox import filter_jit
    ODE_solve = filter_jit(diffrax_solve(dsdt_ODE, t[0], t[-1] / norm_factor, 2))

    args = (ne_nc, x, y, z)
    sol = jax.block_until_ready(jax.vmap(lambda s: ODE_solve(s, args))(s0))

    del ne_nc

    return sol.ys[:, -1, :].T