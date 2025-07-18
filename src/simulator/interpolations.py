from jax.scipy.interpolate import RegularGridInterpolator
# from jax import numpy as jnp - would that work too? what's the difference if so?
import jax.numpy as jnp

def omega_pe(ne):
    """Calculate electron plasma freq. Output units are rad/sec. From nrl pp 28"""

    return 5.64e4 * jnp.sqrt(ne)

# NRL formulary inverse brems - cheers Jack Halliday for coding in Python
# Converted to rate coefficient by multiplying by group velocity in plasma
def kappa(ScalarDomain, omega):
    # Useful subroutines
    def v_the(Te):
        """Calculate electron thermal speed. Provide Te in eV. Retrurns result in m/s"""

        return 4.19e5 * jnp.sqrt(Te)

    def V(ne, Te, Z, omega):
        o_pe = omega_pe(ne)
        #o_max = jnp.copy(o_pe)
        #o_max[o_pe < omega] = omega
        o_pe = o_pe.at[:, :].set(jnp.where(o_pe < omega, omega, o_pe))
        L_classical = Z * e / Te
        L_quantum = 2.760428269727312e-10 / jnp.sqrt(Te) # hbar / jnp.sqrt(m_e * e * Te)
        L_max = jnp.maximum(L_classical, L_quantum)

        #return o_max * L_max
        return o_pe * L_max

    def coloumbLog(ne, Te, Z, omega):
        return jnp.maximum(2.0, jnp.log(v_the(Te) / V(ne, Te, Z, omega)))

    ne_cc = ScalarDomain.ne * 1e-6
    # don't think this is actually used?
    #o_pe = omega_pe(ne_cc)
    CL = coloumbLog(ne_cc, ScalarDomain.Te, ScalarDomain.Z, omega)

    result = 3.1e-5 * ScalarDomain.Z * c * jnp.power(ne_cc / omega, 2) * CL * jnp.power(ScalarDomain.Te, -1.5) # 1/s
    del ne_cc

    return result

# Plasma refractive index
def n_refrac(ne, omega):
    return jnp.sqrt(1.0 - (omega_pe(ne * 1e-6) / omega) ** 2)

def set_up_interps(ScalarDomain, omega):
    # Defaults:
    ne_interp = None
    Bx_interp = None
    By_interp = None
    Bz_interp = None
    kappa_interp = None
    refractive_index_interp = None

    # Electron density
    ne_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), ScalarDomain.ne, bounds_error = False, fill_value = 0.0)

    # Magnetic field
    if (ScalarDomain.B_on):
        Bx_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), ScalarDomain.B[:,:,:,0], bounds_error = False, fill_value = 0.0)
        By_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), ScalarDomain.B[:,:,:,1], bounds_error = False, fill_value = 0.0)
        Bz_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), ScalarDomain.B[:,:,:,2], bounds_error = False, fill_value = 0.0)

    # Inverse Bremsstrahlung
    if(ScalarDomain.inv_brems):
        kappa_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), kappa(ScalarDomain, omega), bounds_error = False, fill_value = 0.0)

    # Phase shift
    if(ScalarDomain.phaseshift):
        refractive_index_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), n_refrac(ScalarDomain.ne, omega), bounds_error = False, fill_value = 1.0)

    return {
        "ne_interp": ne_interp,
        "Bx_interp": ne_interp,
        "By_interp": By_interp,
        "Bz_interp": Bz_interp,
        "kappa_interp": kappa_interp,
        "refractive_index_interp": refractive_index_interp
    }