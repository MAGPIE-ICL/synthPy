# from jax import numpy as jnp - would that work too? what's the difference if so?
import jax.numpy as jnp

from jax.scipy.interpolate import RegularGridInterpolator
from equinox import Module

class interpolation_setup(Module):
    ne_interp: RegularGridInterpolator

    Bx_interp: RegularGridInterpolator
    By_interp: RegularGridInterpolator
    Bz_interp: RegularGridInterpolator

    kappa_interp: RegularGridInterpolator

    refractive_index_interp: RegularGridInterpolator

    def __init__(self, ScalarDomain, omega):
        # Defaults: Value of None so there is something to pass even if not allocateds
        self.Bx_interp = None
        self.By_interp = None
        self.Bz_interp = None

        self.kappa_interp = None

        self.refractive_index_interp = None

        # Electron density
        self.ne_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), ScalarDomain.ne, bounds_error = False, fill_value = 0.0)

        # Magnetic field
        if (ScalarDomain.B_on):
            self.Bx_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), ScalarDomain.B[:,:,:,0], bounds_error = False, fill_value = 0.0)
            self.By_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), ScalarDomain.B[:,:,:,1], bounds_error = False, fill_value = 0.0)
            self.Bz_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), ScalarDomain.B[:,:,:,2], bounds_error = False, fill_value = 0.0)

        # Inverse Bremsstrahlung
        if(ScalarDomain.inv_brems):
            self.kappa_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), kappa(ScalarDomain, omega), bounds_error = False, fill_value = 0.0)

        # Phase shift
        if(ScalarDomain.phaseshift):
            self.refractive_index_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), n_refrac(ScalarDomain.ne, omega), bounds_error = False, fill_value = 1.0)

    def omega_pe(self, ne):
        """Calculate electron plasma freq. Output units are rad/sec. From nrl pp 28"""

        return 5.64e4 * jnp.sqrt(ne)

    # NRL formulary inverse brems - cheers Jack Halliday for coding in Python
    # Converted to rate coefficient by multiplying by group velocity in plasma
    def kappa(self, ScalarDomain, omega):
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
    def n_refrac(self, ne, omega):
        return jnp.sqrt(1.0 - (omega_pe(ne * 1e-6) / omega) ** 2)