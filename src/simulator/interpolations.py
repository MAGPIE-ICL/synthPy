from jax.scipy.interpolate import RegularGridInterpolator

def set_up_interps(ScalarDomain, inv_brems, phaseshift):
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
    if(inv_brems):
        kappa_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), kappa(), bounds_error = False, fill_value = 0.0)

    # Phase shift
    if(phaseshift):
        refractive_index_interp = RegularGridInterpolator((ScalarDomain.x, ScalarDomain.y, ScalarDomain.z), n_refrac(), bounds_error = False, fill_value = 1.0)

    return {
        "ne_interp": ne_interp,
        "Bx_interp": ne_interp,
        "By_interp": By_interp,
        "Bz_interp": Bz_interp,
        "kappa_interp": kappa_interp,
        "refractive_index_interp": refractive_index_interp
    }