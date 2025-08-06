import numpy as np

from scipy.signal.windows import tukey
from scipy.interpolate import CloughTocher2DInterpolator as CT2D
from scipy.interpolate import LinearNDInterpolator as LND

#@staticmethod
def prepare_field_for_propagation(U0, pad_factor = 2, alpha = 0.4):
    """
    Prepares a field for propagation using reflection padding and a Tukey window.
    """

    pad_width_x = U0.shape[0] * pad_factor
    pad_width_y = U0.shape[1] * pad_factor

    U0_padded = np.pad(U0, ((pad_width_x, pad_width_x), (pad_width_y, pad_width_y)), mode = 'reflect')

    window_2d = np.outer(
        tukey(U0_padded.shape[0], alpha = alpha), 
        tukey(U0_padded.shape[1], alpha = alpha)
    )

    return U0_padded * window_2d

#@staticmethod
def fresnel_propagate(U0_prepared, L, wavelength, z, original_shape, pad_factor=2, lanex_fwhm_m=None):
    """
    Propagates a prepared field using the Fresnel approximation and optionally applies the LANEX PSF.
    """

    # can be modified for rectangular case? or is this an expansion?
    Nx_orig, Ny_orig = original_shape

    dx, dy = L[0] / Nx_orig, L[1] / Ny_orig

    fx = np.fft.fftfreq(U0_prepared.shape[0], d = dx)
    fy = np.fft.fftfreq(U0_prepared.shape[1], d = dy)

    FX, FY = np.meshgrid(fx, fy, indexing = 'ij')

    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    
    U0_ft = np.fft.fft2(U0_prepared)
    Uz_ft = U0_ft * H

    # --- LANEX PSF Convolution in Fourier Domain (if FWHM is provided) ---
    if lanex_fwhm_m is not None and lanex_fwhm_m > 0:
        sigma = lanex_fwhm_m / (2 * np.sqrt(2 * np.log(2))) # Convert FWHM to sigma
        psf_ft = np.exp(-2 * (np.pi * sigma)**2 * (FX**2 + FY**2))
        Uz_ft *= psf_ft
    
    Uz_padded = np.fft.ifft2(Uz_ft) * np.exp(1j * (2 * np.pi / wavelength) * z) / (1j * wavelength * z)

    pad_width_x = Nx_orig * pad_factor
    pad_width_y = Ny_orig * pad_factor

    start_x, end_x = pad_width_x, pad_width_x + Nx_orig
    start_y, end_y = pad_width_y, pad_width_y + Ny_orig

    return Uz_padded[start_x:end_x, start_y:end_y]

def propagate(lwl, x, y, x_length, y_length, jones_vector, amplitudes, phases, z, pad_factor = 2):
    """
    Prepares and propagates the field, using an energy-dependent PSF.
    """

    N_f = (x_length)**2 / (lwl * z)
    #print(f"Fresnel Number: {N_f:.4f}")

    x_positions = jones_vector[0]
    y_positions = jones_vector[2]

    phases_interp = LND((x_positions, y_positions), phases, fill_value = 0.0)
    amplitudes_interp = LND((x_positions, y_positions), amplitudes, fill_value = 0.0)

    XX, YY = np.meshgrid(x, y)

    phase_grid = phases_interp((XX, YY))
    amplitude_grid = amplitudes_interp((XX, YY))

    del XX
    del YY

    U_0 = amplitude_grid * np.exp(-1j * phase_grid)
    #U_0 = amplitudes_interp(np.meshgrid(x, y))
    #U_0 *= np.exp(-1j * phases_interp(np.meshgrid(x, y)))
    U_0_prepared = prepare_field_for_propagation(U_0, pad_factor = pad_factor)

    # Pass the dynamically calculated FWHM to the propagation function
    U_0_proped = fresnel_propagate(
        U_0_prepared, (x_length, y_length), 
        lwl, z, U_0.shape, pad_factor = pad_factor, lanex_fwhm_m = None
    )

    return U_0_proped