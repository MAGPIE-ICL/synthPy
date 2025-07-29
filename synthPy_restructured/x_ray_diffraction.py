import numpy as np
from scipy.signal.windows import tukey
from scipy.interpolate import CloughTocher2DInterpolator as CT2D
from scipy.interpolate import LinearNDInterpolator as LND
from scipy.interpolate import NearestNDInterpolator as NND
from scipy.interpolate import RegularGridInterpolator as RGI

#@staticmethod
def prepare_field_for_propagation(U0, pad_factor = 2, alpha = 0.4):
    """
    Prepares a field for propagation using reflection padding and a Tukey window.
    """
    pad_width_x = U0.shape[0] * pad_factor
    pad_width_y = U0.shape[1] * pad_factor
    pad_widths = ((pad_width_x, pad_width_x), (pad_width_y, pad_width_y))
    U0_padded = np.pad(U0, pad_widths, mode = 'reflect')
    tukey_x = tukey(U0_padded.shape[0], alpha = alpha)
    tukey_y = tukey(U0_padded.shape[1], alpha = alpha)
    window_2d = np.outer(tukey_x, tukey_y)
    return U0_padded * window_2d
    
#@staticmethod
def fresnel_propagate(U0_prepared, L, wavelength, z, original_shape, pad_factor=2, lanex_fwhm_m=None):
    """
    Propagates a prepared field using the Fresnel approximation and optionally applies the LANEX PSF.
    """
    Nx_orig, Ny_orig = original_shape
    Lx, Ly = L
    Nx_pad, Ny_pad = U0_prepared.shape
    dx, dy = Lx / Nx_orig, Ly / Ny_orig
    k = 2 * np.pi / wavelength
    
    fx = np.fft.fftfreq(Nx_pad, d=dx)
    fy = np.fft.fftfreq(Ny_pad, d=dy)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    
    U0_ft = np.fft.fft2(U0_prepared)
    Uz_ft = U0_ft * H
    
    # --- LANEX PSF Convolution in Fourier Domain (if FWHM is provided) ---
    if lanex_fwhm_m is not None and lanex_fwhm_m > 0:
        sigma = lanex_fwhm_m / (2 * np.sqrt(2 * np.log(2))) # Convert FWHM to sigma
        psf_ft = np.exp(-2 * (np.pi * sigma)**2 * (FX**2 + FY**2))
        Uz_ft = Uz_ft * psf_ft
    
    Uz_padded = np.fft.ifft2(Uz_ft)
    Uz_padded *= np.exp(1j * k * z) / (1j * wavelength * z)

    pad_width_x = Nx_orig * pad_factor
    pad_width_y = Ny_orig * pad_factor
    start_x, end_x = pad_width_x, pad_width_x + Nx_orig
    start_y, end_y = pad_width_y, pad_width_y + Ny_orig
    
    return Uz_padded[start_x:end_x, start_y:end_y]

def propagate(Propagator, jones_vector, amplitudes, phases, z, pad_factor = 2):
    """
    Prepares and propagates the field, using an energy-dependent PSF.
    """
    #print("--- Initialising Propagation ---")
    wavelength = Propagator.Beam.wavelength
    # if self.group_centres is None or self.phase_data is None:
    #     raise ValueError("Data must be read before propagation. Call read_data() and read_tabular_spectrum() first.")

    # Determine which data block corresponds to the selected energy
    # self.calc_plot_idx()
    # actual_energy = self.group_centres[self.block_index]
    # print(f"Selected energy: {self.selected_energies} eV. Using closest data block at {actual_energy:.2f} eV.")

    # wavelength = self.ev_to_wavelength(Propagator.energy)
    # k = 2 * np.pi / wavelength
    N_f = (Propagator.ScalarDomain.x_length)**2 / (wavelength * z)
    print(f"Fresnel Number: {N_f:.4f}")

    # --- Get the energy-dependent PSF ---
    # lanex_fwhm = self.get_lanex_fwhm_by_energy(actual_energy)
    # print(f"LANEX FWHM for {actual_energy:.2f} eV is {lanex_fwhm*1e6:.2f} µm.")

    # attenuation_slice = self.attenuation_data[self.block_index, :, :]
    # phase_slice = self.phase_data[self.block_index, :, :]

    x_positions = jones_vector[0]
    y_positions = jones_vector[2]
    amplitudes = amplitudes[:, -1]
    phases = phases[:, -1]

    # phases_interp = LND((x_positions, y_positions), phases, fill_value = 0.0)
    # amplitudes_interp = LND((x_positions, y_positions), amplitudes, fill_value = 0.0)

    x_bins = Propagator.ScalarDomain.x
    y_bins = Propagator.ScalarDomain.y

    field_grid = np.zeros((len(x_bins)-1, len(y_bins)-1), dtype = complex)
    
    x_indices = np.digitize(x_positions, x_bins) - 1
    y_indices = np.digitize(y_positions, y_bins) - 1
    
    for i in range(0, len(x_positions)):
        if x_indices[i] < field_grid.shape[0] and y_indices[i] < field_grid.shape[1]:
            field_grid[y_indices[i], x_indices[i]] += amplitudes[i] * np.exp(-1j * phases[i])
        



    XX, YY = np.meshgrid(Propagator.ScalarDomain.x, Propagator.ScalarDomain.y)
    # phase_grid = phases_interp((XX, YY))
    # amplitude_grid = amplitudes_interp((XX, YY))
    # U_0 = np.exp(-1j * k * (phase_slice - 1j * attenuation_slice))
    # U_0 = amplitude_grid * np.exp(-1j  * phase_grid)
    U_0 = field_grid
    U_0_prepared = prepare_field_for_propagation(U_0, pad_factor = pad_factor)
    
    # Pass the dynamically calculated FWHM to the propagation function
    U_0_proped = fresnel_propagate(
        U_0_prepared, (Propagator.ScalarDomain.x_length, Propagator.ScalarDomain.y_length), 
        wavelength, z, U_0.shape, pad_factor = pad_factor, lanex_fwhm_m = None
    )
    
    #print("--- Propagation Complete ---")
    return U_0_proped