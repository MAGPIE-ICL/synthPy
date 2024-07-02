import numpy as np
import scipy.fftpack as fft



def fft_1D(l_max, l_min, extent, res, spectrum):
    dx = extent / res
    x = np.linspace(-extent, extent, 2*res, endpoint=False)


    k = 2 * np.pi * np.fft.fftfreq(2*res, d=dx)



    k_min = 2 * np.pi / l_max
    k_max = 2 * np.pi / l_min

    # Create the power spectrum
    S = np.zeros_like(k)
    mask = (k >= k_min) & (k <= k_max)
    S[mask] = spectrum(k[mask])

    # Generate complex Gaussian noise
    noise = np.random.normal(0, 1, k.shape) + 1j * np.random.normal(0, 1, k.shape)

    # Apply the power spectrum
    fft_field = noise * np.sqrt(S)

    # Inverse Fourier transform 
    field = np.fft.ifft(fft_field).real

    return x, field

import numpy as np






def fft_2D(l_max, l_min, extent, res, spectrum):
    dx = extent / res
    x = y = np.linspace(-extent, extent, 2*res, endpoint=False)
    xx, yy = np.meshgrid(x, y)

    kx = ky = 2 * np.pi * np.fft.fftfreq(2*res, d=dx)
    kxx, kyy = np.meshgrid(kx, ky)
    k = np.sqrt(kxx**2 + kyy**2)

    k_min = 2 * np.pi / l_max
    k_max = 2 * np.pi / l_min

    # Create the power spectrum
    S = np.zeros_like(k)
    mask = (k >= k_min) & (k <= k_max)
    S[mask] = spectrum(k[mask])

    # Generate complex Gaussian noise
    noise = np.random.normal(0, 1, k.shape) + 1j * np.random.normal(0, 1, k.shape)

    # Apply the power spectrum
    fft_field = noise * np.sqrt(S)

    # Inverse Fourier transform 
    field = np.fft.ifft2(fft_field).real



    return xx, yy, field

def fft_3D(l_max, l_min, extent, res, spectrum):
    dx = extent / res
    x = y = z =  np.linspace(-extent, extent, 2*res, endpoint=False)
    xx, yy, zz = np.meshgrid(x, y, z)

    kx = ky = kz =  2 * np.pi * np.fft.fftfreq(2*res, d=dx)
    kxx, kyy, kzz = np.meshgrid(kx, ky, kz)
    k = np.sqrt(kxx**2 + kyy**2 + kzz**2)

    k_min = 2 * np.pi / l_max
    k_max = 2 * np.pi / l_min

    # Create the power spectrum
    S = np.zeros_like(k)
    mask = (k >= k_min) & (k <= k_max)
    S[mask] = spectrum(k[mask])

    # Generate complex Gaussian noise
    noise = np.random.normal(0, 1, k.shape) + 1j * np.random.normal(0, 1, k.shape)

    # Apply the power spectrum
    fft_field = noise * np.sqrt(S)

    # Inverse Fourier transform 
    field = np.fft.ifftn(fft_field).real

 
    return xx, yy, zz, field



