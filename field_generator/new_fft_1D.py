import numpy as np

def fft_1D(l_max, l_min, extent, res, spectrum):
    print('we are in!')
    dx = extent / (res)

    x = np.arange(-extent, extent, dx)

    dk = 2*np.pi/l_min

    k = np.fft.fftfreq(2*res, d = dx) * 2 * np.pi

    S = np.where((k >= 2*np.pi/l_max) & (k< 2* np.pi / l_min), spectrum(k), 0)

    std = (S/2)**0.5

    real = np.random.normal(0, std, k.shape)

    imag = np.random.normal(0, std, k.shape)

    spectrum = (real + 1j*imag)*np.sqrt(S)
    noise = np.fft.ifft(spectrum)

    noise = np.abs(noise)

    noise = noise

    # normalise



    return x, noise




