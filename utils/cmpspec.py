# CMPSPEC contains METHODS TO COMPUTE THE POWER SPECTRUM
# GIVEN AS INPUT 1D, 2D, 3D FIELD


"""
Author: Stefano Merlini
date: 14/05/20

"""


#   ___  __   _  _  ____  _  _  ____  __  __ _   ___    ____  ____  ____  ___  ____  ____   __  
#  / __)/  \ ( \/ )(  _ \/ )( \(_  _)(  )(  ( \ / __)  / ___)(  _ \(  __)/ __)(_  _)(  _ \ / _\ 
# ( (__(  O )/ \/ \ ) __/) \/ (  )(   )( /    /( (_ \  \___ \ ) __/ ) _)( (__   )(   )   //    \
#  \___)\__/ \_)(_/(__)  \____/ (__) (__)\_)__) \___/  (____/(__)  (____)\___) (__) (__\_)\_/\_/








import numpy as np
from numpy.fft import fftn


#  ____  _  _   __    __  ____  _  _ 
# / ___)( \/ ) /  \  /  \(_  _)/ )( \
# \___ \/ \/ \(  O )(  O ) )(  ) __ (
# (____/\_)(_/ \__/  \__/ (__) \_)(_/
# Function for smoothing the spectrum
# only for visualisation

def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')



#   __      ____    ____  __  ____  __    ____    ____  ____  ____  ___  ____  ____  _  _  _  _ 
#  /  \ ___(    \  (  __)(  )(  __)(  )  (    \  / ___)(  _ \(  __)/ __)(_  _)(  _ \/ )( \( \/ )
# (_/ /(___)) D (   ) _)  )(  ) _) / (_/\ ) D (  \___ \ ) __/ ) _)( (__   )(   )   /) \/ (/ \/ \
#  (__)    (____/  (__)  (__)(____)\____/(____/  (____/(__)  (____)\___) (__) (__\_)\____/\_)(_/
# Method to compute spectrum from 1D Field

def compute1Dspectrum(r,lx, smooth =  False):
    """
     Parameters:
    ----------------------------------------------------------------
    r:  float-vector
        The 1D random field
    lx: float
        the domain size in the x-direction.
    nx: integer
        the number of grid points in the x-direction
    smooth: boolean
        Active/Disactive smooth function for visualisation
    -----------------------------------------------------------------
"""
    nx = len(r)
    nt = nx
    n = nx
    rh = fftn(r)/nt
    # calculate energy in fourier domain
    tkeh = (rh * np.conj(rh)).real
    k0x = 2.0*np.pi/lx
    knorm = k0x
    kxmax = nx / 2
    wave_numbers = knorm*np.arange(0,n) # array of wavenumbers
    tke_spectrum = np.zeros(len(wave_numbers))
    for kx in range(-nx//2, nx//2-1):
        rk = np.sqrt(kx**2)
        k = int(np.round(rk))
        tke_spectrum[k] = tke_spectrum[k] + tkeh[kx]
    tke_spectrum = tke_spectrum/knorm
    knyquist = knorm * nx / 2
    # If smooth parameter is TRUE: Smooth the computed spectrum
    # ONLY for Visualisation
    if smooth:
        tkespecsmooth = movingaverage(tke_spectrum, 5)  # smooth the spectrum
        tkespecsmooth[0:4] = tke_spectrum[0:4]  # get the first 4 values from the original data
        tke_spectrum = tkespecsmooth
    #
    return knyquist, wave_numbers, tke_spectrum


#  ____      ____    ____  __  ____  __    ____    ____  ____  ____  ___  ____  ____  _  _  _  _ 
# (___ \ ___(    \  (  __)(  )(  __)(  )  (    \  / ___)(  _ \(  __)/ __)(_  _)(  _ \/ )( \( \/ )
#  / __/(___)) D (   ) _)  )(  ) _) / (_/\ ) D (  \___ \ ) __/ ) _)( (__   )(   )   /) \/ (/ \/ \
# (____)    (____/  (__)  (__)(____)\____/(____/  (____/(__)  (____)\___) (__) (__\_)\____/\_)(_/
# Method to compute spectrum from 2D Field

def compute2Dspectrum(r,lx, ly, smooth = False):
    """
     Parameters:
    ----------------------------------------------------------------
    r:  float-vector
        The 2D random field
    lx: float
        the domain size in the x-direction.
    nx: integer
        the number of grid points in the x-direction
    smooth: boolean
        Active/Disactive smooth function for visualisation
    -----------------------------------------------------------------
"""
    nx = len(r[:,0])
    ny = len(r[0,:])
    nt = nx*ny
    n = nx
    rh = fftn(r)/nt
    # calculate energy in fourier domain
    tkeh = (np.abs(rh)**2) / (nx * ny)**2
    k0x = 2.0*np.pi/lx
    k0y = 2.0*np.pi/ly
    knorm = (k0x + k0y) / 2.0

    wave_numbers = knorm*np.arange(0,n)
    tke_spectrum = np.zeros(len(wave_numbers))

    for kx in range(-nx//2, nx//2-1):
       for ky in range(-ny//2, ny//2-1):
           rk = np.sqrt(kx**2 + ky**2)
           k = int(np.round(rk))
           tke_spectrum[k] = tke_spectrum[k] + tkeh[kx, ky]
    tke_spectrum = tke_spectrum/knorm
    knyquist = knorm * min(nx, ny) / 2
    # If smooth parameter is TRUE: Smooth the computed spectrum
    # ONLY for Visualisation
    if smooth:
        tkespecsmooth = movingaverage(tke_spectrum, 5)  # smooth the spectrum
        tkespecsmooth[0:4] = tke_spectrum[0:4]  # get the first 4 values from the original data
        tke_spectrum = tkespecsmooth

    return knyquist, wave_numbers, tke_spectrum


import numpy as np
from numpy.fft import fft2, fftshift

def compute2Dspectrum_new(r, lx, ly, smooth=False):
    nx, ny = r.shape
    
    rh = fftshift(fft2(r))
    

    tkeh = (np.abs(rh)**2) / (nx * ny)**2  # Normalized power spectrum
    
    # Set up wavenumbers
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx/nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly/ny)
    kx, ky = np.meshgrid(kx, ky)
    kx, ky = fftshift(kx), fftshift(ky)
    k = np.sqrt(kx**2 + ky**2)
    
    # Make radial bins, evenly spaced in logspace for ease of plotting
    k_bins = np.logspace(np.log10(k[k>0].min()), np.log10(k.max()), num=50)
    tke_spectrum = np.zeros(len(k_bins)-1)
    
    for i in range(len(k_bins)-1):
        mask = (k >= k_bins[i]) & (k < k_bins[i+1])
        tke_spectrum[i] = np.mean(tkeh[mask])
    
    k_centers = np.sqrt(k_bins[:-1] * k_bins[1:])
    
    knyquist = np.max(k) / 2

    if smooth:
        tke_spectrum = movingaverage(tke_spectrum, 5)
    
    return knyquist, k_centers, tke_spectrum


#  ____      ____    ____  __  ____  __    ____    ____  ____  ____  ___  ____  ____  _  _  _  _ 
# ( __ \ ___(    \  (  __)(  )(  __)(  )  (    \  / ___)(  _ \(  __)/ __)(_  _)(  _ \/ )( \( \/ )
#  (__ ((___)) D (   ) _)  )(  ) _) / (_/\ ) D (  \___ \ ) __/ ) _)( (__   )(   )   /) \/ (/ \/ \
# (____/    (____/  (__)  (__)(____)\____/(____/  (____/(__)  (____)\___) (__) (__\_)\____/\_)(_/
# Method to compute spectrum from 3D Field

def compute3Dspectrum(r,lx, ly, lz, smooth = False):
    """
     Parameters:
    ----------------------------------------------------------------
    r:  float-vector
        The 3D random field
    lx: float
        the domain size in the x-direction.
    nx: integer
        the number of grid points in the x-direction
    smooth: boolean
        Active/Disactive smooth function for visualisation
    -----------------------------------------------------------------
"""
    nx = len(r[:,0,0])
    ny = len(r[0,:,0])
    nz = len(r[0,0,:])
    nt = nx*ny*nz
    n = nx
    rh = fftn(r)/nt
    # calculate energy in fourier domain
    tkeh = (rh * np.conj(rh)).real
    k0x = 2.0*np.pi/lx
    k0y = 2.0*np.pi/ly
    k0z = 2.0*np.pi/lz
    knorm = (k0x + k0y + k0z) / 3.0
    kxmax = nx / 2
    kymax = ny / 2
    kzmax = nz / 2
    wave_numbers = knorm*np.arange(0,n)
    tke_spectrum = np.zeros(len(wave_numbers))

    for kx in range(-nx//2, nx//2-1):
       for ky in range(-ny//2, ny//2-1):
           for kz in range(-nz//2, nz//2-1):
               rk = np.sqrt(kx**2 + ky**2 + kz**2)
               k = int(np.round(rk))
               tke_spectrum[k] = tke_spectrum[k] + tkeh[kx, ky, kz]
    tke_spectrum = tke_spectrum/knorm
    knyquist = knorm * min(nx, ny, nz) / 2
    # If smooth parameter is TRUE: Smooth the computed spectrum
    # ONLY for Visualisation
    if smooth:
        tkespecsmooth = movingaverage(tke_spectrum, 5)  # smooth the spectrum
        tkespecsmooth[0:4] = tke_spectrum[0:4]  # get the first 4 values from the original data
        tke_spectrum = tkespecsmooth
    #
    return knyquist, wave_numbers, tke_spectrum


def compute3Dspectrum_new(r, lx, ly, lz, smooth=False):
    nx, ny, nz = r.shape
    
    rh = fftshift(fftn(r))
    

    tkeh = (np.abs(rh)**2) / (nx * ny * nz)**2  # Normalized power spectrum
    
    # Set up wavenumbers
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx/nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly/ny)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=lz/nz)
    kx, ky, kz = np.meshgrid(kx, ky, kz)
    kx, ky, kz = fftshift(kx), fftshift(ky), fftshift(kz)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    
    # Make radial bins, evenly spaced in logspace for ease of plotting
    k_bins = np.logspace(np.log10(k[k>0].min()), np.log10(k.max()), num=50)
    tke_spectrum = np.zeros(len(k_bins)-1)
    
    for i in range(len(k_bins)-1):
        mask = (k >= k_bins[i]) & (k < k_bins[i+1])
        tke_spectrum[i] = np.mean(tkeh[mask])
    
    k_centers = np.sqrt(k_bins[:-1] * k_bins[1:])
    
    knyquist = np.max(k) / 2

    if smooth:
        tke_spectrum = movingaverage(tke_spectrum, 5)
    
    return knyquist, k_centers, tke_spectrum







