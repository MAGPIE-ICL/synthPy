"""METHODS TO COMPUTE POWER SPECTRUM FROM SCALAR FIELDS
Author: Stefano Merlini
Date: 25/06/24
"""


import numpy as np

#  Method 1 - Calculate Power Spectrum 

def scalar1D_fft(data, dx, k_bin_num=100):
    """Calculates and returns the 2D spectrum for a 2D gaussian field of scalars, assuming isotropy of the turbulence
        Example:
            d=np.random.randn(101,101)
            dx=1
            k_bins_weighted,spect3D=spectrum_2D_scalar(d, dx, k_bin_num=100)

            fig,ax=plt.subplots()
            ax.scatter(k_bins_weighted,spect3D)
    Arguments:
        data {(Mx,My) array of floats} -- 2D Gaussian field of scalars
        dx {float} -- grid spacing, assumed the same for all
        k_bin_num {int} -- number of bins in reciprocal space

    Returns:
        k_bins_weighted {array of floats} -- location of bin centres
        spect2D {array of floats} -- spectral power within bin
    """

    #fourier transform data, shift to have zero freq at centre, find power
    f=np.fft.fftshift(np.fft.fftn(data))
    fsqr=np.real(f*np.conj(f))

    #calculate k vectors in each dimension
    Mx = data.shape[0]

    kx = np.fft.fftshift(np.fft.fftfreq(Mx, dx))

    #calculate magnitude of k at each grid point
    K = np.sqrt(kx**2)

    #determine 1D spectrum of k, measured from origin
    #sort array in ascending k, and sort power by the same factor

    K_flat=K.flatten()
    fsqr_flat=fsqr.flatten()

    K_sort = K_flat[K_flat.argsort()]
    fsqr_sort = fsqr_flat[K_flat.argsort()]
    
    k_bin_width = K_sort.max()/k_bin_num

    k_bins = k_bin_width*np.arange(0,k_bin_num+1)
    k_bins_weighted = 0.5*(k_bins[:-1]+k_bins[1:])
    
    spect1D=np.zeros_like(k_bins_weighted)

    for i in range(1,k_bin_num):
        upper=K_sort<i*k_bin_width # find only values below upper bound: BOOL
        lower=K_sort>=(i-1)*k_bin_width #find only values above upper bound: BOOL
        f_filtered=fsqr_sort[upper*lower] # use super numpy array filtering to select only those which match both!
        spect1D[i-1] = f_filtered.mean() #and take their mean.
        
    return k_bins_weighted, spect1D


def scalar2D_fft(data, dx, k_bin_num=100):
    """Calculates and returns the 2D spectrum for a 2D gaussian field of scalars, assuming isotropy of the turbulence
        Example:
            d=np.random.randn(101,101)
            dx=1
            k_bins_weighted,spect3D=spectrum_2D_scalar(d, dx, k_bin_num=100)

            fig,ax=plt.subplots()
            ax.scatter(k_bins_weighted,spect3D)
    Arguments:
        data {(Mx,My) array of floats} -- 2D Gaussian field of scalars
        dx {float} -- grid spacing, assumed the same for all
        k_bin_num {int} -- number of bins in reciprocal space

    Returns:
        k_bins_weighted {array of floats} -- location of bin centres
        spect2D {array of floats} -- spectral power within bin
    """

    #fourier transform data, shift to have zero freq at centre, find power
    f=np.fft.fftshift(np.fft.fftn(data))
    fsqr=np.real(f*np.conj(f))

    #calculate k vectors in each dimension
    [Mx,My] = data.shape

    kx = np.fft.fftshift(np.fft.fftfreq(Mx, dx))
    ky = np.fft.fftshift(np.fft.fftfreq(My, dx))

    #calculate magnitude of k at each grid point
    [KX,KY]=np.meshgrid(kx,ky)
    K=np.sqrt(KX**2+KY**2)

    #determine 1D spectrum of k, measured from origin
    #sort array in ascending k, and sort power by the same factor

    K_flat=K.flatten()
    fsqr_flat=fsqr.flatten()

    K_sort = K_flat[K_flat.argsort()]
    fsqr_sort = fsqr_flat[K_flat.argsort()]
    
    k_bin_width = K_sort.max()/k_bin_num

    k_bins = k_bin_width*np.arange(0,k_bin_num+1)
    k_bins_weighted = (0.5*(k_bins[:-1]**2+k_bins[1:]**2))**(1/2)
    
    spect2D=np.zeros_like(k_bins_weighted)

    for i in range(1,k_bin_num):
        upper=K_sort<i*k_bin_width # find only values below upper bound: BOOL
        lower=K_sort>=(i-1)*k_bin_width #find only values above upper bound: BOOL
        f_filtered=fsqr_sort[upper*lower] # use super numpy array filtering to select only those which match both!
        spect2D[i-1] = f_filtered.mean() #and take their mean.
        
    return k_bins_weighted, spect2D


def scalar3D_fft(data, dx, k_bin_num=100):
    """Calculates and returns the 3D spectrum for a 3D gaussian field of scalars, assuming isotropy of the turbulence
        Example:
            d=np.random.randn(101,91,111)
            dx=1
            k_bins_weighted,spect3D=spectrum_3D_scalar(d, dx, k_bin_width=0.01)

            fig,ax=plt.subplots()
            ax.scatter(k_bins_weighted,spect3D)
    Arguments:
        data {(Mx,My,Mz) array of floats} -- 3D Gaussian field of scalars
        dx {float} -- grid spacing, assumed the same for all
        k_bin_width {float} -- width of bins in reciprocal space

    Returns:
        k_bins_weighted {array of floats} -- location of bin centres
        spect3D {array of floats} -- spectral power within bin
    """

    #fourier transform data, shift to have zero freq at centre, find power
    f=np.fft.fftshift(np.fft.fftn(data))
    fsqr=np.real(f*np.conj(f))

    #calculate k vectors in each dimension
    [Mx,My,Mz] = data.shape

    kx = np.fft.fftshift(np.fft.fftfreq(Mx, dx))
    ky = np.fft.fftshift(np.fft.fftfreq(My, dx))
    kz = np.fft.fftshift(np.fft.fftfreq(Mz, dx))

    #calculate magnitude of k at each grid point
    [KX,KY,KZ]=np.meshgrid(kx,ky,kz, indexing='ij')
    K=np.sqrt(KX**2+KY**2+KZ**2)

    #determine 1D spectrum of k, measured from origin
    #sort array in ascending k, and sort power by the same factor

    K_flat=K.flatten()
    fsqr_flat=fsqr.flatten()

    K_sort = K_flat[K_flat.argsort()]
    fsqr_sort = fsqr_flat[K_flat.argsort()]
    
    k_bin_width = K_sort.max()/k_bin_num

    k_bins = k_bin_width*np.arange(0,k_bin_num+1)
    k_bins_weighted = (0.5*(k_bins[:-1]**3+k_bins[1:]**3))**(1/3)
    
    spect3D=np.zeros_like(k_bins_weighted)

    for i in range(1,k_bin_num):
        upper=K_sort<i*k_bin_width # find only values below upper bound: BOOL
        lower=K_sort>=(i-1)*k_bin_width #find only values above upper bound: BOOL
        f_filtered=fsqr_sort[upper*lower] # use super numpy array filtering to select only those which match both!
        spect3D[i-1] = f_filtered.mean() #and take their mean.
        
    return k_bins_weighted, spect3D

# Method 2 - Calculate Power Spectrum

#  ____  _  _   __    __  ____  _  _ 
# / ___)( \/ ) /  \  /  \(_  _)/ )( \
# \___ \/ \/ \(  O )(  O ) )(  ) __ (
# (____/\_)(_/ \__/  \__/ (__) \_)(_/
# Function for smoothing the spectrum
# only for visualisation

def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

def scalar1D_knyquist(r,lx, smooth = False):
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
    rh = np.fft.fftn(r)/nt
    # calculate energy in fourier domain
    tkeh = (rh * np.conj(rh)).real
    k0x = 2.0*np.pi/lx
    knorm = k0x
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

def scalar2D_knyquist(r,lx, ly, smooth = False):
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
    rh = np.fft.fftn(r)/nt
    # calculate energy in fourier domain
    tkeh = (rh * np.conj(rh)).real
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
    #
    return knyquist, wave_numbers, tke_spectrum

def scalar3D_knyquist(r,lx, ly, lz, smooth = False):
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
    rh = np.fft.fftn(r)/nt
    # calculate energy in fourier domain
    tkeh = (rh * np.conj(rh)).real
    k0x = 2.0*np.pi/lx
    k0y = 2.0*np.pi/ly
    k0z = 2.0*np.pi/lz
    knorm = (k0x + k0y + k0z) / 3.0
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




def radial_1Dspectrum(r, lx, smooth = False):
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
    import numpy as np
    from numpy.fft import fft2, fftshift, fftn, fft

    nx = len(r)
    
    rh = fftshift(fft(r))
    

    tkeh = (np.abs(rh)**2) / (nx)**2  # Normalized power spectrum
    
    # Set up wavenumbers
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx/nx)
    k = fftshift(kx)
    
    # Make radial bins, evenly spaced in logspace for ease of plotting
    k_bins = np.logspace(np.log10(k[k>0].min()), np.log10(k.max()), num=100)
    tke_spectrum = np.zeros(len(k_bins)-1)
    
    for i in range(len(k_bins)-1):
        mask = (k >= k_bins[i]) & (k < k_bins[i+1])
        tke_spectrum[i] = np.mean(tkeh[mask])
    
    k_centers = np.sqrt(k_bins[:-1] * k_bins[1:])
    
    knyquist = np.max(k) / 2

    if smooth:
        tke_spectrum = movingaverage(tke_spectrum, 5)
    
    return knyquist, k_centers, tke_spectrum



def radial_2Dspectrum(r, lx, ly, smooth=False):
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
    import numpy as np
    from numpy.fft import fft2, fftshift, fftn, fft
    
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
    k_bins = np.logspace(np.log10(k[k>0].min()), np.log10(k.max()), num=100)
    tke_spectrum = np.zeros(len(k_bins)-1)
    
    for i in range(len(k_bins)-1):
        mask = (k >= k_bins[i]) & (k < k_bins[i+1])
        tke_spectrum[i] = np.mean(tkeh[mask])
    
    k_centers = np.sqrt(k_bins[:-1] * k_bins[1:])
    
    knyquist = np.max(k) / 2

    if smooth:
        tke_spectrum = movingaverage(tke_spectrum, 5)
    
    return knyquist, k_centers, tke_spectrum



def radial_3Dspectrum(r, lx, ly, lz, smooth=False):
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
    import numpy as np
    from numpy.fft import fft2, fftshift, fftn, fft
    
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







