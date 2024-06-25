"""
Author: Stefano Merlini
Created: 14/05/2020
Modified: 24/06/2024
"""

import numpy as np

#   __      ____     ___   __   _  _  ____  ____  __   __   __ _ 
#  /  \ ___(    \   / __) / _\ / )( \/ ___)/ ___)(  ) / _\ (  ( \  
# (_/ /(___)) D (  ( (_ \/    \) \/ (\___ \\___ \ )( /    \/    /  
#  (__)    (____/   \___/\_/\_/\____/(____/(____/(__)\_/\_/\_)__)  

class gaussian1D:
    def __init__(self, k_func):

        """
            Parameters:
                k_func {function} -- a function which takes an input k 
        """
        self.xc = None
        self.k_func = k_func

    def cos(self, lx, nx, nmodes, wn1):
        """
        this method is from reference: 1988, Yamasaki, "Digital Generation of Non-Goussian Stochastic Fields"
        Additional reference: Shinozuka, M. and Deodatis, G. (1996) 
        Given a specific energy spectrum, this function generates
        1-D Gaussian field whose energy spectrum corresponds to the  
        the input energy spectrum.

        Parameters:
        -----------------------------------------------------------------
        lx: float
            the domain size in the x-direction.
        nx: integer
            the number of grid points in the x-direction
        nmodes: integer
            Number of modes
        wn1: float
            Smallest wavenumber. Typically dictated by spectrum or domain
        -----------------------------------------------------------------
        
        EXAMPLE:
        # define spectrum
        
        class k41:
        def evaluate(self, k):
            espec = pow(k,-5.0/3.0)
            return espec

        # user input

        nx = 64
        lx = 1
        nmodes = 100
        inputspec = 'k41'
        whichspect = k41().evaluate
        wn1 = 2.0*np.pi/lx

        rx = tg.gaussian1D(lx, nx, nmodes, wn1, whichspect)

        dx = lx/nx
        X = np.arange(0, lx, dx)
        plt.plot(X,rx, 'k-', label='computed')   

        """
        # -----------------------------------------------------------------

        # cell size in X-direction
        dx = lx/nx
        # Compute the highest wavenumber (wavenumber cutoff)
        wnn = np.pi/dx
        print("This function will generate data up to wavenumber: ", wnn)
        # compute the infinitesiaml wavenumber (step dk)
        dk = (wnn-wn1)/nmodes
        # compute an array of equal-distance wavenumbers at the cells centers
        wn = wn1 + 0.5*dk +  np.arange(0,nmodes)*dk
        dkn = np.ones(nmodes)*dk
        # Calculating the proportional factor (using the input power spectrum)
        espec = self.k_func(wn)
        espec = espec.clip(0.0)
        A_m = np.sqrt(2.0*espec*dkn) # for each mode I need a proportional factor ('colouring' the spectrum)
        # Generate Random phase angles from a normal distribution between 0 and 2pi
        phi = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
        psi = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
        kx = wn
        # computing the center position of the cell
        self.xc = dx/2.0 + np.arange(0,nx)*dx
        _r = np.zeros(nx)
        print("Generating 1-D turbulence...")
        for i in range(0,nx):
            # for every step i along x-direction do the fourier summation
            arg1 = kx*self.xc[i] + phi
            bmx = A_m * np.sqrt(2.0) *(np.cos(arg1))
            _r[i] = np.sum(bmx)
        print("Done! 1-D Turbulence has been generated!")
        
        self.ne = _r
        return _r
    
    def fft(self, N, d = 1):
        """A FFT based generator for scalar gaussian fields in 1D
        Reference:Timmer, J and König, M. “On Generating Power Law Noise.” Astronomy & Astrophysics 300 (1995):
        1–30. https://doi.org/10.1017/CBO9781107415324.004.
        Arguments:
            L_drive {float} -- Driving length scale
            N {int}  -- size of domain will be (2*N+1)
            k_func {function} -- a function which takes an input k 
        Returns:
            signal {1D array of floats} -- a realisation of a 1D Gaussian process.
        Example:
            L_drive = 1e-2
            def power_spectrum(k,a):
                return k**-a

            def k41(k):
                return power_spectrum(k, 5/3)        
            
            sig = gaussian1D_FFT( N, k41)
            
            fig,ax = plt.subplots()
            x = np.linspace(-N,N, 2*N+1)
            ax.plot(x, sig)
        """
        M=2*N+1
        k=np.fft.fftfreq(M, d) #these are the frequencies, starting from 0 up to f_max, then -f_max to 0.

        K=np.sqrt(k**2)
        K=np.fft.fftshift(K)#numpy convention, highest frequencies at the centre

        Wr=np.random.randn(M) # random number from Gaussian for both 
        Wi=np.random.randn(M) # real and imaginary components

        Wr = Wr + np.flip(Wr) #f(-k)=f*(k)
        Wi = Wi - np.flip(Wi)

        W = Wr+1j*Wi

        F = W*np.sqrt(self.k_func(K)) # power spectra follows power law, so sqrt here.

        F_shift=np.fft.ifftshift(F)

        F_shift[0]=0 # 0 mean

        signal=np.fft.ifftn(F_shift)
        
        self.ne = signal.real

        return self.ne
    

    def export_scalar_field(self, property: str = 'ne', fname: str = None):

        '''
        Export the current scalar electron density profile as a pvti file format, property added for future scalability to export temperature, B-field, etc.

        Args:
            property: str, 'ne': export the electron density (default)
            
            fname: str, file path and name to save under. A VTI pointed to by a PVTI file are saved in this location. If left blank, the name will default to:

                    ./plasma_PVTI_DD_MM_YYYY_HR_MIN
        
        
        '''
    
        if fname is None:
            import datetime as dt
            year = dt.datetime.now().year
            month = dt.datetime.now().month
            day = dt.datetime.now().day
            min = dt.datetime.now().minute
            hour = dt.datetime.now().hour


            fname = f'./plasma_PVTI_{day}_{month}_{year}_{hour}_{min}' #default fname to the current date and time 


        if property == 'ne':

            try: #check to ensure electron density has been added
                np.shape(self.ne)
                rnec = self.ne
            except:
                raise Exception('No electron density currently loaded!')


            if self.xc is None:
                extent = (np.shape(self.ne)[0])//2 + 1/2
                xc = np.arange(-extent,extent, 1)
            else:
                xc = self.xc


            #scaling

            # Add the data values to the cell data

        x_y_data = np.column_stack((xc,self.ne))

        print(x_y_data)
        with open(f'{fname}.pvti', 'w') as file:
            np.save(f'{fname}.npy', x_y_data)

        print(f'Scalar Domain electron density succesfully saved under {fname}.npy !')
