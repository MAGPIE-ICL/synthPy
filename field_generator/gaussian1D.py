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
        # define self.xc now to check whether cos or fft was used in generation
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

    def fft(self, l_max, l_min, extent, res):
    #def domain_fft(self, l_max, l_min, extent, res):, alternate name in case previous no longer works past merge
        '''
        A FFT based generator for scalar gaussian fields in 1D
        Reference:Timmer, J and König, M. “On Generating Power Law Noise.” Astronomy & Astrophysics 300 (1995):
        1–30. https://doi.org/10.1017/CBO9781107415324.004.

        Generate a Gaussian random field with a fourier spectrum following k_func in the domain 2*pi/l_max to 2*pi/l_min, and 0 outside
    
        Args:
            l_max: max length scale, usually = 2*extent due to physical boundary conditions
            l_min: min length scale, either resolution, or scale at which energy in = energy out (Re = 1)
            extent: field is made about the origin, from +extent to -extent in each dimension
            res: resolution, number of cells from 0 to extent, (total number of cells = 2*res*N_dim)
        
        Returns:
            x: spatial coordinates
            field: array of GRF noise of length 2*res
        '''

        dx = extent / res
        x = np.linspace(-extent, extent, 2*res, endpoint=False)

        self.xc = x

        k = 2 * np.pi * np.fft.fftfreq(2*res, d=dx)

        k_min = 2 * np.pi / l_max
        k_max = 2 * np.pi / l_min

        # Create the power spectrum
        S = np.zeros_like(k)
        mask = (k >= k_min) & (k <= k_max)
        S[mask] = self.k_func(k[mask])

        # Generate complex Gaussian noise
        noise = np.random.normal(0, 1, k.shape) + 1j * np.random.normal(0, 1, k.shape)

        # Apply the power spectrum
        fft_field = noise * np.sqrt(S)

        # Inverse Fourier transform 
        field = np.fft.ifft(fft_field).real

        field = (field) / (np.abs(field).max())

        self.ne = field

        return x, field
    
    def export_scalar_field(self, property: str = 'ne', fname: str = None):

        '''
        Export the current scalar electron density profile as a npy file format, 'property' added for future scalability to export temperature, B-field, etc.

        Args:
            property: str, 'ne': export the electron density (default)
            
            fname: str, file path and name to save under. A .npy file will be saved. If left blank, the name will default to:

                    ./plasma_PVTI_DD_MM_YYYY_HR_MIN
        
        returns:
            text file : x_values = [:, 0], field_values = [:,1]
        
            
        load in file with e.g: 

            import numpy

            ne = numpy.loadtxt(fname, dtype = float)
        
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
            except:
                raise Exception('No electron density currently loaded!')


            if self.xc is None:
                extent = (np.shape(self.ne)[0])//2
                xc = np.arange(-extent,extent + 1, 1)
            else:
                xc = self.xc


            #scaling

            # Add the data values to the cell data

            x_y_data = np.column_stack((xc,self.ne))
            x_y_data = np.column_stack((xc,self.ne))

        else:
            x_y_data = None
        else:
            x_y_data = None
        # write the file to fname

        with open(f'{fname}.txt', 'w') as file:
            np.savetxt(f'{fname}.txt', x_y_data)

        print(f'Scalar Domain electron density succesfully saved under {fname}.txt !')