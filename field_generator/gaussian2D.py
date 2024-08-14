"""
Author: Stefano Merlini
Created: 14/05/2020
Modified: 24/06/2024
"""

import numpy as np

#  ____      ____     ___   __   _  _  ____  ____  __   __   __ _     ___  __   ____ 
# (___ \ ___(    \   / __) / _\ / )( \/ ___)/ ___)(  ) / _\ (  ( \   / __)/  \ / ___)
#  / __/(___)) D (  ( (_ \/    \) \/ (\___ \\___ \ )( /    \/    /  ( (__(  O )\___ \
# (____)    (____/   \___/\_/\_/\____/(____/(____/(__)\_/\_/\_)__)   \___)\__/ (____/

class gaussian2D:
    def __init__(self, k_func):
        """
            Parameters:
                k_func {function} -- a function which takes an input k 
        """
        # define self.xc now to check whether cos or fft was used in generation
        self.xc = None

        self.k_func = k_func

    def cos(self, lx, ly, nx, ny, nmodes, wn1):
        """
        this method is from reference: 1988, Yamasaki, "Digital Generation of Non-Goussian Stochastic Fields"
        Additional reference: Shinozuka, M. and Deodatis, G. (1996) 
        Given a specific energy spectrum, this function generates
        2-D Gaussian field whose energy spectrum corresponds to the  
        the input energy spectrum.

        Parameters:
        ----------------------------------------------------------------
        lx: float
            the domain size in the x-direction.
        ly: float
            the domain size in the y-direction.
        nx: integer
            the number of grid points in the x-direction
        ny: integer
            the number of grid points in the y-direction
        nmodes: integer
            Number of modes
        wn1: float
            Smallest wavenumber. Typically dictated by spectrum or domain
        espec: function
            A callback function representing the energy spectrum in input
        -----------------------------------------------------------------

        EXAMPLE:
        import turboGen as tg
        import calcspec

        # define spectrum
        class k41:
        def evaluate(self, k):
            espec = pow(k,-5.0/3.0)
            return espec
        
        # user input
        
        nx = 64
        ny = 64
        lx = 1
        ly = 1
        nmodes = 100
        inputspec = 'k41'
        whichspect = k41().evaluate
        wn1 = min(2.0*np.pi/lx, 2.0*np.pi/ly)

        r = tg.gaussian1D(lx, ly, nx, ny, nmodes, wn1, whichspect)
        
        dx = lx/nx
        dy = ly/ny
        X = np.arange(0, lx, dx)
        Y = np.arange(0, ly, dy)
        X, Y = np.meshgrid(np.arange(0,lx,dx), np.arange(0,ly,dy))
        cp = plt.contourf(X, Y, r)
        cb = plt.colorbar(cp)

        # I you want to calculate the spectrum

        knyquist, wavenumbers, tkespec = calcspec.compute2Dspectum(r, lx, ly, False)

        """
        # --------------------------------------------------------------------------

        # cell size in X and Y directions
        dx = lx/nx
        dy = ly/ny
        # Compute the highest wavenumber (wavenumber cutoff)
        wnn = max(np.pi/dx,np.pi/dy)
        print("This function will generate data up to wavenumber: ", wnn)
        # compute the infinitesiaml wavenumber (step dk)
        dk = (wnn - wn1)/nmodes
        # compute an array of equal-distance wavenumbers at the cells centers
        wn = wn1 + 0.5*dk +  np.arange(0,nmodes)*dk
        dkn = np.ones(nmodes)*dk
        # Calculating the proportional factor (using the input power spectrum)
        espec = self.k_func(wn)
        espec = espec.clip(0.0)
        A_m = np.sqrt(2.0*espec*(dkn)**2) # for each mode I need a proportional factor ('colouring' the spectrum)
        # Generate Random phase angles from a normal distribution between 0 and 2pi
        phi = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
        psi = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
        theta = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
        #
        kx = np.cos(theta)*wn
        ky = np.sin(theta)*wn

        # perfom the Fourier Summation

        # computing the center position of the cell
        self.xc = dx/2.0 + np.arange(0,nx)*dx
        self.yc = dy/2.0 + np.arange(0,ny)*dy

        _r = np.zeros((nx,ny))

        print("Generating 2-D turbulence...")
        for j in range(0,ny):
            for i in range(0,nx):
                # for every step i along x-y direction do the fourier summation
                arg1 = kx*self.xc[i] + ky*self.yc[j] + phi
                arg2 = kx*self.xc[i] - ky*self.yc[j] + psi
                bm = A_m * np.sqrt(2.0) *(np.cos(arg1) + np.cos(arg2))
                _r[i,j] = np.sum(bm)
        print("Done! 2-D Turbulence has been generated!")

        _r = self.ne

        return _r
    
    def fft(self, l_max, l_min, extent, res):
        '''
        Generate a Gaussian random field with a fourier spectrum following k_func in the domain 2*pi/l_max to 2*pi/l_min, and 0 outside
        Reference:Timmer, J and König, M. “On Generating Power Law Noise.” Astronomy & Astrophysics 300 (1995):
        1–30. https://doi.org/10.1017/CBO9781107415324.004.

    
        Args:
            l_max: max length scale, usually = 2*extent due to physical boundary conditions
            l_min: min length scale, either resolution, or scale at which energy in = energy out (Re = 1)
            extent: field is made about the origin, from +extent to -extent in each dimension
            res: resolution, number of cells from 0 to extent, (total number of cells = 2*res*N_dim)
        
        Returns:
            x: spatial coordinates
            y: spatial coordinates
            field: 2*res x 2*res array of GRF noise
        '''

        dx = extent / res
        x = y = np.linspace(-extent, extent, 2*res, endpoint=False)
        xx, yy = np.meshgrid(x, y)
        self.xc = x
        self.yc = y

        kx = ky = 2 * np.pi * np.fft.fftfreq(2*res, d=dx)
        kxx, kyy = np.meshgrid(kx, ky)
        k = np.sqrt(kxx**2 + kyy**2)

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
        field = np.fft.ifft2(fft_field).real

        field = (field) / (np.abs(field).max())

        self.ne = field

        return xx, yy, field
    
    def domain_fft(self, l_max, l_min, extent, res):
        '''
        Generate a Gaussian random field with a fourier spectrum following k_func in the domain 2*pi/l_max to 2*pi/l_min, and 0 outside

    
        Args:
            l_max: max length scale, usually = 2*extent due to physical boundary conditions
            l_min: min length scale, either resolution, or scale at which energy in = energy out (Re = 1)
            extent: field is made about the origin, from +extent to -extent in each dimension
            res: resolution, number of cells from 0 to extent, (total number of cells = 2*res*N_dim)
        
        Returns:
            x: spatial coordinates
            y: spatial coordinates
            field: 2*res x 2*res array of GRF noise
        '''

        dx = extent / res
        x = y = np.linspace(-extent, extent, 2*res, endpoint=False)
        xx, yy = np.meshgrid(x, y)
        self.xc = x
        self.yc = y

        kx = ky = 2 * np.pi * np.fft.fftfreq(2*res, d=dx)
        kxx, kyy = np.meshgrid(kx, ky)
        k = np.sqrt(kxx**2 + kyy**2)

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
        field = np.fft.ifft2(fft_field).real

        self.ne = field

        return xx, yy, field
    

    def export_scalar_field(self, property: str = 'ne', fname: str = None):

        '''
        Export the current scalar electron density profile as a pvti file format, property added for future scalability to export temperature, B-field, etc.

        Args:
            property: str, 'ne': export the electron density (default)
            
            fname: str, file path and name to save under. A VTI pointed to by a PVTI file are saved in this location. If left blank, the name will default to:

                    ./plasma_PVTI_DD_MM_YYYY_HR_MIN
        
        Returns:

            pickle file : x_values = [:, 0], y_values = [:,1], field_values = [:,2]
        
        load in file with e.g: 

            import pickle

            ne = pickle.load( open(fname.pkl, "rb" ) )
        
        
        '''
        import pyvista as pv

        import pickle

    
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

            #get scale information 

            if self.xc is None:
                extent = (np.shape(self.ne)[0])//2
                print(extent)
                xc, yc = np.arange(-extent,extent + 1, 1), np.arange(-extent,extent + 1, 1)
            else:
                xc = self.xc
                yc = self.yc
            
            spatial_vals = np.column_stack((xc, yc))

            values = np.concatenate((spatial_vals, rnec), axis = 1)


        filehandler = open(f'{fname}.pkl',"wb")
        pickle.dump(values,filehandler)
        filehandler.close()
        
        