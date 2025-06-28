"""
Author: Stefano Merlini, Louis Evans
Created: 14/05/2020
"""

import numpy as np

class gaussian3D:
    def __init__(self, k_func):
        """
            Parameters:
                k_func {function} -- a function which takes an input k 
        """
        # define self.xc now to check whether cos or fft was used in generation
        self.xc = None
        
        self.k_func = k_func

    def cos(self, lx, ly, lz, nx, ny, nz, nmodes, wn1):
        """
        This method is from reference: 1988, Yamasaki, "Digital Generation of Non-Goussian Stochastic Fields"
        Additional reference: Shinozuka, M. and Deodatis, G. (1996) 
        Given a specific energy spectrum, this function generates
        3-D Gaussian field whose energy spectrum corresponds to the  
        the input energy spectrum.

        Parameters:
        ----------------------------------------------------------------
        lx: float
            the domain size in the x-direction.
        ly: float
            the domain size in the y-direction.
        lz: float
            the domain size in the z-direction.
        nx: integer
            the number of grid points in the x-direction
        ny: integer
            the number of grid points in the y-direction
        nz: integer
            the number of grid points in the z-direction
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
        nz = 64
        lx = 1
        ly = 1
        lz = 1

        nmodes = 100
        inputspec = 'k41'
        whichspect = k41().evaluate
        wn1 = min(2.0*np.pi/lx, 2.0*np.pi/ly, 2.0*np.pi/lz)

        r = tg.gaussian1D(lx, ly, lz, nx, ny, nz, nmodes, wn1, whichspect)
        
        dx = lx/nx
        dy = ly/ny
        dz = lz/nz
        X = np.arange(0, lx, dx)
        Y = np.arange(0, ly, dy)
        Z = np.arange(0, lz, dz)

        X, Y = np.meshgrid(np.arange(0,lx,dx), np.arange(0,ly,dy))
        cp = plt.contourf(X, Y, r(:,:,1))
        cb = plt.colorbar(cp)

        # I you want to calculate the spectrum

        knyquist, wavenumbers, tkespec = calcspec.compute3Dspectum(r, lx, ly, lz, False)

        """
        # --------------------------------------------------------------------------

        # cell size in X, Y, Z directions
        dx = lx/nx
        dy = ly/ny
        dz = lz/nz

        # Compute the highest wavenumber (wavenumber cutoff)
        wnn = max(np.pi/dx,np.pi/dy,np.pi/dz)
        print("This function will generate data up to wavenumber: ", wnn)
        # compute the infinitesiaml wavenumber (step dk)
        dk = (wnn - wn1)/nmodes
        # compute an array of equal-distance wavenumbers at the cells centers
        wn = wn1 + 0.5*dk +  np.arange(0,nmodes)*dk
        dkn = np.ones(nmodes)*dk
        # Calculating the proportional factor (using the input power spectrum)
        espec = self.k_func(wn)
        espec = espec.clip(0.0)
        A_m = np.sqrt(2.0*espec*(dkn)**3) # for each mode I need a proportional factor ('colouring' the spectrum)
        # Generate Random phase angles from a normal distribution between 0 and 2pi
        
        psi_1 = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
        psi_2 = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
        psi_3 = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
        psi_4 = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)

        theta = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
        phi = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)

        #
        kx = np.sin(theta) * np.cos(phi) * wn
        ky = np.sin(theta) * np.sin(phi) * wn
        kz = np.cos(theta) * wn

        # Looping through 3-Dimensions nx, ny, nz and perfom the Fourier Summation

        # computing the center position of the cell
        self.xc = dx/2.0 + np.arange(0,nx)*dx
        self.yc = dy/2.0 + np.arange(0,ny)*dy
        self.zc = dz/2.0 + np.arange(0,nz)*dz

        _r = np.zeros((nx,ny,nz))

        print("Generating 3-D turbulence...")

        for k in range(0,nz):
            for j in range(0,ny):
                for i in range(0,nx):
                    # for every step i along x-y-z direction do the fourier summation
                    arg1 = kx*self.xc[i] + ky*self.yc[j] + kz*self.zc[k] + psi_1
                    arg2 = kx*self.xc[i] + ky*self.yc[j] - kz*self.zc[k] + psi_2
                    arg3 = kx*self.xc[i] - ky*self.yc[j] + kz*self.zc[k] + psi_3
                    arg4 = kx*self.xc[i] - ky*self.yc[j] - kz*self.zc[k] + psi_4
                    bm = A_m * np.sqrt(2.0) * (np.cos(arg1) + np.cos(arg2) + np.cos(arg3) + np.cos(arg4))
                    _r[i,j,k] = np.sum(bm)

        print("Done! 3-D Turbulence has been generated!")

        self.ne = _r

        return _r

    def fft(self, l_max, l_min, extent, res, factor):
        """
        A FFT based generator for scalar gaussian fields in 1D

        Reference: Timmer, J and König, M. “On Generating Power Law Noise.” Astronomy & Astrophysics 300 (1995):
        1–30. https://doi.org/10.1017/CBO9781107415324.004.

        Arguments:
            N {int}  -- size of domain will be (2*N+1)^3
            k_func {function} -- a function which takes an input k
 
        Returns:
            signal {3D array of floats} -- a realisation of a 3D Gaussian process.

        Example:
            N = 100
            def power_spectrum(k,a):
                return k**-a

            def k41(k):
                return power_spectrum(k, 5/3)        
            
            sig = gaussian3D_FFT(N, k41)
            
            fig,ax=plt.subplots(3,3, figsize=(8,8), sharex=True, sharey=True)
            ax=ax.flatten()
            
            for a in ax:
                r=np.random.randint(0,ny)
                d=sig[r,:,:]
                a.imshow(d, cmap='bwr', extent=[-N,N,-N,N])
                a.set_title("y="+str(r))
        """

        M = 2*N+1
        k = np.fft.fftfreq(M) #these are the frequencies, starting from 0 up to f_max, then -f_max to 0.

        KX,KY,KZ = np.meshgrid(k,k,k)
        K = np.sqrt(KX**2+KY**2+KZ**2)
        K = np.fft.fftshift(K)#numpy convention, highest frequencies at the centre

        Wr = np.random.randn(M, M, M) # random number from Gaussian for both 
        Wi = np.random.randn(M, M, M) # real and imaginary components

        Wr = Wr + np.flip(Wr) #f(-k)=f*(k)
        Wi = Wi - np.flip(Wi)

        W = Wr+1j*Wi

        F = W*np.sqrt(self.k_func(K)) # power spectra follows power law, so sqrt here.

        F_shift = np.fft.ifftshift(F)

        F_shift[0,0,0] = 0 # 0 mean

        signal = np.fft.ifftn(F_shift)

        self.ne = signal.real

        return self.ne

    # new function from louis branch - need to check how it works
    def domain_fft(self, l_max, l_min, extent, res, factor):
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
            z: spatial coordinates
            field: 2*res x 2*res x 2*res array of GRF noise
        '''

        dx = extent / res
        x = y =  np.linspace(-extent, extent, 2*res, endpoint=False, dtype=np.float32)
        z = np.linspace(-extent*factor, extent*factor, int(2*res*factor), endpoint=False, dtype=np.float32)
        self.xc, self.yc, self.zc = x, y, z

        kx = ky = 2 * np.pi * np.fft.fftfreq(2*res, d=dx )
        kz = 2 * np.pi * np.fft.fftfreq(int(2*res * factor), d=dx)
        kxx, kyy, kzz = np.meshgrid(kx, ky, kz, copy = False)

        del kx
        del ky
        del kz

        k = np.sqrt(kxx**2 + kyy**2 + kzz**2, dtype = np.float32)

        del kxx
        del kyy
        del kzz

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
        field = np.fft.ifftn(fft_field).real
        field = (field) / (np.abs(field).max())

        self.ne = field

        return field

    def export_scalar_field(self, property: str = 'ne', fname: str = None):
        '''
        Export the current scalar electron density profile as a pvti file format, property added for future scalability to export temperature, B-field, etc.

        Args:
            property: str, 'ne': export the electron density (default)
            
            fname: str, file path and name to save under. A VTI pointed to by a PVTI file are saved in this location. If left blank, the name will default to:

                    ./plasma_PVTI_DD_MM_YYYY_HR_MIN
        '''
        import pyvista as pv

    
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
        
            # Create the spatial reference  
            grid = pv.ImageData()

            # Set the grid dimensions: shape + 1 because we want to inject our values on
            # the CELL data
            grid.dimensions = np.array(rnec.shape) + 1
            # Edit the spatial reference
            grid.origin = (0, 0, 0)  # The bottom left corner of the data set

            if self.xc is None:
                extent = (np.shape(self.ne)[0])//2
                xc,yc,zc = np.arange(-extent,extent, 1), np.arange(-extent,extent, 1), np.arange(-extent,extent, 1)
            else:
                xc = self.xc
                yc = self.yc
                zc = self.zc

            #scaling
            x_size = np.max(xc) / ((np.shape(self.ne)[0] - 1)//2 )  #assuming centering about the origin
            y_size = np.max(yc) / ((np.shape(self.ne)[1] - 1)//2 ) 
            z_size = np.max(zc) / ((np.shape(self.ne)[2] - 1)//2 )
            grid.spacing = (x_size, y_size, z_size)  # These are the cell sizes along each axis

            # Add the data values to the cell data
            grid.cell_data["rnec"] = rnec.flatten(order="F")  # Flatten the array

            grid.save(f'{fname}.vti')

            print(f'VTI saved under {fname}.vti')

        #prep values to write the pvti, written to match the exported vti using pyvista

        relative_fname = fname.split('/')[-1]
        spacing_x = (2*xc.max())/np.shape(xc)[0]
        spacing_y = (2*yc.max())/np.shape(yc)[0]
        spacing_z = (2*zc.max())/np.shape(zc)[0]
        content = f'''<?xml version="1.0"?>
        <VTKFile type="PImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
            <PImageData WholeExtent="0 {np.shape(self.ne)[0]} 0 {np.shape(self.ne)[1]} 0 {np.shape(self.ne)[2]}" GhostLevel="0" Origin="0 0 0" Spacing="{x_size} {y_size} {z_size}">
                <PCellData Scalars="rnec">
                    <PDataArray type="Float64" Name="rnec">
                    </PDataArray>
                </PCellData>
                <Piece Extent="0 {np.shape(self.ne)[0]} 0 {np.shape(self.ne)[1]} 0 {np.shape(self.ne)[2]}" Source="{relative_fname}.vti"/>
            </PImageData>
        </VTKFile>'''
        # write file
        with open(f'{fname}.pvti', 'w') as file:
            file.write(content)
        
        print(f'Scalar Domain electron density succesfully saved under {fname}.pvti !')