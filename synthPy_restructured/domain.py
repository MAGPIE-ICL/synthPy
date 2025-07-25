import numpy as np

class ScalarDomain:
    """
    A class to hold and generate scalar domains.
    This contains also the method to propagate rays through the scalar domain
    """

    def __init__(self, lengths, dim, B_on = False):
        """
        Example:
            N_V = 100
            M_V = 2*N_V+1
            ne_extent = 5.0e-3
            ne_x = np.linspace(-ne_extent,ne_extent,M_V)
            ne_y = np.linspace(-ne_extent,ne_extent,M_V)
            ne_z = np.linspace(-ne_extent,ne_extent,M_V)

        Args:
            x (float array): x coordinates, m
            y (float array): y coordinates, m
            z (float array): z coordinates, m
            extent (float): physical size, m
        """

        # if 1 length given, assumes all are the same
        if isinstance(lengths, (float, int)):
            self.x_length, self.y_length, self.z_length = lengths, lengths, lengths
            self.lengths = np.array([lengths, lengths, lengths])
        # if array given, checks len = 3 and assigns accordingly
        else:
            if len(lengths) != 3:
                raise Exception('lengths must have len = 3: (x,y,z)')

            self.x_length, self.y_length, self.z_length = lengths[0], lengths[1], lengths[2]
            self.lengths = np.array(lengths)
        
        #likewise for dim
        self.dim = dim
        if isinstance(dim, (int, float)):
            self.x_n, self.y_n, self.z_n = dim, dim, dim
            self.dim = np.array([dim,dim,dim])
        else:
            if len(dim) != 3:
                raise Exception('n must have len = 3: (x_n, y_n, z_n)')

            self.x_n, self.y_n, self.z_n = dim[0], dim[1], dim[2]
            self.dim = np.array(dim)


        # define coordinate space
        self.x = np.float32(np.linspace(-self.x_length/2, self.x_length/2, self.x_n))
        self.y = np.float32(np.linspace(-self.y_length/2, self.y_length/2, self.y_n))
        self.z = np.float32(np.linspace(-self.z_length/2, self.z_length/2, self.z_n))
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x, self.y, self.z, indexing='ij', copy = False)

        # Logical switches
        self.B_on = B_on

    def test_null(self):
        """
        Null test, an empty cube
        """

        self.ne = np.zeros_like(self.XX)
    
    def josie_test(self, radius, position, n1, n2):
        """
        Simple test to compare with Josie's experiments. 
        A cyclinder (plume) of material with a refractive index n2 within a 
        box of refractive index n1. Axis of the cylinder is along the y-axis.

        Args:
            radius (float): radius of cylinder
            position (tuple): tuple of the form (x-postion, z-position), giving the x-z position of
                              the centre of the cylinder.
            n1 (float): refractive index outside the cylinder
            n2 (float): refractive index inside the cylinder
        NB: returns refractive index, not electron density
        """
        def circular_tophat(radius, position):
            posx, posz = position
            result = np.zeros_like(self.XX)
            for i in range(0,np.size(self.XX, axis=0)):
                for j in range(0,np.size(self.XX, axis=1)):
                    for k in range(0,np.size(self.XX, axis=2)):
                        distance = np.sqrt((self.XX[i,j,k]-posx)**2 + (self.ZZ[i,j,k]-posz)**2)
                        if distance < radius:
                            result[i,j,k] = 1
            return result
        

        self.refrac_field = np.full_like(self.XX, n1) + (n2-n1)*circular_tophat(radius, position)
    def test_slab(self, s=1, n_e0=2e23):
        """A slab with a linear gradient in x:
        n_e =  n_e0 * (1 + s*x/extent)

        Will cause a ray deflection in x

        Args:
            s (int, optional): scale factor. Defaults to 1.
            n_e0 ([type], optional): mean density. Defaults to 2e23 m^-3.
        """

        self.ne = n_e0*(1.0+s*self.XX/self.x_length)
    
    def test_linear_cos(self,s1=0.1,s2=0.1,n_e0=2e23,Ly=1):
        """Linearly growing sinusoidal perturbation

        Args:
            s1 (float, optional): scale of linear growth. Defaults to 0.1.
            s2 (float, optional): amplitude of sinusoidal perturbation. Defaults to 0.1.
            n_e0 ([type], optional): mean electron density. Defaults to 2e23 m^-3.
            Ly (int, optional): spatial scale of sinusoidal perturbation. Defaults to 1.
        """

        self.ne = n_e0*(1.0+s1*self.XX/self.x_length)*(1+s2*np.cos(2*np.pi*self.YY/Ly))
    
    def test_exponential_cos(self, n_e0=1e24, Ly=1e-3, s=2e-3):
        """Exponentially growing sinusoidal perturbation

        Args:
            n_e0 ([type], optional): mean electron density. Defaults to 1e24 m^-3.
            Ly (int, optional): spatial scale of sinusoidal perturbation. Defaults to 1e-3 m.
            s ([type], optional): scale of exponential growth. Defaults to 2e-3 m.
        """

        self.ne = n_e0*10**(self.XX/s)*(1+np.cos(2*np.pi*self.YY/Ly))
        
    def test_onaxis_sin(self, ne0 = 1e24, A = 0.5):
        """
        Electron density that varies sinusoidally in the z direction. the frequency of oscillation
        increases linearly in the x direction. 
        Args: 
            ne0 (float, optional): mean electron density. Defaults to 1e24 m^-3.
            A (float, optional): amplitude of oscillations.
        """
        m = np.zeros((self.x_n, self.y_n, self.z_n))
        for i in range (0, self.x_n):
            m[i, :, :] = np.full((self.y_n, self.z_n), i+1)

        self.ne = ne0*(1 + A*np.sin(2*np.pi*m*self.ZZ/self.z_length))

    def double_gaussian(self, ne0 = 1e24, sigma = 0.5e-3, d = 1e-3):
        gauss1 = np.exp(-((self.XX - d)**2 + self.YY**2) / (2 * sigma**2))
        gauss2 = np.exp(-((self.XX + d)**2 + self.YY**2) / (2 * sigma**2))
        
        self.ne = ne0 * (gauss1 + gauss2)

    def gaussian(self, ne0 = 1e24, sigma = 3e-3):
        self.ne   =  ne0 * np.exp(- (self.XX**2 + self.YY**2) / (2 * sigma**2))

    def external_ne(self, ne):
        """Load externally generated grid

        Args:
            ne ([type]): MxMxM grid of density in m^-3
        """

        self.ne = ne

    def external_B(self, B):
        """Load externally generated grid

        Args:
            B ([type]): MxMxMx3 grid of B field in T
        """

        self.B = B

    def external_Te(self, Te, Te_min = 1.0):
        """Load externally generated grid

        Args:
            Te ([type]): MxMxM grid of electron temperature in eV
        """

        self.Te = np.maximum(Te_min,Te)
        


    def external_mass_density(self, rho):
        """Load external mass density field. To be used for opacity calculations

        Args:: MxMxM grid of the mass density in kg/m^3
            rho (array)
        """
        self.rho = rho

    def external_Z(self, Z):
        """Load externally generated grid

        Args:
            Z ([type]): MxMxM grid of ionisation
        """

        self.Z = Z
        
    def test_B(self, Bmax=1.0):
        """A Bz field with a linear gradient in x:
        Bz =  Bmax*x/extent

        Args:
            Bmax ([type], optional): maximum B field, default 1.0 T
        """

        self.B          = np.zeros(np.append(np.array(self.XX.shape),3))
        self.B[:,:,:,2] = Bmax*self.XX/self.x_length

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

            #scaling
            x_size = np.max(self.x) / ((np.shape(self.ne)[0] - 1)//2 )  #assuming centering about the origin
            y_size = np.max(self.y) / ((np.shape(self.ne)[1] - 1)//2 ) 
            z_size = np.max(self.z) / ((np.shape(self.ne)[2] - 1)//2 )
            grid.spacing = (x_size, y_size, z_size)  # These are the cell sizes along each axis

            # Add the data values to the cell data
            grid.cell_data["rnec"] = rnec.flatten(order="F")  # Flatten the array

            grid.save(f'{fname}.vti')

            print(f'VTI saved under {fname}.vti')

        #prep values to write the pvti, written to match the exported vti using pyvista

        relative_fname = fname.split('/')[-1]
        spacing_x = (2*np.max(self.x))/np.shape(self.x)[0]
        spacing_y = (2*np.max(self.y))/np.shape(self.y)[0]
        spacing_z = (2*np.max(self.z))/np.shape(self.z)[0]
        content = f'''<?xml version="1.0"?>
                        <VTKFile type="PImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
                            <PImageData WholeExtent="0 {np.shape(self.ne)[0]} 0 {np.shape(self.ne)[1]} 0 {np.shape(self.ne)[2]}" GhostLevel="0" Origin="0 0 0" Spacing="{spacing_x} {spacing_y} {spacing_z}">
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