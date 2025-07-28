import numpy as np
import jax.numpy as jnp
import jax
import equinox as eqx

from utils import mem_conversion
from utils import colour
from utils import dalloc
from utils import domain_estimate

class ScalarDomain(eqx.Module):
    """
    A class to hold and generate scalar domains.
    This contains also the method to propagate rays through the scalar domain
    """

    inv_brems: bool
    phaseshift: bool
    B_on: bool

    probing_direction: str

    ne_type: str

    x_length: jnp.int64
    y_length: jnp.int64
    z_length: jnp.int64

    lengths: jax.Array

    x_n: jnp.int64
    y_n: jnp.int64
    z_n: jnp.int64

    dim: jax.Array

    x: jax.Array
    y: jax.Array
    z: jax.Array

    max_dim: jnp.int64

    coordinates: jax.Array

    XX: jax.Array
    YY: jax.Array
    ZZ: jax.Array

    ne: jax.Array

    B: jax.Array
    Te: jax.Array
    Z: jax.Array

    region_count: jnp.int64
    lengths_backup: jnp.int64
    dims_backup: jnp.int64

    length_ungenerated: jnp.float64
    indices_unused: jnp.int64

    def __init__(self, lengths, dim, *, ne_type = None, inv_brems = False, phaseshift = False, B_on = False, probing_direction = 'z', auto_batching = True):
        """
        Example:
            N_V = 100
            M_V = 2*N_V+1
            ne_extent = 5.0e-3
            ne_x = jnp.linspace(-ne_extent,ne_extent,M_V)
            ne_y = jnp.linspace(-ne_extent,ne_extent,M_V)
            ne_z = jnp.linspace(-ne_extent,ne_extent,M_V)

        Args:
            x (float array): x coordinates, m
            y (float array): y coordinates, m
            z (float array): z coordinates, m
            extent (float): physical size, m
        """

        # initalise to none for equinox incase not initialised properly later on
        self.ne = None
        self.B = None
        self.Te = None
        self.Z = None

        # Logical switches
        self.inv_brems = inv_brems
        self.phaseshift = phaseshift
        self.B_on = B_on

        self.probing_direction = probing_direction

        self.ne_type = ne_type

        valid_types = (int, float, jnp.int64)

        ##
        ## NOT FORCING THESE CONVERSIONS MAY CAUSE ISSUES WITH EQUINOX CLASS LATER DOWN THE LINE DEPENDING ON USER INPUT
        ##

        # if 1 length given, assumes all are the same
        if isinstance(lengths, valid_types):
            self.x_length, self.y_length, self.z_length = lengths, lengths, lengths
            self.lengths = jnp.array([lengths, lengths, lengths])
        # if array given, checks len = 3 and assigns accordingly
        else:
            if len(lengths) != 3:
                raise Exception('lengths must have len = 3: (x,y,z)')

            self.x_length, self.y_length, self.z_length = lengths[0], lengths[1], lengths[2]
            self.lengths = jnp.array(lengths)

        del lengths
        
        #likewise for dim
        #self.dims = dim
        if isinstance(dim, valid_types):
            self.x_n, self.y_n, self.z_n = dim, dim, dim
            self.dims = jnp.array([dim, dim, dim])
        else:
            if len(dim) != 3:
                raise Exception('n must have len = 3: (x_n, y_n, z_n)')

            self.x_n, self.y_n, self.z_n = dim[0], dim[1], dim[2]
            self.dims = jnp.array(dim)

        del dim
        del valid_types

        predicted_domain_allocation = domain_estimate(self.dims)
        print("Predicted size in memory of domain:", mem_conversion(predicted_domain_allocation))

        # define coordinate space
        self.x = jnp.float32(jnp.linspace(-self.x_length / 2, self.x_length / 2, self.x_n))
        self.y = jnp.float32(jnp.linspace(-self.y_length / 2, self.y_length / 2, self.y_n))
        self.z = jnp.float32(jnp.linspace(-self.z_length / 2, self.z_length / 2, self.z_n))

        self.region_count = 1
        self.lengths_backup = 0.0
        self.dims_backup = 0.0

        if auto_batching:
            from jax.lib import xla_bridge
            running_device = xla_bridge.get_backend().platform

            if running_device == 'cpu':
                from psutil import virtual_memory

                free_mem = virtual_memory().available

                print("\nFree memory:", mem_conversion(free_mem))
            elif running_device == 'gpu':
                from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

                nvmlInit()

                h = nvmlDeviceGetHandleByIndex(0)
                info = nvmlDeviceGetMemoryInfo(h)

                free_mem = info.free

                print("\nMemory prior to domain creation:")
                print(f'total : {mem_conversion(info.total)}')
                print(f'free  : {mem_conversion(info.free)}')
                print(f'used  : {mem_conversion(info.used)}')
            elif running_device == 'tpu':
                free_mem = None
            else:
                assert "\nNo suitable device detected when checking ram/vram available."

            ##
            ## Need to work out the max allocation at any point and that estimated size
            ##

            # 2 for ne and ne_nc in calc_dndr(...) before ne is deleted
            # at peak mem usage ne should have been deleted, therefore this contributes only 1 domain
            # +1 for ne_interp
            # +2 for the 2 sqeuentially repeated domain sized allocations in dndr(...)
            allocation_count = 4

            # up to +5 in calc_dndr(...) depending on the number of extra interps
            if B_on:
                # there are 4 B based interps
                # and they also require a ScalarDomain.B domain sized matrice
                allocation_count += 4
            if inv_brems:
                # unsure how many intermediaries exist at peak mem usage for this allocation - need to check and adjust this
                allocation_count += 1
            if phaseshift:
                allocation_count += 1

            # working with 10% leeway in estimate for now
            estimate_limit = predicted_domain_allocation * allocation_count * 300 #1.1
            print(mem_conversion(estimate_limit))
            if estimate_limit > free_mem:
                print(colour.BOLD + "\nESTMATE SUGGESTS DOMAIN CANNOT FIT IN AVAILABLE MEMORY." + colour.END)
                print("--> Auto-batching domain based on memory available and domain size estimate...")

                ##
                ## Used backed up information to re-assign to ScalarDomain in propagator
                ## Then call generate_electron_density_profile(...) and re-do calculations with end of prior domain
                ##

                from math import ceil
                self.region_count = ceil(estimate_limit / free_mem)

                # save the intended length and resolution of batching axes for later
                #self.lengths_backup = self.lengths[['x', 'y', 'z'].index(self.probing_direction)]
                #self.dims_backup = self.dims[['x', 'y', 'z'].index(self.probing_direction)]

                print("--> Batching calculation completed. Domain will be split into " + str(self.region_count) + " parts.")
                print(colour.BOLD + "\nWARNING:" + colour.END + " This functionality will cause the solver to run slower due to domain regeneration.")
                print("For optimal performance, increase the memory available to this program.")

        self.max_dim = self.dims[0]
        for dimension in self.dims:
            if dimension > self.max_dim:
                self.max_dim = dimension

        # pad coordinates but not arrays themselves, that way only interpolator takes in padded values - no needless extra mem allocation by the domain
        self.coordinates = jnp.stack([
                jnp.pad(self.x, (0, self.max_dim - self.x_n), constant_values = jnp.nan),
                jnp.pad(self.y, (0, self.max_dim - self.y_n), constant_values = jnp.nan),
                jnp.pad(self.z, (0, self.max_dim - self.z_n), constant_values = jnp.nan)
            ], axis = 1)

        if self.region_count == 1:
            if self.ne_type is not None:
                self.generate_electron_density_profile()
            else:
                print("\nWARNING: Electron density profile to generate not passed. You will need to initialise this yourself with a call to this library.")
                print("If you run low on memory, you can enforce a manual domain cleanup with a call to ScalarDomain.cleanup()")
                self.XX, self.YY, self.ZZ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)#False) - has to be true for jnp

        print("")
        jax.print_environment_info()

    def generate_electron_density_profile(self):
        if self.ne_type == "test_null":
            print("\nGenerating test null -e field.")
            self.XX, _, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.YY = None
            self.ZZ = None

            result = self.test_null()
        elif self.ne_type == "test_slab":
            print("\nGenerating test slab -e field.")
            self.XX, _, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.YY = None
            self.ZZ = None

            result = self.test_slab()
        elif self.ne_type == "test_linear_cos":
            print("\nGenerating test linear decay periodic -e field.")
            self.XX, self.YY, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.ZZ = None

            result = self.test_linear_cos()
        elif self.ne_type == "test_exponential_cos":
            print("\nGenerating test exponential decay periodic -e field.")
            self.XX, self.YY, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.ZZ = None

            result = self.test_exponential_cos()
        else:
            assert "\nNo valid profile detected! Ensure passed name is correct or call yourself."

        self.cleanup()
        return result

    def setup_next_domain(self, i):
        #...# Batching logic
        length = self.lengths[['x', 'y', 'z'].index(self.probing_direction)] // self.region_count
        dim = self.dims[['x', 'y', 'z'].index(self.probing_direction)] // self.region_count

        length_ungenerated = self.lengths[['x', 'y', 'z'].index(self.probing_direction)] - length * (i - 1)
        indices_unused = self.dims[['x', 'y', 'z'].index(self.probing_direction)] - dim * (i - 1)
        if length_ungenerated < length or indices_unused < dim:
            length = length_ungenerated
            dim = indices_unused

        coord_line = jnp.float32(jnp.linspace(-length / 2, self.length / 2, self.dim))

        # setup proper indexing so that we can set lengths and dim available based on which region we need at the time - WITHOUT MISSING ANY POINTS

        if self.probing_direction == 'x':
            return (
                    jnp.array([length, self.lengths[1], self.lengths[2]]),
                    jnp.array([dim, self.dims[1], self.dims[2]]),
                    jnp.array([coord_line, self.y, self.z])                
                )
        elif self.probing_direction == 'y':
            return (
                    jnp.array([self.lengths[0], length, self.lengths[2]]),
                    jnp.array([self.dims[0], dim, self.dims[2]]),
                    jnp.array([self.x, coord_line, self.z])                
                )
        elif self.probing_direction == 'z':
            return (
                    jnp.array([self.lengths[0], self.lengths[1], length]),
                    jnp.array([self.dims[0], self.dims[1], dim]),
                    jnp.array([self.x, self.y, coord_line])                
                )
        else:
            assert colour.BOLD + "Invalid entry for probing_direction!" + colour.END

    def generate_next_domain(self, i):
        self.setup_next_domain(i)
        if self.ne_type is not None:
            self.generate_electron_density_profile()
        else:
            assert colour.BOLD + "\nne_type must be passed to domain creation in order to utilise auto-batching." + colour.END
        # do the equivalent for these later
        if self.B is not None:
            pass
        if self.Te is not None:
            pass
        if self.Z is not None:
            pass

    def test_null(self):
        """
        Null test, an empty cube
        """

        self.ne = jnp.zeros_like(self.XX)
    
    def test_slab(self, s = 1, n_e0 = 2e23):
        """
        A slab with a linear gradient in x:
        n_e =  n_e0 * (1 + s*x/extent)

        Will cause a ray deflection in x

        Args:
            s (int, optional): scale factor. Defaults to 1.
            n_e0 ([type], optional): mean density. Defaults to 2e23 m^-3.
        """

        self.ne = n_e0 * (1.0 + s * self.XX / self.x_length)

    def test_linear_cos(self, s1 = 0.1, s2 = 0.1, n_e0 = 2e23, Ly = 1):
        """
        Linearly growing sinusoidal perturbation

        Args:
            s1 (float, optional): scale of linear growth. Defaults to 0.1.
            s2 (float, optional): amplitude of sinusoidal perturbation. Defaults to 0.1.
            n_e0 ([type], optional): mean electron density. Defaults to 2e23 m^-3.
            Ly (int, optional): spatial scale of sinusoidal perturbation. Defaults to 1.
        """

        self.ne = n_e0 * (1.0 + s1 * self.XX / self.x_length) * (1 + s2 * jnp.cos(2 * jnp.pi * self.YY / Ly))
    
    def test_exponential_cos(self, XX, YY n_e0 = 1e24, Ly = 1e-3, s = 2e-3):
        """
        Exponentially growing sinusoidal perturbation

        Args:
            n_e0 ([type], optional): mean electron density. Defaults to 1e24 m^-3.
            Ly (int, optional): spatial scale of sinusoidal perturbation. Defaults to 1e-3 m.
            s ([type], optional): scale of exponential growth. Defaults to 2e-3 m.
        """

        self.XX = self.XX.at[:, :].set(self.XX / s)
        self.XX = self.XX.at[:, :].set(10 ** self.XX)

        self.YY = self.YY.at[:, :].set(self.YY / Ly)
        self.YY = self.YY.at[:, :].set(jnp.pi * self.YY)
        self.YY = self.YY.at[:, :].set(2 * self.YY)
        self.YY = self.YY.at[:, :].set(jnp.cos(self.YY))
        self.YY = self.YY.at[:, :].set(1 + self.YY) # any difference if float or not? shouldn't be.

        # jnp.float64(), both here and on final assignment
        # should be float32 surely?
        # is it needed at all?
        self.ne = self.XX * self.YY
        self.cleanup()

        self.ne = self.ne.at[:, :].set(n_e0 * self.ne)

        #self.ne = jnp.float64(n_e0 * 10 ** (self.XX / s) * (1 + jnp.cos(2 * jnp.pi * self.YY / Ly)))

    def external_ne(self, ne):
        """
        Load externally generated grid

        Args:
            ne ([type]): MxMxM grid of density in m^-3
        """

        self.ne = ne

    def external_B(self, B):
        """
        Load externally generated grid

        Args:
            B ([type]): MxMxMx3 grid of B field in T
        """

        self.B = B

    def external_Te(self, Te, Te_min = 1.0):
        """
        Load externally generated grid

        Args:
            Te ([type]): MxMxM grid of electron temperature in eV
        """

        self.Te = jnp.maximum(Te_min, Te)

    def external_Z(self, Z):
        """
        Load externally generated grid

        Args:
            Z ([type]): MxMxM grid of ionisation
        """

        self.Z = Z
        
    def test_B(self, Bmax=1.0):
        """
        A Bz field with a linear gradient in x:
        Bz =  Bmax*x/extent

        Args:
            Bmax ([type], optional): maximum B field, default 1.0 T
        """

        self.B = jnp.zeros(jnp.append(jnp.array(self.XX.shape), 3))
        self.B[:, :, :, 2] = Bmax * self.XX / self.x_length

    def export_scalar_field(self, property: str = 'ne', fname: str = None):
        """
        Export the current scalar electron density profile as a pvti file format, property added for future scalability to export temperature, B-field, etc.

        Args:
            property: str, 'ne': export the electron density (default)
            fname: str, file path and name to save under. A VTI pointed to by a PVTI file are saved in this location. If left blank, the name will default to:
                    ./plasma_PVTI_DD_MM_YYYY_HR_MIN
        """

        import pyvista as pv

        if fname is None:
            import datetime as dt
            year = dt.datetime.now().year
            month = dt.datetime.now().month
            day = dt.datetime.now().day
            min = dt.datetime.now().minute
            hour = dt.datetime.now().hour

            # filename extended to include the name of the property to be exported
            fname = f'./plasma_PVTI_{property}_{day}_{month}_{year}_{hour}_{min}' #default fname to the current date and time 

        if property == 'ne':

            try: #check to ensure electron density has been added
                jnp.shape(self.ne)
                rnec = self.ne
            except:
                raise Exception('No electron density currently loaded!')
        
            # Create the spatial reference  
            grid = pv.ImageData()

            # Set the grid dimensions: shape + 1 because we want to inject our values on
            # the CELL data
            grid.dimensions = jnp.array(rnec.shape) + 1
            # Edit the spatial reference
            grid.origin = (0, 0, 0)  # The bottom left corner of the data set

            #scaling
            x_size = jnp.max(self.x) / ((jnp.shape(self.ne)[0] - 1)//2 )  #assuming centering about the origin
            y_size = jnp.max(self.y) / ((jnp.shape(self.ne)[1] - 1)//2 ) 
            z_size = jnp.max(self.z) / ((jnp.shape(self.ne)[2] - 1)//2 )
            grid.spacing = (x_size, y_size, z_size)  # These are the cell sizes along each axis

            # Add the data values to the cell data
            grid.cell_data["rnec"] = rnec.flatten(order="F")  # Flatten the array

            grid.save(f'{fname}.vti')

            print(f'VTI saved under {fname}.vti')

        #prep values to write the pvti, written to match the exported vti using pyvista

        relative_fname = fname.split('/')[-1]
        spacing_x = (2*jnp.max(self.x))/jnp.shape(self.x)[0]
        spacing_y = (2*jnp.max(self.y))/jnp.shape(self.y)[0]
        spacing_z = (2*jnp.max(self.z))/jnp.shape(self.z)[0]
        content = f"""<?xml version="1.0"?>
                        <VTKFile type="PImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
                            <PImageData WholeExtent="0 {jnp.shape(self.ne)[0]} 0 {jnp.shape(self.ne)[1]} 0 {jnp.shape(self.ne)[2]}" GhostLevel="0" Origin="0 0 0" Spacing="{spacing_x} {spacing_y} {spacing_z}">
                                <PCellData Scalars="rnec">
                                    <PDataArray type="Float64" Name="rnec">
                                    </PDataArray>
                                </PCellData>
                                <Piece Extent="0 {jnp.shape(self.ne)[0]} 0 {jnp.shape(self.ne)[1]} 0 {jnp.shape(self.ne)[2]}" Source="{relative_fname}.vti"/>
                            </PImageData>
                        </VTKFile>"""
    
        # write file
        with open(f'{fname}.pvti', 'w') as file:
            file.write(content)

        print(f'Scalar Domain electron density succesfully saved under {fname}.pvti !')

    def cleanup(self):
        if self.XX is not None:
            dalloc(self.XX)
        if self.YY is not None:
            dalloc(self.YY)
        if self.ZZ is not None:
            dalloc(self.ZZ)