import jax
import jax.numpy as jnp
import numpy as np

import equinox as eqx

#from functools import partial

from math import ceil
from math import floor

from shared.utils import mem_conversion
from shared.printing import colour
from shared.utils import dalloc
from shared.utils import domain_estimate
from shared.utils import memory_report
from shared.utils import getsizeof_default

# decorating a function like this will initiate manual rematerialisation - could reduce memory usage for large array calculations??
# test? also test in propagator? where?
'''
from jax import checkpoint

@checkpoint
'''

class ScalarDomain(eqx.Module):
    s: jnp.float32
    s1: jnp.float32
    s2: jnp.float32

    Ly: jnp.float32

    ne_0: jnp.float32

    Bmax: jnp.float32

    Te_min: jnp.float32

    inv_brems: bool
    phaseshift: bool
    opacity: bool
    B_on: bool
    edensity: bool

    probing_direction: str

    ne_type: str

    leeway_factor: jnp.float32

    x_length: jnp.int32
    y_length: jnp.int32
    z_length: jnp.int32

    lengths: jax.Array

    x_n: jnp.int32
    y_n: jnp.int32
    z_n: jnp.int32

    dims: jax.Array

    x: jax.Array
    y: jax.Array
    z: jax.Array

    coordinates: jax.Array

    XX: jax.Array
    YY: jax.Array
    ZZ: jax.Array

    ne: jax.Array

    B: jax.Array
    Te: np.array
    Z: jax.Array
  
    region_count: jnp.int32

    coord_backup: jax.Array
    future_dims: jax.Array

    extra_info: bool
    memory_reporting: bool

    memory_limit: np.int64

    Np_total: np.int64
    ray_batch_count: np.int64

    opacity_files: list
    densities: list
    num_materials: jnp.int32

    refrac_field: jax.Array

    def __init__(self, lengths, dims, *, ne_type = None, inv_brems = False, opacity = False, phaseshift = False, B_on = False, probing_direction = 'z', auto_batching = True, iteration = 1, region_count = 1, leeway_factor = None, coord_backup = None, future_dims = None, extra_info = False, memory_reporting = False, memory_limit = None, Np = None,
        s = None, s1 = None, s2 = None, Ly = None, ne_0 = None, ne = None, B = None, Bmax = None, Te = None, Te_min = None, Z = None, opacity_files = None, densities = None, num_materials = None, edensity = True, refrac_field = None):
        """
        A class to set-up/generate the scalar simulation domains and store for later use.

        :param lengths: Specifies the size of the domain, this is full length, so +/- half this length from the origin.
            e.g. 2mm means -1mm --> 1mm size
        :type lengths: shared.utils.generic_valid_types

        :param dims: Specifies the resolution of the domain, number of divisions in each axis from which the cells are formed.
        :type dims: shared.utils.generic_valid_types

        :param ne_type: Sets the type of domain for the class functions to allocate.
        :type ne_type: str or None

        :param inv_brems: Disables python multithreading to prevent conflict with jax parallelisation in some instances.
        :type inv_brems: bool (default = True)

        :param opacity:
        :type opacity:

        :param phaseshift: Enable 64-bit values in jax (double precision floating point arithmetic).
        :type phaseshift: bool (default = False)

        :param B_on: Enables debug flags, increases runtime.
        :type B_on: bool (default = False)

        :param probing_direction: Set's the direction the beam is propagating in.
        :type probing_direction: char (default = 'z')

        :param auto_batching: Enable automatic domain and ray batching algorithm.
        :type auto_batching: bool (default = True)

        :param iteration: Indicates which batch of rays this is, important to distinguish if it is the first or not.
        :type iteration: int (default = 1)

        :param region_count: The number of regions the domain is batched into.
        :type region_count: int, default: 1

        :param leeway_factor: Scalar to memory usage estimations, used to give extra leeway to total predictions.
        :type leeway_factor: float, default: 1.1 - 10% margin

        :param coord_backup: Co-ordinates of the previous domain object.
        :type coord_backup: jax.Array, default: None

        :param future_dims: Array of dim allocations for past/current/future domain objects.
        :type future_dims: jax.Array, default: None

        :param extra_info: Flag to enable printing of extra information - namely domain parameters.
        :type extra_info: bool, default: False

        :param memory_reporting: Flag to enable priting of memory information.
        :type memory_reporting: bool, default: False

        :param memory_limit: Set arbritrary jax memory usage limit in KiB, overrides autodetection of available space, see [here] for more details.
        :type memory_limit: np.int64, default: None

        :param Np: The total number of rays to be simulated - needs to be set if intending to batch ray generation.
        :type Np: int, default: None

        + plus an assortment of paramters for domain generation that can be set to override defaults
            (s, s1, s2, Ly, ne_0, ne, B, Bmax, Te, Te_min, Z)

        :raise Exception: If lengths or dims are an array of len(...) != 1 but not len(...) == 3
        :raise AssertionError: If ne_type is changed from the default but not set to a valid type.
        :raise AssertionError: If probing_direction is not == "x", "y" or "z".

        :return: Returns an equinox.Module inheriting object containing information about and the generated/imported domain itself.
        :rtype: simulator.domain.ScalarDomain
        """

        ###
        ### Can some of these flags be moved to propagator.py instead?
        ###

        # Logical switches
        self.inv_brems = inv_brems
        del inv_brems
        self.opacity = opacity
        del opacity
        self.phaseshift = phaseshift
        del phaseshift
        self.B_on = B_on
        del B_on
        self.edensity = edensity
        del edensity

        # initalise
        self.s = s
        del s

        self.s1 = s1
        del s1

        self.s2 = s2
        del s2

        self.Ly = Ly
        del Ly

        self.ne_0 = ne_0
        del ne_0

        self.ne = ne
        del ne

        self.B = B
        del B

        self.Bmax = Bmax
        del Bmax

        self.Te_min = Te_min
        del Te_min

        if self.Te_min is not None and Te is not None:
            self.Te = jnp.maximum(self.Te_min, Te)
        else:
            self.Te = Te

        del Te

        self.Z = Z
        del Z

        if self.edensity == True and refrac_field is not None:
            print(colour.BOLD + "\nBy setting edensity == True, refrac_field will not be used. If this is intended, we suggest you do not pass this value in future." + colour.END)
            print(" --> Overriding self.refrac_field entry to None")

            self.refrac_field = None
        else:
            self.refrac_field = refrac_field
        del refrac_field

        self.opacity_files = opacity_files
        del opacity_files

        self.densities = densities
        del densities

        self.num_materials = num_materials
        del num_materials

        self.probing_direction = probing_direction

        self.ne_type = ne_type

        assert (self.edensity == True and (self.ne is not None or self.ne_type is not None)), "\nMust pass either a pre-generated field or a type of field to generate."
        assert not (self.edensity == False and self.refrac_field is not None), "\nIf edensity == False, refrac_field must be supplied."

        # working with 10% leeway in estimate for now
        if leeway_factor is not None:
            self.leeway_factor = leeway_factor
        else:
            # set to 1.1 by default, gives 10% leeway in prediction
            self.leeway_factor = 1.1

        self.extra_info = extra_info
        # not used right now but probably will be in the future so not bothering to remove
        self.memory_reporting = memory_reporting
        self.memory_limit = memory_limit

        if Np is not None:
            self.Np_total = np.int64(Np)
        else:
            self.Np_total = None

        self.ray_batch_count = 1

        ##
        ## NOT FORCING THESE CONVERSIONS MAY CAUSE ISSUES WITH EQUINOX CLASS LATER DOWN THE LINE DEPENDING ON USER INPUT
        ##

        from shared.utils import generic_valid_types as valid_types

        # if 1 length given, assumes all are the same
        if isinstance(lengths, valid_types):
            self.x_length, self.y_length, self.z_length = lengths, lengths, lengths
            self.lengths = jnp.array([lengths, lengths, lengths])
        # if array given, checks len = 3 and assigns accordingly
        else:
            self.lengths = jnp.array(lengths)
            if self.lengths.shape != (3,):
                raise Exception('lengths must have len = 3: (x,y,z)')

            self.x_length, self.y_length, self.z_length = self.lengths[0], self.lengths[1], self.lengths[2]

        del lengths
        
        #likewise for dims
        #self.dims = dims
        if isinstance(dims, valid_types):
            self.x_n, self.y_n, self.z_n = dims, dims, dims
            self.dims = jnp.array([dims, dims, dims])
        else:
            self.dims = jnp.array(dims)
            if self.dims.shape != (3,):
                raise Exception('n must have len = 3: (x_n, y_n, z_n)')

            self.x_n, self.y_n, self.z_n = self.dims[0], self.dims[1], self.dims[2]

        del dims
        del valid_types

        ###
        ### Why has this been set to override it?
        ### Explains the override in propagator, but does this functionality make sense?
        ###

        if self.opacity:
            self.inv_brems = False

        # changed function to pass to np.int64 to prevent overflow - this was causing the negatives
        # --> (exactly 0 in the case of a 1024^3 domain as it is right on the limit)
        predicted_domain_allocation = domain_estimate(self.x_n, self.y_n, self.z_n)
        print("Predicted size in memory of domain:", mem_conversion(predicted_domain_allocation))

        if iteration == 1 and auto_batching and self.edensity == True:
            memory_stats = memory_report(memory_limit = memory_limit)

            print("\nMemory prior to domain creation:")
            print(f" - total : {memory_stats['total']}")
            print(f" - free  : {memory_stats['free']}")
            print(f" - used  : {memory_stats['used']}")

            ###
            ### Need to work out the max allocation at any point and that estimated size
            ###

            # 2 for ne and ne_nc in calc_dndr(...) before ne is deleted
            # at peak mem usage ne should have been deleted, therefore this contributes only 1 domain
            # +1 for ne_interp
            # +2 for the 2 sequentially repeated domain sized allocations in dndr(...)

            # 1 for ne
            # +1 for sequential interps (dndx, dndy, dndz)
            # no more need for domain allocation in dndr as now functionally interpolated
            allocation_count = 2

            # up to +5 in calc_dndr(...) depending on the number of extra interps
            if self.B_on:
                # there are 4 B based interps
                # and they also require a ScalarDomain.B domain sized matrice
                allocation_count += 4
            if self.inv_brems:
                # unsure how many intermediaries exist at peak mem usage for this allocation - need to check and adjust this
                allocation_count += 1
            if self.phaseshift:
                allocation_count += 1

            # compare to max allocation in domain setup and return the greatest
            if self.ne_type == "test_null" or self.ne_type == "test_slab" or self.ne_type == "test_B":
                allocation_count = max(allocation_count, 2)
            elif self.ne_type == "test_linear_cos" or self.ne_type == "test_exponential_cos" or self.ne_type is None:
                allocation_count = max(allocation_count, 3)
            elif self.ne_type == "import":
                allocation_count = max(allocation_count, 1)
            else:
                raise AssertionError("\nNo valid profile detected! Ensure passed name is correct or call yourself.")

            print("")
            if self.Np_total is not None:
                import simulator.beam as ray_test_case

                test_beam = ray_test_case.Beam(1, 1, 1, 1)
                single_ray = test_beam.s0 # just initialises 1 ray of any variety
                del test_beam

                ray_memory_raw = np.float64(getsizeof_default(single_ray) * self.Np_total) * self.leeway_factor
                del single_ray

                print("Est. ray size in memory:", mem_conversion(ray_memory_raw))

            estimate_limit = np.float64(predicted_domain_allocation * allocation_count * self.leeway_factor)
            print("Est. domain memory limit: {}".format(mem_conversion(estimate_limit)))
            print(" --> inc. +{}% variance margin".format(jnp.float32((self.leeway_factor - 1) * 100)))

            if self.Np_total is not None:
                limiting_value = estimate_limit + ray_memory_raw * self.leeway_factor
                print("Total estimated maximum: {}".format(mem_conversion(limiting_value)))
            else:
                limiting_value = estimate_limit

            # when jnp.float32 is not used, will cause overflow error if 64 bit floats are not enabled
            if limiting_value > np.float64(memory_stats['free_raw']):
                if self.ne is None:
                    if self.Np_total is None:
                        print(colour.BOLD + "\nESTIMATE SUGGESTS DOMAIN CANNOT FIT IN AVAILABLE MEMORY." + colour.END)
                    else:
                        print(colour.BOLD + "\nESTIMATE SUGGESTS DOMAIN + RAYS CANNOT FIT IN AVAILABLE MEMORY." + colour.END)
                        self.ray_batch_count = np.int64(ceil(ray_memory_raw * self.leeway_factor / np.float64(memory_stats['free_raw'])))
                    print(" --> Auto-batching domain based on memory available and domain size estimate...")

                    ##
                    ## Used backed up information to re-assign to ScalarDomain in propagator
                    ## Then call generate_electron_density_profile(...) and re-do calculations with end of prior domain
                    ##

                    #self.region_count = ceil((limiting_value - ray_memory_raw / self.ray_batch_count) / np.float64(memory_stats['free_raw']))
                    self.region_count = ceil(np.float64(predicted_domain_allocation * allocation_count) / (np.float64(memory_stats['free_raw']) - ceil(ray_memory_raw / self.ray_batch_count)))

                    self.coord_backup = jnp.float32(jnp.linspace(
                    -self.lengths[['x', 'y', 'z'].index(self.probing_direction)] / 2,
                        self.lengths[['x', 'y', 'z'].index(self.probing_direction)] / 2,
                        self.dims[['x', 'y', 'z'].index(self.probing_direction)]
                    ))

                    dim_per_region = self.dims[['x', 'y', 'z'].index(self.probing_direction)] // self.region_count
                    self.future_dims = jnp.concatenate([
                        jnp.expand_dims(0, axis = 0), jnp.array([dim_per_region] * self.region_count),
                        jnp.array([self.dims[['x', 'y', 'z'].index(self.probing_direction)] - dim_per_region * self.region_count])
                    ])

                    print(" --> Batching calculation completed. Domain will be split into " + str(self.region_count) + " regions with " + str(dim_per_region) + " dims per region.")
                    print(colour.BOLD + "\nWARNING:" + colour.END + " This functionality will cause the solver to run slower due to domain regeneration, for optimal performance, increase the memory available to this program.")
                    if self.Np_total is not None:
                        print(" --> The domain is batched with the goal of minimising ray batching. Ray batches introduce sequantiality which reduces speed.")
                else:
                    self.region_count = 1

                    print(colour.BOLD + "\nESTIMATE SUGGESTS DOMAIN + RAYS CANNOT FIT IN AVAILABLE MEMORY." + colour.END)
                    self.ray_batch_count = np.int64(ceil(ray_memory_raw * self.leeway_factor / np.float64(memory_stats['free_raw'])))
                    print(" --> Auto-batching rays based on memory available and domain size estimate...")
                    print(" --> Can't auto-batch the domain as it is imported (limitation will be resolved in the future)")
            else:
                self.region_count = 1
        else:
            self.region_count = region_count

        if self.region_count == 1:
            self.coord_backup = None
            self.future_dims = None

            # define coordinate space
            self.x = jnp.float32(jnp.linspace(-self.x_length / 2, self.x_length / 2, self.x_n))
            self.y = jnp.float32(jnp.linspace(-self.y_length / 2, self.y_length / 2, self.y_n))
            self.z = jnp.float32(jnp.linspace(-self.z_length / 2, self.z_length / 2, self.z_n))
        else:
            if iteration != 1:
                self.coord_backup = coord_backup
                self.future_dims = future_dims

            if iteration == 1:
                lower = 0
                upper = 65#self.future_dims[1] + 1
            else:
                lower = 64#jnp.sum(self.future_dims[0:iteration])
                upper = 128#lower + self.future_dims[iteration]

            if self.probing_direction == 'x':
                # define coordinate space
                self.x = self.coord_backup[lower:upper]
                self.y = jnp.float32(jnp.linspace(-self.y_length / 2, self.y_length / 2, self.y_n))
                self.z = jnp.float32(jnp.linspace(-self.z_length / 2, self.z_length / 2, self.z_n))

                self.x_length = self.x[-1] - self.x[0]
                self.lengths = self.lengths.at[0].set(self.x_length)

                self.x_n = len(self.x)
                self.dims = self.dims.at[0].set(self.x_n)
            elif self.probing_direction == 'y':
                # define coordinate space
                self.x = jnp.float32(jnp.linspace(-self.x_length / 2, self.x_length / 2, self.x_n))
                self.y = self.coord_backup[lower:upper]
                self.z = jnp.float32(jnp.linspace(-self.z_length / 2, self.z_length / 2, self.z_n))

                self.y_length = self.y[-1] - self.y[0]
                self.lengths = self.lengths.at[1].set(self.y_length)

                self.y_n = len(self.y)
                self.dims = self.dims.at[1].set(self.y_n)
            elif self.probing_direction == 'z':
                # define coordinate space
                self.x = jnp.float32(jnp.linspace(-self.x_length / 2, self.x_length / 2, self.x_n))
                self.y = jnp.float32(jnp.linspace(-self.y_length / 2, self.y_length / 2, self.y_n))
                self.z = self.coord_backup[lower:upper]

                self.z_length = self.z[-1] - self.z[0]
                self.lengths = self.lengths.at[2].set(self.z_length)

                self.z_n = len(self.z)
                self.dims = self.dims.at[2].set(self.z_n)
            else:
                raise AssertionError(colour.BOLD + "Invalid entry for probing_direction!" + colour.END)

        print("\nCoordinates have shape of ({}, {}, {})".format(len(self.x), len(self.y), len(self.z)), end = " --> ")

        if self.x.shape == self.y.shape and self.y.shape == self.z.shape and self.z.shape == self.x.shape:
            self.coordinates = jnp.stack([self.x, self.y, self.z], axis = 1, dtype = jnp.float32)

            print("no padding required.")
        else:
            max_dim = self.dims[0]
            for dimension in self.dims:
                if dimension > max_dim:
                    max_dim = dimension

            # pad coordinates but not arrays themselves, that way only interpolator takes in padded values - no needless extra mem allocation by the domain
            self.coordinates = jnp.stack([
                    jnp.pad(self.x, (0, max_dim - self.x_n), constant_values = jnp.nan),
                    jnp.pad(self.y, (0, max_dim - self.y_n), constant_values = jnp.nan),
                    jnp.pad(self.z, (0, max_dim - self.z_n), constant_values = jnp.nan)
                ], axis = 1)

            print("padded up-to {} entries.".format(max_dim))
            print(" --> x padded with: {} nan's".format(max_dim - self.x_n))
            print(" --> y padded with: {} nan's".format(max_dim - self.y_n))
            print(" --> z padded with: {} nan's".format(max_dim - self.z_n))

        if self.ne is None:
            if self.ne_type is not None:
                self.generate_electron_density_profile()
            else:
                assert auto_batching == True, colour.BOLD + "\nne_type must be passed to domain creation in order to utilise auto-batching." + colour.END

                # can't initialise yourself as equinox.Module inherited class is not mutable and self.ne is set during creation -- FIX!
                print("\nWARNING: Electron density profile to generate not passed. You will need to initialise this yourself with a call to this library.")
                print("\t If you run low on memory, you can enforce a manual domain cleanup with a call to ScalarDomain.cleanup()")

                self.XX, self.YY, self.ZZ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)#False) - has to be true for jnp
                self.ne = jnp.zeros((self.dims[0], self.dims[1], self.dims[2]))
        else:
            print("\nUsing imported ne domain. Be careful that your import matches with other passed variables, this is not sanity checked by the init function.")

            self.XX = None
            self.YY = None
            self.ZZ = None

        if self.extra_info:
            from shared.utils import round_to_n

            print(colour.BOLD + "\nScalarDomain object attribute info:" + colour.END)
            print(" --> lengths: {}, {}, {}".format(self.x_length, self.y_length, self.z_length))
            print(" --> lengths: {}".format(self.lengths))
        
            print("\n --> dims: {}, {}, {}".format(self.x_n, self.y_n, self.z_n))
            print(" --> dims: {}".format(self.dims))

            #arr = jnp.array([self.x[0], self.x[-1], self.y[0], self.y[-1], self.z[0], self.z[-1]])
            aim = 3

            '''
            round_to = find_sig_n(arr[0], aim)
            for i in arr:
                cache = find_sig_n(i, aim)
                if abs(cache) > abs(round_to):
                    round_to = cache

            for i in range(len(arr)):
                arr = arr.at[i].set(jnp.round(arr[i], round_to))
            '''

            print(f"\n --> x, y, z (s,e):",
                " [", round_to_n(self.x[0], aim), ", ", round_to_n(self.x[-1], aim), "],",
                " [", round_to_n(self.y[0], aim), ", ", round_to_n(self.y[-1], aim), "],",
                " [", round_to_n(self.z[0], aim), ", ", round_to_n(self.z[-1], aim), "]",
            sep = "")
            print(" --> their len's: {}, {}, {}".format(len(self.x), len(self.y), len(self.z)))

            if self.region_count != 1 and auto_batching:
                print("\n --> coord_backup: {}".format(
                    round_to_n(self.coord_backup[0], aim),
                    round_to_n(self.coord_backup[-1], aim)))
                print(" --> it's len: {}".format(len(self.coord_backup)))

                print("\n --> future dims: {}".format(self.future_dims))

        '''
        if B_on:
            self.B = jnp.array
        else:
            self.B = None

        etc...
        '''

    #@partial(jax.jit, static_argnames=("self",))  
    def generate_electron_density_profile(self):
        """
        Generate/import the selected electron density profile

        :param self: ScalarDomain object containing the domain to be generated's parameters.
        :type self: simulator.domain.ScalarDomain object

        :raise AssertionError: If ne_type is changed from the default but not set to a valid type.

        :return: No return, selects domain generation or import function and calls it from its assignment in the passed self object.
        :rtype: None
        """

        print("\nGenerating test", end = " ")
        if self.ne_type == "test_null":
            print("null -e field...")
            self.XX, _, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.YY = None
            self.ZZ = None

            self.test_null()
        elif self.ne_type == "test_slab":
            print("slab -e field...")
            self.XX, _, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.YY = None
            self.ZZ = None

            self.test_slab()
        elif self.ne_type == "test_linear_cos":
            print("linear decay periodic -e field...")
            self.XX, self.YY, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.ZZ = None

            self.test_linear_cos()
        elif self.ne_type == "test_exponential_cos" or self.ne_type is None:
            print("exponential decay periodic -e field...")
            self.XX, self.YY, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.ZZ = None

            self.test_exponential_cos()
        elif self.ne_type == "test_B":
            print("testB field...")
            self.XX, _, _ = jnp.meshgrid(self.x, self.y, self.z, indexing = 'ij', copy = True)

            self.YY = None
            self.ZZ = None

            self.test_B()
        elif self.ne_type == "import":
            print("pre-generated ne field is auto-imported if passed (not None)...")
        else:
            raise AssertionError("\nNo valid profile detected! Ensure passed name is correct or call yourself.")

        self.cleanup()

    #@partial(jax.jit, static_argnames=("self",))  
    def test_null(self):
        """
        Null test, an empty cube

        :param self: ScalarDomain object containing the domain to be generated's parameters.
        :type self: simulator.domain.ScalarDomain object

        :return: No return, exports the empty (zeroed) cubic domain as an attribute to the passed self object.
        :rtype: None
        """

        self.ne = self.ne.at[:, :, :].set(jnp.zeros_like(self.XX))

    #@partial(jax.jit, static_argnames=("self",))  
    def test_slab(self, *, s = 1, ne_0 = 2e23):
        """
        A slab with a linear gradient in x: n_e =  ne_0 * (1 + s * x / extent) - will cause a ray deflection in x

        :param self: ScalarDomain object containing the domain to be generated's parameters.
        :type self: simulator.domain.ScalarDomain object

        :param s: scale factor
        :type s: float, default: 1

        :param ne_0: mean electron density
        :type ne_0: float, default: 2e23 m\ :sup:`-3`

        :return: No return, generates domain as attribute to passed self object.
        :rtype: None
        """

        if self.s is not None:
            s = self.s
        if self.ne_0 is not None:
            ne_0 = self.ne_0

        self.ne = self.ne.at[:, :, :].set(ne_0 * (1.0 + s * self.XX / self.x_length))

    #@partial(jax.jit, static_argnames=("self",))  
    def test_linear_cos(self, *, s1 = 0.1, s2 = 0.1, ne_0 = 2e23, Ly = 1):
        """
        Linearly growing sinusoidal perturbation

        :param self: ScalarDomain object containing the domain to be generated's parameters.
        :type self: simulator.domain.ScalarDomain object

        :param s1: scale of linear growth
        :type s1: float, default: 0.1

        :param s2:  amplitude of sinusoidal perturbation
        :type 2: float, default: 0.1

        :param ne_0: mean electron density
        :type ne_0: float, default: 2e23

        :param Ly: spatial scale of sinusoidal perturbation
        :type Ly: float, default: 1

        :return: No return, generates domain as attribute to passed self object.
        :rtype: None
        """

        if self.s1 is not None:
            s1 = self.s1
        if self.s2 is not None:
            s2 = self.s2
        if self.ne_0 is not None:
            ne_0 = self.ne_0
        if self.Ly is not None:
            Ly = self.Ly

        self.ne = self.ne.at[:, :, :].set(ne_0 * (1.0 + s1 * self.XX / self.x_length) * (1 + s2 * jnp.cos(2 * jnp.pi * self.YY / Ly)))
    
    #@partial(jax.jit, static_argnames=("self",))  
    def test_exponential_cos(self, *, ne_0 = 1e24, Ly = 1e-3, s = -2e-3):
        """
        Exponentially growing/decaying sinusoidal perturbation

        :param self: ScalarDomain object containing the domain to be generated's parameters.
        :type self: simulator.domain.ScalarDomain object

        :param ne_0: mean electron density
        :type ne_0: float, default: 1e24 m\ :sup:`-3`

        :param Ly: spatial scale of sinusoidal perturbation
        :type Ly: float, default: 1e-3 [exponential decay]

        :param s: scale of exponential change
        :type s: float, default: -2e-3

        :return: No return, generates domain as attribute to passed self object.
        :rtype: None
        """

        if self.ne_0 is not None:
            ne_0 = self.ne_0
        if self.Ly is not None:
            Ly = self.Ly
        if self.s is not None:
            s = self.s

        self.XX = self.XX.at[:, :, :].set(self.XX / s)
        self.XX = self.XX.at[:, :, :].set(10 ** self.XX)

        self.YY = self.YY.at[:, :, :].set(self.YY / Ly)
        self.YY = self.YY.at[:, :, :].set(jnp.pi * self.YY)
        self.YY = self.YY.at[:, :, :].set(2 * self.YY)
        self.YY = self.YY.at[:, :, :].set(jnp.cos(self.YY))
        self.YY = self.YY.at[:, :, :].set(1 + self.YY)

        # any difference if float32 (or even 64 if changed later) or not? shouldn't be.
        self.ne = self.XX * self.YY
        self.cleanup()

        self.ne = self.ne.at[:, :, :].set(ne_0 * self.ne)

        #self.ne = jnp.float32(ne_0 * 10 ** (self.XX / s) * (1 + jnp.cos(2 * jnp.pi * self.YY / Ly)))

    #@partial(jax.jit, static_argnames=("self",))  
    def external_ne(self):
        """
        Load externally generated MxMxM grid of electron density (ne) in m\ :sup:`-3`

        :param self: ScalarDomain object containing the domain to be generated's parameters.
        :type self: simulator.domain.ScalarDomain object

        :return: No return, loads domain as an attribute to the self referenced object.
        :rtype: None
        """

        self.ne = self.ne.at[:, :, :].set(self.ne)

    '''
    #@partial(jax.jit, static_argnames=("self",))  
    def external_B(self):
        """
        Load externally generated MxMxMx3 grid of B field in T

        :param self: ScalarDomain object containing the domain to be generated's parameters.
        :type self: simulator.domain.ScalarDomain object

        :return: No return, loads domain as an attribute to the self referenced object.
        :rtype: None
        """

        self.B = self.B.at[:, :, :, :].set(B)

    #@partial(jax.jit, static_argnames=("self",))  
    def external_Te(self, *, Te, Te_min = 1.0):
        """
        Load externally generated MxMxM grid of electron temperature in eV

        :param self: ScalarDomain object containing the domain to be generated's parameters.
        :type self: simulator.domain.ScalarDomain object

        :param Te: MxMxM grid of electron temperature in eV
        :type Te: jax.Array or numpy.array of shape M^3

        :param Te_min: Set the minimum temperature of the grid
        :type Te_min: float, default: 1.0

        :return: No return, loads domain as an attribute to the self referenced object.
        :rtype: None
        """

        self.Te = self.Te.at[:, :, :].set(jnp.maximum(Te_min, Te))

    #@partial(jax.jit, static_argnames=("self",))  
    def external_Z(self, *, Z):
        """
        Load externally generated grid

        Args:
            Z ([type]): MxMxM grid of ionisation
        """
        """
        Load externally generated MxMxM grid of electron temperature in eV

        :param self: ScalarDomain object containing the domain to be generated's parameters.
        :type self: simulator.domain.ScalarDomain object

        :param Te: MxMxM grid of electron temperature in eV
        :type Te: jax.Array or numpy.array of shape M^3

        :param Te_min: Set the minimum temperature of the grid
        :type Te_min: float, default: 1.0

        :return: No return, loads domain as an attribute to the self referenced object.
        :rtype: None
        """

        self.Z = self.Z.at[:, :, :].set(Z)
    '''

    #@partial(jax.jit, static_argnames=("self",))  
    def test_B(self, *, Bmax = 1.0):
        """
        Generate a Bz field with a linear gradient in x: Bz =  Bmax * x / extent

        :param self: ScalarDomain object containing the domain to be generated's parameters.
        :type self: simulator.domain.ScalarDomain object

        :param Bmax: Limiting max value B field in a cell
        :type Te: float, default: 1.0 T [Tesla]

        :return: No return, loads domain as an attribute to the self referenced object.
        :rtype: None
        """

        if self.Bmax is not None:
            Bmax = self.Bmax

        self.B = self.B.at[:, :, :, :].set(jnp.zeros(jnp.append(jnp.array(self.XX.shape), 3)))
        self.B = self.B.at[:, :, :, 2].set(Bmax * self.XX / self.x_length)

    def export_scalar_field(self, property: str = 'ne', fname: str = None):
        """
        Export the current scalar electron density profile as a pvti file format, property added for future scalability to export temperature, B-field, etc.

        :param self: Part of the ScalarDomain class and thus takes in a self object.
        :type self: simulator.domain.ScalarDomain

        :param property: Sets the scalar field to export
        :type property: str (default = "ne")

        :param fname: file path and name to save under. A VTI pointed to by a PVTI file are saved in this location. If left blank, the name will default to: ./plasma_PVTI_DD_MM_YYYY_HR_MIN
        :type fname: str or None

        :raise Exception: Raises an exception if the desired scalar field is not loaded.

        :return: Has no return, saves a file to disk.
        :rtype: None
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
            fname = f"./plasma_PVTI_{property}_{day}_{month}_{year}_{hour}_{min}" #default fname to the current date and time 

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

            grid.save(f"{fname}.vti")

            print(f"VTI saved under {fname}.vti")

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
        with open(f"{fname}.pvti", "w") as file:
            file.write(content)

        print(f"Scalar Domain electron density succesfully saved under {fname}.pvti !")

    #@jax.jit
    def cleanup(self):
        """
        Deallocates unused temporary meshgrid variables. Calls to shared.utils.dalloc(...)

        :param self: Part of the ScalarDomain class and thus takes in a self object.
        :type self: simulator.domain.ScalarDomain

        :return: Updates the passed simulator.domain.ScalarDomain object.
        :rtype: None
        """

        if self.XX is not None:
            dalloc(self.XX)
        if self.YY is not None:
            dalloc(self.YY)
        if self.ZZ is not None:
            dalloc(self.ZZ)