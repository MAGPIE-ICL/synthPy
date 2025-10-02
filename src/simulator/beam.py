import numpy as np
import jax.numpy as jnp

from shared.utils import random_array
from shared.utils import random_array_n
from shared.utils import random_inv_pow_array

from shared.printing import colour

class Beam:
# Initialise beam
    def __init__(self, Np, beam_size, divergence, ne_extent, *, probing_direction = 'z', beam_type = 'circular', seeded = False):
        """
        Initialises a number of rays (initial positions, velocities) that form the probing beam of some set shape, size, collimation and direction
            -   sets up an object containing all this information
            -   calls a class function that initialises the Beam based on this

        :param Np: Number of photons
        :type Np: int

        :param beam_size: beam radius, m
        :type beam_size: float

        :param divergence: divergence of beam, radians
        :type divergence: float

        :param ne_extent: size of electron density cube, m. Used in initialisation of ray starting positions in auto init_beam() call
        :type ne_extent: float

        :param probing_direction: direction of probing. I suggest "z", the best tested
        :type probing_direction: str (default = "z")

        :param beam_type: The shape of the probing beam
        :type beam_type: str (allowed: "circular", "square", "rectangular", "linear", "even", "rect_trackers") (default = "circular")

        :param seeded: Sets a seed for when runs require consistency (eg. for benchmarks).
        :type seeded: bool (default = False). So long and thanks for all the fish.

        :raise AssertionError: If beam_size variable is not of the correct format for the selected beam shape.

        :return: Returns a Beam object containing laser probe and ray information.
        :rtype: simulator.beam.Beam
        """

        self.Np = np.int64(Np)
        self.beam_size = beam_size
        self.divergence = divergence
        self.ne_extent = ne_extent
        self.probing_direction = probing_direction
        self.beam_type = beam_type

        if seeded:
            self.seed = 42
        else:
            self.seed = None

        # calls actual initialisation of beam automatically, first function just initialises variables
        # forces ne_extent to negative when passed to init_beam(... ne_extent < 0 ...)
        Beam.init_beam(self, -self.ne_extent, self.seed) # [x if x < 0 else -x for x in jnp.array(ne_extent)]

    def init_beam(self, ne_extent, seed):
        """
        Function designed to be called by the Beam class during probe initialisation to complete the construction ray construction from beam parameterss.

        Updated object definitions:
            s0, 9 x N float: N rays with (x, y, z, vx, vy, vz) in m, m/s and amplitude, phase and polarisation (a, p, r)

        :param self: Beam object containing its parameters
        :type fn: simulator.beam.Beam object

        :param ne_extent: Shall be deprecated soon and passed directly from self after forced negative of array is fixed.
        :type ne_extent: float or array of floats

        :param seed: Shall be deprecated soon and passed directly from self after forced negative of array is fixed.
        :type seed: int or None

        :return: No return, updates then self object instance of class simulator.beam.Beam
        :rtype: None
        """

        from scipy.constants import c

        s0 = jnp.zeros((9, self.Np))
        if(self.beam_type == 'circular'):
            from shared.utils import generic_valid_types as valid_types
            assert isinstance(self.beam_size, valid_types), "\nReceived beam_size of shape" + str(len(self.beam_size)) + "expected a float."

            # position, uniformly within a circle
            t  = 2 * jnp.pi * random_array(self.Np, seed) #polar angle of position

            # inversely weights probability with radius so that positions are uniformly distributed
            u = random_inv_pow_array(2, self.Np, seed) # radial coordinate of position

            # angle
            ϕ = jnp.pi * random_array(self.Np) #azimuthal angle of velocity
            χ = self.divergence * random_array_n(self.Np, seed) #polar angle of velocity

            if(self.probing_direction == 'x'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.cos(χ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(ne_extent)
                s0 = s0.at[1, :].set(self.beam_size * u * jnp.cos(t))
                s0 = s0.at[2, :].set(self.beam_size * u * jnp.sin(t))
            elif(self.probing_direction == 'z'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
                s0 = s0.at[5, :].set(c * jnp.cos(χ))

                # Initial position
                s0 = s0.at[0, :].set(self.beam_size * u * jnp.cos(t))
                s0 = s0.at[1, :].set(self.beam_size * u * jnp.sin(t))
                s0 = s0.at[2, :].set(ne_extent)
            else: # Default to y
                #print("Default to y")
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(self.beam_size * u * jnp.cos(t))
                s0 = s0.at[1, :].set(ne_extent)
                s0 = s0.at[2, :].set(self.beam_size * u * jnp.sin(t))
        elif(self.beam_type == 'square'):
            from shared.utils import generic_valid_types as valid_types
            assert isinstance(self.beam_size, valid_types), "\nReceived beam_size of shape" + str(len(self.beam_size)) + "expected a float."

            # position, uniformly within a square
            t  = 2 * random_array(self.Np, seed) - 1.0
            u  = 2 * random_array(self.Np, seed) - 1.0

            # angle
            ϕ = jnp.pi * random_array(self.Np, seed) #azimuthal angle of velocity
            χ = self.divergence * random_array_n(self.Np, seed) #polar angle of velocity

            if(self.probing_direction == 'x'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.cos(χ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(ne_extent)
                s0 = s0.at[1, :].set(self.beam_size * u)
                s0 = s0.at[2, :].set(self.beam_size * t)
            elif(self.probing_direction == 'z'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
                s0 = s0.at[5, :].set(c * jnp.cos(χ))

                # Initial position
                s0 = s0.at[0, :].set(self.beam_size * u)
                s0 = s0.at[1, :].set(self.beam_size * t)
                s0 = s0.at[2, :].set(ne_extent)
            else: # Default to y
                #print("Default to y")
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(self.beam_size * u)
                s0 = s0.at[1, :].set(ne_extent)
                s0 = s0.at[2, :].set(self.beam_size * t)
        elif(self.beam_type == 'rectangular'):
            size_dim = len(self.beam_size)
            assert size_dim == 2, colour.BOLD + "\nERROR: " + colour.END + "Must pass a list of length 2 to initialise a rectangular beam," + str(size_dim) + "item was passed."

            # position, uniformly within a square
            t  = 2 * random_array(self.Np, seed) - 1.0
            u  = 2 * random_array(self.Np, seed) - 1.0

            # angle
            ϕ = jnp.pi * random_array(self.Np, seed) #azimuthal angle of velocity
            χ = self.divergence * random_array_n(self.Np, seed) #polar angle of velocity

            beam_size_1 = self.beam_size[0] #m
            beam_size_2 = self.beam_size[1] #m

            if(self.probing_direction == 'x'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.cos(χ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(ne_extent)
                s0 = s0.at[1, :].set(beam_size_1 * u)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            elif(self.probing_direction == 'z'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
                s0 = s0.at[5, :].set(c * jnp.cos(χ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(beam_size_2 * t)
                s0 = s0.at[2, :].set(ne_extent)
            else: # Default to y
                print("Default to y")
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(ne_extent)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            
            del beam_size_1
            del beam_size_2
        elif(self.beam_type == 'linear'):
            from shared.utils import generic_valid_types as valid_types
            assert isinstance(self.beam_size, valid_types), "\nReceived beam_size of shape" + str(len(self.beam_size)) + "expected a float."

            # position, uniformly along a line - probing direction is defaulted z, solved in x,z plane
            t  = 2 * random_array(self.Np, seed) - 1.0
            # angle
            χ = self.divergence * random_array_n(self.Np, seed) #polar angle of velocity

            # Initial velocity
            s0 = s0.at[3, :].set(c * jnp.sin(χ))
            s0 = s0.at[4, :].set(0.0)
            s0 = s0.at[5, :].set(c * jnp.cos(χ))
            # Initial position
            s0 = s0.at[0, :].set(self.beam_size * t)
            s0 = s0.at[1, :].set(0.0)
            s0 = s0.at[2, :].set(ne_extent)
        elif(self.beam_type == 'even'): # evenly distributed circular ray using concentric discs
            # number of concentric discs and points
            num_of_circles = (-1 + jnp.sqrt(1 + 8 * (self.Np // 6))) / 2 
            self.Np = 3 * (num_of_circles + 1) * num_of_circles + 1 

            # angle
            ϕ = jnp.pi * random_array(self.Np, seed) #azimuthal angle of velocity
            χ = self.divergence * random_array_n(self.Np, seed) #polar angle of velocity

            # position, uniformly within a circle
            t = [0]
            u = [0]

            # vectorise?
            for i in range(1, num_of_circles + 1): # for every disc
                for j in range(0, i * 6): # for every point in the disc
                    u.append(i / num_of_circles)
                    t.append(j * 2 * jnp.pi / (i * 6))  
        elif(self.beam_type == 'rect_trackers'):
            size_dim = len(self.beam_size)
            assert size_dim == 2, colour.BOLD + "\nERROR: " + colour.END + "Must pass a list of length 2 to initialise a rectangular beam," + str(size_dim) + "item was passed."

            # Randomly choose N_trackers indices to mark as tracking particles
            # tracker_indices = jnp.random.choice(self.Np, N_trackers, replace=False)

            # position, uniformly within a square
            t  = 2 * random_array(self.Np, seed) - 1.0
            u  = 2 * random_array(self.Np, seed) - 1.0

            # angle
            ϕ = jnp.pi * random_array(self.Np, seed) #azimuthal angle of velocity
            χ = self.divergence * random_array_n(self.Np, seed) #polar angle of velocity

            beam_size_1 = self.beam_size[0] #m
            beam_size_2 = self.beam_size[1] #m

            if(self.probing_direction == 'x'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.cos(χ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(ne_extent)
                s0 = s0.at[1, :].set(beam_size_1 * u)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            elif(self.probing_direction == 'y'):
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(ne_extent)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            elif(self.probing_direction == 'z'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
                s0 = s0.at[5, :].set(c * jnp.cos(χ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(beam_size_2 * t)
                s0 = s0.at[2, :].set(ne_extent)
            else: # Default to y
                print("Default to y")
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(ne_extent)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            
            del beam_size_1
            del beam_size_2
        else:
            print("\nself.beam_type unrecognised! Accepted args: circular, square, rectangular, linear, even, rect_trackers.")

        del t
        del u
        del ϕ
        del χ

        # Initialise amplitude, phase and polarisation
        s0 = s0.at[6, :].set(1.0)
        #s0 = s0.at[7, :].set(0.0)
        #s0 = s0.at[8, :].set(0.0)

        self.s0 = s0
        #self.rf = s0

        del s0

    def save_rays_pos(self, fn = None):
        """
        Saves the output rays as a binary numpy format for minimal size.
        Auto-names the file using the current date and time.

        :param fn: Overrides the default filename if set
        :type fn: str or None

        :return: No return, saves a file to disk.
        :rtype: None
        """

        from datetime import datetime

        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

        if fn is None:
            fn = '{} rays.npy'.format(dt_string)
        else:
            fn = '{}.npy'.format(fn)
        with open(fn,'wb') as f:
            jnp.save(f, self.s0)