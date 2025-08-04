#import numpy as np
import jax.numpy as jnp

from utils import random_array
from utils import random_array_n
from utils import random_inv_pow_array

class Beam:
# Initialise beam
    def __init__(self, Np, beam_size, divergence, ne_extent, *, probing_direction = 'z', wavelength = 1064e-9, beam_type = 'circular', seeded = False):
        """
        [summary]

        Args:
            Np (int): Number of photons
            beam_size (float): beam radius, m
            divergence (float): beam divergence, radians
            ne_extent (float): size of electron density cube, m. Used in initialisation of ray starting positions in auto init_beam() call
            probing_direction (str): direction of probing. I suggest 'z', the best tested

        Returns:
            s0, 9 x N float: N rays with (x, y, z, vx, vy, vz) in m, m/s and amplitude, phase and polarisation (a, p, r) 
        """

        self.Np = jnp.int64(Np)
        self.beam_size = beam_size
        self.divergence = divergence
        self.probing_direction = probing_direction
        self.beam_type = beam_type
        self.wavelength = wavelength

        #calls actual initialisation of beam automatically, first function just initialises variables
        Beam.init_beam(self, ne_extent, seeded)

    def init_beam(self, ne_extent, seeded):
        """
        function designed to be called by the propagtor class during propagator init to complete the construction of the beam using parameters about the scalar domain
        [summary]

        Values from object:
            Np (int): Number of photons
            beam_size (float): beam radius, m
            divergence (float): beam divergence, radians
            ne_extent (float): size of electron density cube, m. Used to back propagate the rays to the start
            probing_direction (str): direction of probing. I suggest 'z', the best tested

        Updated object definitions:
            s0, 9 x N float: N rays with (x, y, z, vx, vy, vz) in m, m/s and amplitude, phase and polarisation (a, p, r)

        Returns:
            Beam (class Beam): Updated Beam object instance of class
        """

        # take all variables from object properties
        Np = self.Np
        beam_size = self.beam_size
        divergence = self.divergence
        probing_direction = self.probing_direction 
        beam_type = self.beam_type

        from scipy.constants import c

        s0 = jnp.zeros((9, Np))
        if(beam_type == 'circular'):
            # position, uniformly within a circle
            t  = 2 * jnp.pi * random_array(Np, seeded) #polar angle of position

            #u  = random_array(Np)+random_array(Np) # radial coordinate of position
            #u[u > 1] = 2-u[u > 1]
            u = random_array(Np, seeded) # radial coordinate of position

            # inversely weights probability with radius so that positions are uniformly distributed
            u = random_inv_pow_array(2, Np, seeded) # radial coordinate of position

            # angle
            ϕ = jnp.pi * random_array(Np) #azimuthal angle of velocity
            χ = divergence * random_array_n(Np, seeded) #polar angle of velocity

            if(probing_direction == 'x'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.cos(χ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(-ne_extent)
                s0 = s0.at[1, :].set(beam_size * u * jnp.cos(t))
                s0 = s0.at[2, :].set(beam_size * u * jnp.sin(t))
            elif(probing_direction == 'z'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
                s0 = s0.at[5, :].set(c * jnp.cos(χ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size * u * jnp.cos(t))
                s0 = s0.at[1, :].set(beam_size * u * jnp.sin(t))
                s0 = s0.at[2, :].set(-ne_extent)
            else: # Default to y
                #print("Default to y")
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size * u * jnp.cos(t))
                s0 = s0.at[1, :].set(-ne_extent)
                s0 = s0.at[2, :].set(beam_size * u * jnp.sin(t))
        elif(beam_type == 'square'):
            # position, uniformly within a square
            t  = 2 * random_array(Np, seeded) - 1.0
            u  = 2 * random_array(Np, seeded) - 1.0

            # angle
            ϕ = jnp.pi * random_array(Np, seeded) #azimuthal angle of velocity
            χ = divergence * random_array_n(Np, seeded) #polar angle of velocity

            if(probing_direction == 'x'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.cos(χ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(-ne_extent)
                s0 = s0.at[1, :].set(beam_size * u)
                s0 = s0.at[2, :].set(beam_size * t)
            elif(probing_direction == 'z'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
                s0 = s0.at[5, :].set(c * jnp.cos(χ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size * u)
                s0 = s0.at[1, :].set(beam_size * t)
                s0 = s0.at[2, :].set(-ne_extent)
            else: # Default to y
                #print("Default to y")
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size * u)
                s0 = s0.at[1, :].set(-ne_extent)
                s0 = s0.at[2, :].set(beam_size * t)
        elif(beam_type == 'rectangular'):
            # position, uniformly within a square
            t  = 2 * random_array(Np, seeded) - 1.0
            u  = 2 * random_array(Np, seeded) - 1.0

            # angle
            ϕ = jnp.pi * random_array(Np, seeded) #azimuthal angle of velocity
            χ = divergence * random_array_n(Np, seeded) #polar angle of velocity

            beam_size_1 = beam_size[0] #m
            beam_size_2 = beam_size[1] #m

            if(probing_direction == 'x'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.cos(χ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(-ne_extent)
                s0 = s0.at[1, :].set(beam_size_1 * u)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            elif(probing_direction == 'z'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
                s0 = s0.at[5, :].set(c * jnp.cos(χ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(beam_size_2 * t)
                s0 = s0.at[2, :].set(-ne_extent)
            else: # Default to y
                print("Default to y")
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(-ne_extent)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            
            del beam_size_1
            del beam_size_2
        elif(beam_type == 'linear'):
            # position, uniformly along a line - probing direction is defaulted z, solved in x,z plane
            t  = 2 * random_array(Np, seeded) - 1.0
            # angle
            χ = divergence * random_array_n(Np, seeded) #polar angle of velocity

            # Initial velocity
            s0 = s0.at[3, :].set(c * jnp.sin(χ))
            s0 = s0.at[4, :].set(0.0)
            s0 = s0.at[5, :].set(c * jnp.cos(χ))
            # Initial position
            s0 = s0.at[0, :].set(beam_size * t)
            s0 = s0.at[1, :].set(0.0)
            s0 = s0.at[2, :].set(-ne_extent)
        elif(beam_type == 'even'): # evenly distributed circular ray using concentric discs
            # number of concentric discs and points
            num_of_circles = (-1 + jnp.sqrt(1 + 8 * (Np // 6))) / 2 
            Np = 3 * (num_of_circles + 1) * num_of_circles + 1 

            # angle
            ϕ = jnp.pi * random_array(Np, seeded) #azimuthal angle of velocity
            χ = divergence * random_array_n(Np, seeded) #polar angle of velocity

            # position, uniformly within a circle
            t = [0]
            u = [0]

            # vectorise?
            for i in range(1, num_of_circles + 1): # for every disc
                for j in range(0, i * 6): # for every point in the disc
                    u.append(i / num_of_circles)
                    t.append(j * 2 * jnp.pi / (i * 6))  
        elif(beam_type == 'rect_trackers'):
            # Randomly choose N_trackers indices to mark as tracking particles
            # tracker_indices = jnp.random.choice(Np, N_trackers, replace=False)

            # position, uniformly within a square
            t  = 2 * random_array(Np, seeded) - 1.0
            u  = 2 * random_array(Np, seeded) - 1.0

            # angle
            ϕ = jnp.pi * random_array(Np, seeded) #azimuthal angle of velocity
            χ = divergence * random_array_n(Np, seeded) #polar angle of velocity

            beam_size_1 = beam_size[0] #m
            beam_size_2 = beam_size[1] #m

            if(probing_direction == 'x'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.cos(χ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(-ne_extent)
                s0 = s0.at[1, :].set(beam_size_1 * u)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            elif(probing_direction == 'y'):
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(-ne_extent)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            elif(probing_direction == 'z'):
                # Initial velocity
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[4, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))
                s0 = s0.at[5, :].set(c * jnp.cos(χ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(beam_size_2 * t)
                s0 = s0.at[2, :].set(-ne_extent)
            else: # Default to y
                print("Default to y")
                # Initial velocity
                s0 = s0.at[4, :].set(c * jnp.cos(χ))
                s0 = s0.at[3, :].set(c * jnp.sin(χ) * jnp.cos(ϕ))
                s0 = s0.at[5, :].set(c * jnp.sin(χ) * jnp.sin(ϕ))

                # Initial position
                s0 = s0.at[0, :].set(beam_size_1 * u)
                s0 = s0.at[1, :].set(-ne_extent)
                s0 = s0.at[2, :].set(beam_size_2 * t)
            
            del beam_size_1
            del beam_size_2
        else:
            print("beam_type unrecognised! Accepted args: circular, square, rectangular, linear")
        
        del t
        del u
        del ϕ
        del χ

        # Initialise amplitude, phase and polarisation
        s0 = s0.at[6, :].set(1.0)
        s0 = s0.at[8, :].set(0.0)
        s0 = s0.at[7, :].set(0.0)

        self.s0 = s0
        #self.rf = s0

        del s0

    def save_rays_pos(self, fn = None):
        """
        Saves the output rays as a binary numpy format for minimal size.
        Auto-names the file using the current date and time.
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