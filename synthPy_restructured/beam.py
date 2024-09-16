import numpy as np

class Beam:
# Initialise beam
    def __init__(self, Np, beam_size, divergence, probing_direction = 'z', wavelength = 450e-9, beam_type = 'circular'):
        """[summary]

        Args:
            Np (int): Number of photons
            beam_size (float): beam radius, m
            divergence (float): beam divergence, radians
            probing_direction (str): direction of probing. I suggest 'z', the best tested

        Returns:
            s0, 9 x N float: N rays with (x, y, z, vx, vy, vz) in m, m/s and amplitude, phase and polarisation (a, p, r) 
        """
        self.Np = Np
        self.beam_size = beam_size
        self.divergence = divergence
        self.probing_direction = probing_direction
        self.beam_type = beam_type
        self.wavelength = wavelength


    def init_beam(self, ne_extent):
        """
        function designed to be called by the propagtor class during propagator init to complete the construction of the beam using parameters about the scalar domain
        """
        # take all variables from object properties
        Np = self.Np
        beam_size = self.beam_size
        divergence = self.divergence
        probing_direction = self.probing_direction 
        beam_type = self.beam_type

        s0 = np.zeros((9,Np))
        if(beam_type == 'circular'):
            # position, uniformly within a circle
            t  = 2*np.pi*np.random.rand(Np) #polar angle of position
            u  = np.random.rand(Np)+np.random.rand(Np) # radial coordinate of position
            u[u > 1] = 2-u[u > 1]
            # angle
            ϕ = np.pi*np.random.rand(Np) #azimuthal angle of velocity
            χ = divergence*np.random.randn(Np) #polar angle of velocity

            if(probing_direction == 'x'):
                # Initial velocity
                s0[3,:] = c * np.cos(χ)
                s0[4,:] = c * np.sin(χ) * np.cos(ϕ)
                s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
                # Initial position
                s0[0,:] = -ne_extent
                s0[1,:] = beam_size*u*np.cos(t)
                s0[2,:] = beam_size*u*np.sin(t)
            elif(probing_direction == 'y'):
                # Initial velocity
                s0[4,:] = c * np.cos(χ)
                s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
                s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
                # Initial position
                s0[0,:] = beam_size*u*np.cos(t)
                s0[1,:] = -ne_extent
                s0[2,:] = beam_size*u*np.sin(t)
            elif(probing_direction == 'z'):
                # Initial velocity
                s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
                s0[4,:] = c * np.sin(χ) * np.sin(ϕ)
                s0[5,:] = c * np.cos(χ)
                # Initial position
                s0[0,:] = beam_size*u*np.cos(t)
                s0[1,:] = beam_size*u*np.sin(t)
                s0[2,:] = -ne_extent
            else: # Default to y
                print("Default to y")
                # Initial velocity
                s0[4,:] = c * np.cos(χ)
                s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
                s0[5,:] = c * np.sin(χ) * np.sin(ϕ)        
                # Initial position
                s0[0,:] = beam_size*u*np.cos(t)
                s0[1,:] = -ne_extent
                s0[2,:] = beam_size*u*np.sin(t)
        elif(beam_type == 'square'):
            # position, uniformly within a square
            t  = 2*np.random.rand(Np)-1.0
            u  = 2*np.random.rand(Np)-1.0
            # angle
            ϕ = np.pi*np.random.rand(Np) #azimuthal angle of velocity
            χ = divergence*np.random.randn(Np) #polar angle of velocity

            if(probing_direction == 'x'):
                # Initial velocity
                s0[3,:] = c * np.cos(χ)
                s0[4,:] = c * np.sin(χ) * np.cos(ϕ)
                s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
                # Initial position
                s0[0,:] = -ne_extent
                s0[1,:] = beam_size*u
                s0[2,:] = beam_size*t
            elif(probing_direction == 'y'):
                # Initial velocity
                s0[4,:] = c * np.cos(χ)
                s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
                s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
                # Initial position
                s0[0,:] = beam_size*u
                s0[1,:] = -ne_extent
                s0[2,:] = beam_size*t
            elif(probing_direction == 'z'):
                # Initial velocity
                s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
                s0[4,:] = c * np.sin(χ) * np.sin(ϕ)
                s0[5,:] = c * np.cos(χ)
                # Initial position
                s0[0,:] = beam_size*u
                s0[1,:] = beam_size*t
                s0[2,:] = -ne_extent
            else: # Default to y
                print("Default to y")
                # Initial velocity
                s0[4,:] = c * np.cos(χ)
                s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
                s0[5,:] = c * np.sin(χ) * np.sin(ϕ)        
                # Initial position
                s0[0,:] = beam_size*u
                s0[1,:] = -ne_extent
                s0[2,:] = beam_size*t
        elif(beam_type == 'rectangular'):
            # position, uniformly within a square
            t  = 2*np.random.rand(Np)-1.0
            u  = 2*np.random.rand(Np)-1.0
            # angle
            ϕ = np.pi*np.random.rand(Np) #azimuthal angle of velocity
            χ = divergence*np.random.randn(Np) #polar angle of velocity

            beam_size_1 = beam_size[0] #m
            beam_size_2 = beam_size[1] #m

            if(probing_direction == 'x'):
                # Initial velocity
                s0[3,:] = c * np.cos(χ)
                s0[4,:] = c * np.sin(χ) * np.cos(ϕ)
                s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
                # Initial position
                s0[0,:] = -ne_extent
                s0[1,:] = beam_size_1*u
                s0[2,:] = beam_size_2*t
            elif(probing_direction == 'y'):
                # Initial velocity
                s0[4,:] = c * np.cos(χ)
                s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
                s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
                # Initial position
                s0[0,:] = beam_size_1*u
                s0[1,:] = -ne_extent
                s0[2,:] = beam_size_2*t
            elif(probing_direction == 'z'):
                # Initial velocity
                s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
                s0[4,:] = c * np.sin(χ) * np.sin(ϕ)
                s0[5,:] = c * np.cos(χ)
                # Initial position
                s0[0,:] = beam_size_1*u
                s0[1,:] = beam_size_2*t
                s0[2,:] = -ne_extent
            else: # Default to y
                print("Default to y")
                # Initial velocity
                s0[4,:] = c * np.cos(χ)
                s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
                s0[5,:] = c * np.sin(χ) * np.sin(ϕ)        
                # Initial position
                s0[0,:] = beam_size_1*u
                s0[1,:] = -ne_extent
                s0[2,:] = beam_size_2*t
        elif(beam_type == 'linear'):
            # position, uniformly along a line - probing direction is defaulted z, solved in x,z plane
            t  = 2*np.random.rand(Np)-1.0
            # angle
            χ = divergence*np.random.randn(Np) #polar angle of velocity

            # Initial velocity
            s0[3,:] = c * np.sin(χ)
            s0[4,:] = 0.0
            s0[5,:] = c * np.cos(χ)
            # Initial position
            s0[0,:] = beam_size*t
            s0[1,:] = 0.0
            s0[2,:] = -ne_extent
        else:
            print("beam_type unrecognised! Accepted args: circular, square, rectangular, linear")

        # Initialise amplitude, phase and polarisation
        s0[6,:] = 1.0
        s0[7,:] = 0.0
        s0[8,:] = 0.0
        self.s0 = s0

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
            np.save(f, self.rf)

    def interfere_ref_beam(self, E, n_fringes, deg):
            ''' input beam ray positions and electric field component, and desired angle of evenly spaced background fringes. 
            Deg is angle in degrees from the vertical axis
            returns:
                'interfered with' E field
            '''
            rf = self.rf
            if deg >= 45:
                deg = - np.abs(deg - 90)

                
            rad = deg* np.pi /180 #deg to rad
            y_weight = np.arctan(rad)#take x_weight is 1
            x_weight = np.sqrt(1-y_weight**2)

            ref_beam = np.exp(2*n_fringes/3 * 1.0j*(x_weight*rf[0,:] + y_weight * rf[2,:]))

            E[1,:] += ref_beam # assume ref_beam is polarised in y

            return E