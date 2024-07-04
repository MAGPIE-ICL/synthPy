import sys

sys.path.insert(1, 'z/:synthPy')

import numpy as np
from mpi4py import MPI
import pickle
from mpi4py.util import pkl5

import field_generator.gaussian1D as g1
import field_generator.gaussian2D as g2
import field_generator.gaussian3D as g3

import utils.cmpspec as cm

import solver.full_solver as s
import solver.rtm_solver as rtm
import matplotlib.pyplot as plt
import gc

## Initialise the MPI
comm = pkl5.Intracomm(MPI.COMM_WORLD)

rank = comm.Get_rank()

Np_ray_split = int(5e5)

num_processors = comm.Get_size()

if rank == 0:



    #  construct field

    def power_spectrum(k,a):
        return k**-a
        
    def k41(k):
        return power_spectrum(k, 5/3)

    def k42(k):
        return power_spectrum(k, 7/3)

    field_1 = g2.gaussian2D(k41)

    # define plasma quantities and simulate with max and min length scales 
    extent = 5e-3
    res = 1024
    l_max = 2*extent
    l_min = extent/res

    x, y, scalar_field = field_1.domain_fft(l_max, l_min, extent, res)

    field_1.export_scalar_field(fname = './16_field')

    scalar_field = 1e23 + 1e3*np.abs(scalar_field/np.max(np.abs(scalar_field)))

    x = MPI.pickle.dumps(x)
    y = MPI.pickle.dumps(y)
    scalar_field = MPI.pickle.dumps(scalar_field)
else: 
    x = None
    y = None
    scalar_field = None

x = comm.bcast(x, root = 0)
y = comm.bcast(y, root = 0)
scalar_field = comm.bcast(scalar_field, root = 0)

x = MPI.pickle.loads(x)
y = MPI.pickle.loads(y)
scalar_field = MPI.pickle.loads(scalar_field)

comm.barrier()

print(f'{rank} recieved field of dim: {scalar_field.shape}')

if rank == 0:
    print('''    
.______          ___   ____    ____    .___________..______          ___       ______  __  .__   __.   _______ 
|   _  \        /   \  \   \  /   /    |           ||   _  \        /   \     /      ||  | |  \ |  |  /  _____|
|  |_)  |      /  ^  \  \   \/   /     `---|  |----`|  |_)  |      /  ^  \   |  ,----'|  | |   \|  | |  |  __  
|      /      /  /_\  \  \_    _/          |  |     |      /      /  /_\  \  |  |     |  | |  . `  | |  | |_ | 
|  |\  \----./  _____  \   |  |            |  |     |  |\  \----./  _____  \ |  `----.|  | |  |\   | |  |__| | 
| _| `._____/__/     \__\  |__|            |__|     | _| `._____/__/     \__\ \______||__| |__| \__|  \______| 
                                                                                                               
''')


def system_solve(Np,beam_size,divergence,field, ne_extent):
	## Initialise laser beam
	ss = field.init_beam(Np = Np, beam_size=beam_size, divergence = divergence, ne_extent = ne_extent)
	## Propogate rays through ne_cube
	rf = field.solve(ss)
	# Save memory by deleting initial ray positions
	del ss
	# Convert to mm, a nicer unit
	rf[0:4:2,:] *= 1e3 

	r = rtm.RefractometerRays(rf)
	r.solve()
	r.histogram(bin_scale = 1, clear_mem = True)

	sh=rtm.ShadowgraphyRays(rf)
	sh.solve(displacement = 0)
	sh.histogram(bin_scale = 1, clear_mem=True)

	sc=rtm.SchlierenRays(rf)
	sc.solve()
	sc.histogram(bin_scale = 1, clear_mem=True)

	return sc,sh,r


# make electron cube

scalar_field = np.repeat(scalar_field[:, :, np.newaxis], scalar_field.shape[0], axis=2)
domain = s.ScalarDomain(x,y,y,extent)
domain.external_ne(scalar_field)

if rank == 0:
    domain.export_scalar_field(fname = './16_field')

domain.calc_dndr()
domain.clear_memory()


beam_size = 5e-3
divergence = 0.05e-3

# split ray bundle 

Np = int(float(sys.argv[1]))

number_of_splits = Np//Np_ray_split
remaining_rays   = Np%Np_ray_split

sc,sh,r = system_solve(remaining_rays,beam_size,divergence, domain, extent)

for i in range(number_of_splits):
    if(rank == 0):
        print("%d of %d"%(i+1,number_of_splits))
    # Solve subsystem
    sc_split,sh_split,r_split = system_solve(Np_ray_split,beam_size,divergence,domain, extent)
    # Force garbage collection - this may be unnecessary but better safe than sorry
    gc.collect()
    # Add in results from splitting
    sc.H += sc_split.H
    sh.H += sh_split.H
    r.H  += r_split.H


# sum results back to master

sc.H = comm.reduce(sc.H,root=0,op=MPI.SUM)
sh.H = comm.reduce(sh.H,root=0,op=MPI.SUM)
r.H  = comm.reduce(r.H,root=0,op=MPI.SUM)

# save

if(rank == 0):

	# Save diagnostics as a pickle
	# For Schlieren
	filehandler = open("output/Schlieren_16.pkl","wb")
	pickle.dump(sc,filehandler)
	filehandler.close()
	# For Shadowgraphy
	filehandler = open("output/Shadowgraphy_16.pkl","wb")
	pickle.dump(sh,filehandler)
	filehandler.close()
	# For Refractometer
	filehandler = open("output/Refractometer_16.pkl","wb")
	pickle.dump(r,filehandler)
	filehandler.close()
    




