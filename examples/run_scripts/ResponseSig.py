import sys

sys.path.insert(1, '/rds/general/user/le322/home/synthPy/')

import numpy as np
from mpi4py import MPI
import pickle
from mpi4py.util import pkl5

import solver.full_solver as s
import solver.rtm_solver as rtm

import gc

## Initialise the MPI
comm = pkl5.Intracomm(MPI.COMM_WORLD)

rank = comm.Get_rank()

Np_ray_split = int(5e5)

num_processors = comm.Get_size()

# define plasma quantities and simulate with max and min length scales 
extent = 5e-3
res = 16

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
	ss = s.init_beam(Np = Np, beam_size=beam_size, divergence = divergence, ne_extent = ne_extent, beam_type = 'square')
	## Propogate rays through ne_cube
	rf = field.solve(ss)
	# Save memory by deleting initial ray positions
	del ss
	# Convert to mm
	rf[0:4:2,:] *= 1e3 

	r = rtm.RefractometerRays(rf)
	r.solve()
	r.histogram(bin_scale = 1, clear_mem = True)

	sh=rtm.ShadowgraphyRays(rf)
	sh.solve(displacement = 0)
	sh.histogram(bin_scale = 1, clear_mem=True)



	return sh,r


# make electron cube

x = np.linspace(-extent, extent, res)
domain = s.ScalarDomain(x,x,x,extent)
domain.test_null()


domain.calc_dndr()
domain.clear_memory()


beam_size = 5e-3
divergence = 0.05e-3

# split ray bundle 

Np = int(float(sys.argv[1]))

number_of_splits = Np//Np_ray_split
remaining_rays   = Np%Np_ray_split

sh,r = system_solve(remaining_rays,beam_size,divergence, domain, extent)

for i in range(number_of_splits):

	# Solve subsystem
	sh_split, r_split = system_solve(Np_ray_split,beam_size,divergence,domain, extent)
	# Force garbage collection - this may be unnecessary but better safe than sorry
	gc.collect()
	# Add in results from splitting
	sh.H += sh_split.H
	r.H  += r_split.H
	if(rank == 0):
		print("%d of %d"%(i+1,number_of_splits))


# sum results back to master

sh.H = comm.reduce(sh.H,root=0,op=MPI.SUM)
r.H  = comm.reduce(r.H,root=0,op=MPI.SUM)

# save



output_loc = str(sys.argv[2])
if(rank == 0):

	# Save diagnostics as a pickle

	# For Shadowgraphy
	filehandler = open(output_loc + 'shadow.pkl',"wb")
	pickle.dump(sh,filehandler)
	filehandler.close()
	# For Refractometer
	filehandler = open(output_loc + 'refract.pkl',"wb")
	pickle.dump(r,filehandler)
	filehandler.close()
    




