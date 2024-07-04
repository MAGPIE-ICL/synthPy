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

import solver.minimal_solver as s
import solver.rtm_solver as rtm
import matplotlib.pyplot as plt
import gc

## Initialise the MPI
comm = pkl5.Intracomm(MPI.COMM_WORLD)

rank = comm.Get_rank()

Np_ray_split = int(5e5)

num_processors = comm.Get_size()

def power_spectrum(k,a):
    return k**-a
	
def k41(k):
    return power_spectrum(k, 5/3)


n_extent = 300

if(rank == 0):
	print('rank 0!')
    # generate field
	field = g1.gaussian1D(k41)



	ne_pert = field.fft(n_extent)

	#make all positive and normalise
	ne_pert = ne_pert/np.max(ne_pert)
	ne_vals = 1e24 + 1e24*ne_pert

	field.ne = ne_vals

	field.export_scalar_field(fname = './1D_export')



	ne = MPI.pickle.dumps(ne_vals)

	requests = []

	for i in range(1, num_processors):

		req = comm.isend(ne, dest=i, tag=0)
		requests.append(req)
	

	print(f'''Field created, with
	extent = {10e-3}
	n_cells = {2*n_extent}
	power spectrum = {-5/3}''')

	print('''ray tracing''')

	
	
comm.barrier()


if(rank != 0):
    serialized_ne = comm.recv(source=0, tag=0)
    
    ne_vals = MPI.pickle.loads(serialized_ne)

    print(f"Processor {rank} received the field")


comm.barrier()


def system_solve(Np,beam_size,divergence,cube):
	'''
	Main function called by all processors, considers Np rays traversing the electron density volume ne_cube

	beam_size and divergence set the initial properties of the laser beam, ne_extent contains the size of the electron density volume

	'''

	## Initialise laser beam
	ss = cube.init_beam(Np = Np, beam_size=beam_size, divergence = divergence)
	## Propogate rays through ne_cube
	rf = cube.solve(ss) # output of solve is the rays in (x, theta, y, phi) format
	# Save memory by deleting initial ray positions
	del ss
	# Convert to mm, a nicer unit
	rf[0:4:2,:] *= 1e3 

	## Ray transfer matrix
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




Np=int(float(sys.argv[1]))

if rank == 0 :
	print("Number of processors: %s"%num_processors)
	print("Rays per processors: %s"%Np)



beam_size = 5e-3 # 5 mm
divergence = 0.05e-3 #0.05 mrad, realistic

extent = 10e-3
xs = np.linspace(-extent/2, extent/2, 2*n_extent + 1)
ys = np.linspace(-extent/2, extent/2, 2*n_extent + 1)
zs = np.linspace(-extent/2, extent/2, 2*n_extent + 1)
field = s.ScalarDomain(xs,ys,zs)



D = len(ne_vals)
ne_flatten = list(ne_vals)*D*D
ne = np.array(ne_flatten).reshape((D,D,D)).T




field.external_ne(ne)

field.calc_dndr()

if(rank == 0):
	field.export_scalar_field(fname = './output/1D_vti_export')
field.clear_memory()

# May trip memory limit, so split up calculation

if(Np > Np_ray_split):
	number_of_splits = Np//Np_ray_split
	remaining_rays   = Np%Np_ray_split
	if(rank == 0):
		print("Splitting to %d ray bundles"%number_of_splits)
		# Solve subsystem once to initialise the diagnostics
		print("Solve for remainder: %d"%remaining_rays)
	# Remaining_rays could be zero, this doesn't matter
	sc,sh,r = system_solve(remaining_rays,beam_size,divergence,field)
	# Iterate over remaining ray bundles
	for i in range(number_of_splits):
		if(rank == 0):
			print("%d of %d"%(i+1,number_of_splits))
		# Solve subsystem
		sc_split,sh_split,r_split = system_solve(Np_ray_split,beam_size,divergence,field)
		# Force garbage collection - this may be unnecessary but better safe than sorry
		gc.collect()
		# Add in results from splitting
		sc.H += sc_split.H
		sh.H += sh_split.H
		r.H  += r_split.H
else:
	print("Solving whole system...")
	sc,sh,r = system_solve(Np_ray_split,beam_size,divergence,field)

## Now each processor has calculated Schlieren, Shadowgraphy and Refractometer results
## Must sum pixel arrays and give to root processor

# Collect and sum all results and store on only root processor
sc.H = comm.reduce(sc.H,root=0,op=MPI.SUM)
sh.H = comm.reduce(sh.H,root=0,op=MPI.SUM)
r.H  = comm.reduce(r.H,root=0,op=MPI.SUM)

# Perform file saves on root processor only
if(rank == 0):

	# Save diagnostics as a pickle
	# For Schlieren
	filehandler = open("output/Schlieren.pkl","wb")
	pickle.dump(sc,filehandler)
	filehandler.close()
	# For Shadowgraphy
	filehandler = open("output/Shadowgraphy.pkl","wb")
	pickle.dump(sh,filehandler)
	filehandler.close()
	# For Refractometer
	filehandler = open("output/Refractometer.pkl","wb")
	pickle.dump(r,filehandler)
	filehandler.close()
