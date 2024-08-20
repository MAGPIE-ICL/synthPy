'''
Script to do synthetic interferogram with pvti files

Author: Louis Evans
Reviewer: Stefano Merlini

Run with the following job script:
	#!/bin/sh
	#PBS -l walltime=HH:MM:SS
	#PBS -l select=1:ncpus=N:mpiprocs=N:mem=Mgb
	#PBS -j oe
	cd '/rds/general/user/le322/home/synthPy'

	module load anaconda3/personal

	source activate MAGPIE_venv #load venv

	mpiexec  -n <n_cpus> python run_scripts/interference_MPI.py <number of rays> <path/to/pvti> <output directory>
'''

import sys
sys.path.append('../synthPy/')      # import path/to/synthpy
import numpy as np
from mpi4py import MPI
import pickle
from mpi4py.util import pkl5
import utils.handle_filetypes as utilIO
import solver.full_solver as s
import solver.rtm_solver as rtm
import matplotlib.pyplot as plt
import gc
import vtk
from vtk.util import numpy_support as vtk_np

## Initialise the MPI
comm = pkl5.Intracomm(MPI.COMM_WORLD)
rank = comm.Get_rank()
Np_ray_split = int(5e5)
num_processors = comm.Get_size()

if __name__ == '__main__':
	#retrieve input variables
	Np = int(float(sys.argv[1]))
	file_loc = str(sys.argv[2])
	output_loc = str(sys.argv[3])

	#load pvti 
	ne, dim, spacing = utilIO.pvti_readin(str(file_loc))
	extent_x = ((dim[0]*spacing[0])/2)
	extent_y = ((dim[1]*spacing[1])/2)
	extent_z = ((dim[2]*spacing[2])/2)
	ne_x = np.linspace(-extent_x,extent_x, dim[0])
	ne_y = np.linspace(-extent_y,extent_y, dim[1])
	ne_z = np.linspace(-extent_z,extent_z, dim[2])

	# define beam parameters
	wl = 532e-9
	probing_direction = 'y'
	beam_size = 6e-3
	divergence = 0.05e-3

	field = s.ScalarDomain(ne_x, ne_y, ne_z, extent_x, phaseshift = True, probing_direction = probing_direction)
	field.external_ne(ne)
	field.calc_dndr(lwl = wl)
	del ne_x
	del ne_y
	del ne_z
	del ne

	if rank == 0:
		print('not background')
		print(f'''
			file_loc is {file_loc}
			wl is {wl}
			''')
		print('''Ray-Tracing...''')

	def system_solve(Np,beam_size,divergence,field, ne_extent, probing_direction, wl):
		## Initialise laser beam
		ss = s.init_beam(Np = Np, beam_size=beam_size, divergence = divergence, ne_extent = ne_extent, beam_type = 'circular', probing_direction = probing_direction)
		## Propogate rays through ne_cube
		rf, E = field.solve(ss, include_E = True)
		# Save memory by deleting initial ray positions
		del ss
		# Convert to mm
		rf[0:4:2,:] *= 1e3 
		print('solved, passing through optics')
		E = s.interfere_ref_beam(rf, E,120,-20) #vary n_fringes and degree for desired background signal
		interferogram =rtm.InterferometerRays(rf, E = E)
		interferogram.solve(wl = wl)
		interferogram.interferogram(bin_scale = 1, clear_mem=True)
		return interferogram

	# split ray bundle 
	number_of_splits = Np//Np_ray_split
	remaining_rays   = Np%Np_ray_split

	interferogram = system_solve(remaining_rays,beam_size,divergence, field, extent_y, probing_direction, wl)

	for i in range(number_of_splits):
		if(rank == 0):
			print("%d of %d"%(i+1,number_of_splits))
		# Solve subsystem
		interferogram_split = system_solve(Np_ray_split,beam_size,divergence,field, extent_y, probing_direction, wl)
		# Force garbage collection - this may be unnecessary but better safe than sorry
		gc.collect()
		interferogram.H += interferogram_split.H
	# sum results back to master
	interferogram.H = comm.reduce(interferogram.H,root=0,op=MPI.SUM)

# save result
	if(rank == 0):
		filehandler = open(output_loc + 'interferogram.pkl',"wb")
		pickle.dump(interferogram,filehandler)
		filehandler.close()
		




