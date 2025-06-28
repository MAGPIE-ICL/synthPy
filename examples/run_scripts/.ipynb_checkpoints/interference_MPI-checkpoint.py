import sys

sys.path.insert(1, '/rds/general/user/le322/home/synthPy')

import numpy as np
from mpi4py import MPI
import pickle
from mpi4py.util import pkl5


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

print(num_processors)

def pvti_readin(filename):
	'''
	Reads in data from pvti with filename, use this to read in electron number density data

	'''

	reader = vtk.vtkXMLPImageDataReader()
	reader.SetFileName(filename)
	reader.Update()

	data = reader.GetOutput()
	dim = data.GetDimensions()
	spacing = np.array(data.GetSpacing())

	v = vtk_np.vtk_to_numpy(data.GetCellData().GetArray(0))
	n_comp = data.GetCellData().GetArray(0).GetNumberOfComponents()
	
	vec = [int(i-1) for i in dim]

	if(n_comp > 1):
		vec.append(n_comp)

	if(n_comp > 2):
		img = v.reshape(vec,order="F")[0:dim[0]-1,0:dim[1]-1,0:dim[2]-1,:]
	else:
		img = v.reshape(vec,order="F")[0:dim[0]-1,0:dim[1]-1,0:dim[2]-1]

	dim = img.shape

	return img,dim,spacing


# load vti file
scale_factor = float(sys.argv[2])

file_loc = str(sys.argv[3])



wl = 532e-9

probing_direction = 'y'

# ne, dim, spacing = pvti_readin(str(file_loc))
dim = [10,10,10]
extent_x = 1e-2
extent_y = 1.5e-2
extent_z = 1e-2



ne_x = np.linspace(-extent_x,extent_x, dim[0])
ne_y = np.linspace(-extent_y,extent_y, dim[1])
ne_z = np.linspace(-extent_z,extent_z, dim[2])


field = s.ScalarDomain(ne_x, ne_y, ne_z, extent_x, phaseshift = True, probing_direction = probing_direction)

if rank == 0:
	print('not background')
	print(f'''
		scale factor is : {scale_factor}
		file_loc is {file_loc}
		wl is {wl}
		''')

# field.external_ne(ne*scale_factor)




field.test_null()
field.calc_dndr(lwl = wl)
# field.clear_memory()
del ne_x
del ne_y
del ne_z
# del ne




if rank == 0:
    print('''    
.______          ___   ____    ____    .___________..______          ___       ______  __  .__   __.   _______ 
|   _  \        /   \  \   \  /   /    |           ||   _  \        /   \     /      ||  | |  \ |  |  /  _____|
|  |_)  |      /  ^  \  \   \/   /     `---|  |----`|  |_)  |      /  ^  \   |  ,----'|  | |   \|  | |  |  __  
|      /      /  /_\  \  \_    _/          |  |     |      /      /  /_\  \  |  |     |  | |  . `  | |  | |_ | 
|  |\  \----./  _____  \   |  |            |  |     |  |\  \----./  _____  \ |  `----.|  | |  |\   | |  |__| | 
| _| `._____/__/     \__\  |__|            |__|     | _| `._____/__/     \__\ \______||__| |__| \__|  \______| 
                                                                                                               
''')


def system_solve(Np,beam_size,divergence,field, ne_extent, probing_direction, wl):
	## Initialise laser beam
	ss = s.init_beam(Np = Np, beam_size=beam_size, divergence = divergence, ne_extent = ne_extent, beam_type = 'circular', probing_direction = probing_direction)
	## Propogate rays through ne_cube
	rf, E = field.solve_with_E(ss)
	# Save memory by deleting initial ray positions
	del ss
	# Convert to mm
	rf[0:4:2,:] *= 1e3 
	print('solved, passing through optics')
	
	E = s.interfere_ref_beam(rf, E,120,-20)

	# r = rtm.RefractometerRays(rf)
	# r.solve()
	# r.histogram(bin_scale = 1, clear_mem = True)

	sh=rtm.ShadowgraphyRays(rf, E = E, interfere = True )
	sh.solve(displacement = 0, interfere = True, wl = wl)
	sh.interferogram(bin_scale = 1, clear_mem=True)

	# sc=rtm.SchlierenRays(rf)
	# sc.solve()
	# sc.histogram(bin_scale = 1, clear_mem=True)

	return sh




beam_size = 5e-3
divergence = 0.05e-3


# split ray bundle 

Np = int(float(sys.argv[1]))

number_of_splits = Np//Np_ray_split
remaining_rays   = Np%Np_ray_split

sh = system_solve(remaining_rays,beam_size,divergence, field, extent_x, probing_direction, wl)



for i in range(number_of_splits):
    if(rank == 0):
        print("%d of %d"%(i+1,number_of_splits))
    # Solve subsystem
    sh_split = system_solve(Np_ray_split,beam_size,divergence,field, extent_x, probing_direction, wl)



    # Force garbage collection - this may be unnecessary but better safe than sorry
    gc.collect()
    # Add in results from splitting
    # sc.H += sc_split.H
    sh.H += sh_split.H
    # r.H  += r_split.H


# sum results back to master

# sc.H = comm.reduce(sc.H,root=0,op=MPI.SUM)
sh.H = comm.reduce(sh.H,root=0,op=MPI.SUM)
# r.H  = comm.reduce(r.H,root=0,op=MPI.SUM)

# save

output_loc = str(sys.argv[4])

# output_loc = './background_'

if(rank == 0):

	# Save diagnostics as a pickle
	# For Schlieren
	# filehandler = open(output_loc + 'schlieren.pkl',"wb")
	# pickle.dump(sc,filehandler)
	# filehandler.close()
	# For Shadowgraphy
	filehandler = open(output_loc + '.pkl',"wb")
	pickle.dump(sh,filehandler)
	filehandler.close()
	# For Refractometer
	# filehandler = open(output_loc + 'refract.pkl',"wb")
	# pickle.dump(r,filehandler)
	# filehandler.close()
    




