import sys

sys.path.insert(1, '/rds/general/user/le322/home/synthPy')

import numpy as np
from mpi4py import MPI
import pickle
from mpi4py.util import pkl5

import field_generator.gaussian1D as g1
import field_generator.gaussian2D as g2
import field_generator.gaussian3D as g3

import solver.minimal_solver as s
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

file_loc = str(sys.argv[2])

ne, dim, spacing = pvti_readin(str(file_loc))

extent_x = ((dim[0] // 2)*spacing[0])* 1e-3
extent_y = ((dim[1] // 2)*spacing[1])* 1e-3
extent_z = ((dim[2] // 2)*spacing[2])* 1e-3

print(extent_x, extent_y, extent_z)

ne_x = np.linspace(-extent_x,extent_x, dim[0])
ne_y = np.linspace(-extent_y,extent_y, dim[1])
ne_z = np.linspace(-extent_z,extent_z, dim[2])


field = s.ScalarDomain(ne_x, ne_y, ne_z)
field.external_ne(ne)

field.calc_dndr()
field.clear_memory()
del ne_x
del ne_y
del ne_z
del ne 



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
	ss = field.init_beam(Np = Np, beam_size=beam_size, divergence = divergence)
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


beam_size = extent_x
divergence = 0.05e-3


# split ray bundle 

Np = int(float(sys.argv[1]))

number_of_splits = Np//Np_ray_split
remaining_rays   = Np%Np_ray_split

sc,sh,r = system_solve(remaining_rays,beam_size,divergence, field, extent_x)

for i in range(number_of_splits):
    if(rank == 0):
        print("%d of %d"%(i+1,number_of_splits))
    # Solve subsystem
    sc_split,sh_split,r_split = system_solve(Np_ray_split,beam_size,divergence,field, extent_x)
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

output_loc = str(sys.argv[3])

if(rank == 0):

	# Save diagnostics as a pickle
	# For Schlieren
	filehandler = open(output_loc + 'schlieren.pkl',"wb")
	pickle.dump(sc,filehandler)
	filehandler.close()
	# For Shadowgraphy
	filehandler = open(output_loc + 'shadow.pkl',"wb")
	pickle.dump(sh,filehandler)
	filehandler.close()
	# For Refractometer
	filehandler = open(output_loc + 'refract.pkl',"wb")
	pickle.dump(r,filehandler)
	filehandler.close()
    




