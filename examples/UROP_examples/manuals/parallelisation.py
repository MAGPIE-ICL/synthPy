'''
Parallelisation (running mulptiple processes in parallel) greatly speeds up the ray tracing process and has been applied in two ways:
MPI4py: 
this library uses MPI (read the hpc wiki on MPI) for python library, and is run with the line: 

mpiexec -n python script.py

When the terminal sees mpiexec, it gets n cpus to run the process in parallel. Therefore, we add steps that can only be done for the 0th 'root' process, these steps include initialising
the domain, and saving the data at the end. (Nessecary becuase we only want to do these things once, as oppose to have every process run it)

This method is quick, but uses lots of memory. Every process has its own memory store, and thus if the script takes x GB of RAM on one processor, it will take n*x GB in paralllel, this 
issue is fixed by the next method, but takes slightly longer.

multiprocessing:
this library is native to python, and works a bit different to the prior method. It is run normally like:

python script.py

and is run on a master processor. This master processor sets up the problem by hosting a MemoryManager, which cleverly hosts a server, keeping the data stored at one address in RAM. 
The master processor then creates a list of tasks, and a pool of worker processors, assigning one or more tasks to each worker. Each worker can then access the main bulk of the data 
via the proxy server, without spawning copies of that data. However, because each worker is accessing a server to get to the stored object, a time delay is introduced. 

My advice: run a MPI4py script first, and if memory lacks, look into switching to the multiprocessing method

Find an example of both below.

- Louis

'''

#------------------------------------------------------------------MPI4py-------------------------------------------------------------------------------#
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
import utils.handle_filetypes as load

## Initialise the MPI
comm = pkl5.Intracomm(MPI.COMM_WORLD)
rank = comm.Get_rank() #this is the number assigned to each parallel process
num_processors = comm.Get_size()

#example ray trace: loading in a PVTI file and tracing for refractometry, and shadowgraphy
file_loc = 'file/path/to/file.pvti'

#define spatial scale
extent_x = 5e-3
extent_y = 5e-3
extent_z = 5e-3

if rank ==0: #rank 0 becuase we only want this calculation to happen once
	# load vti file
	ne, dim, spacing = load.pvti_readin(file_loc)
	
	ne_x = np.linspace(-extent_x,extent_x, dim[0])
	ne_y = np.linspace(-extent_y,extent_y, dim[1])
	ne_z = np.linspace(-extent_z,extent_z, dim[2])

	field = s.ScalarDomain(ne_x, ne_y, ne_z, extent_z)
	field.external_ne(ne)
	
	field.calc_dndr()
	field.clear_memory() #when writing scripts, ensure that large data are deleted after use
	del ne_x
	del ne_y
	del ne_z
	del ne 
else:
	field = None


comm.barrier() #barrier makes everyone wait here, so other processes wait for the root process to do the initial calculation

if rank == 0:
    print('''    
.______          ___   ____    ____    .___________..______          ___       ______  __  .__   __.   _______ 
|   _  \        /   \  \   \  /   /    |           ||   _  \        /   \     /      ||  | |  \ |  |  /  _____|
|  |_)  |      /  ^  \  \   \/   /     ---|  |----|  |_)  |      /  ^  \   |  ,----'|  | |   \|  | |  |  __  
|      /      /  /_\  \  \_    _/          |  |     |      /      /  /_\  \  |  |     |  | |  .   | |  | |_ | 
|  |\  \----./  _____  \   |  |            |  |     |  |\  \----./  _____  \ |  ----.|  | |  |\   | |  |__| | 
| _| ._____/__/     \__\  |__|            |__|     | _| ._____/__/     \__\ \______||__| |__| \__|  \______| 
                                                                                                               
''')

def system_solve(Np,beam_size,divergence, field, ne_extent): #define the function to run the ray trace, and synthetic diagnostics
	## Initialise laser beam
	ss = s.init_beam(Np = Np, beam_size=beam_size, divergence = divergence, ne_extent = ne_extent, beam_type = 'square')
	## Propogate rays through ne_cube
	field = comm.bcast(field, root=0) #bcast sends the field object from the root process to all other processes

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

beam_size = extent_x
divergence = 0.05e-3

# split ray bundle up into more manageable pieces, each process runs this, so total rays is n_cpus*Np

Np = 1e7
Np_ray_split = 5e5
number_of_splits = Np//Np_ray_split
remaining_rays   = Np%Np_ray_split

sh,r = system_solve(remaining_rays,beam_size,divergence, field, extent_z)



for i in range(number_of_splits):
    if(rank == 0):
        print("%d of %d"%(i+1,number_of_splits))
    # Solve subsystem
    sh_split,r_split = system_solve(Np_ray_split,beam_size,divergence,field, extent_x)
    # Force garbage collection - this may be unnecessary but better safe than sorry
    gc.collect()
    # Add in results from splitting
    sh.H += sh_split.H
    r.H  += r_split.H

# sum results back to master
sh.H = comm.reduce(sh.H,root=0,op=MPI.SUM) #reduce with the MPI.SUM option takes the sh.H stored in every process and sums them, compiling it to the root process
r.H  = comm.reduce(r.H,root=0,op=MPI.SUM)
# save
output_loc = 'path/to/output/loc'

if(rank == 0): #again specify rank, as we don't want to write more than neccesary
	# For Shadowgraphy
	filehandler = open(output_loc + 'shadow.pkl',"wb")
	pickle.dump(sh,filehandler) #pickle is way of saving a complete python object
	# For Refractometer
	filehandler = open(output_loc + 'refract.pkl',"wb")
	pickle.dump(r,filehandler)
	filehandler.close()
	

#------------------------------------------------------------------multiprocessing-------------------------------------------------------------------------------#

import sys
sys.path.append('/rds/general/user/le322/home/synthPy/')
import numpy as np
import pickle
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from multiprocessing import Process
import solver.full_solver as s
import solver.rtm_solver as rtm
import gc
import utils.handle_filetypes as load




class CustomManager(BaseManager): #this is done so we can modify the manager
    pass


def calculate_field(file_loc, manager): #define a function to process the pvti file into a domain object
    ne, dim, spacing = load.pvti_readin(file_loc)
    extent_x = ((dim[0]*spacing[0])/2)*1e-3
    extent_y = ((dim[1]*spacing[1])/2)*1e-3
    extent_z = ((dim[2]*spacing[2])/2)*1e-3

    print(f'extents: {extent_x, extent_y, extent_z}')
    ne_x = np.linspace(-extent_x, extent_x, dim[0])
    ne_y = np.linspace(-extent_y, extent_y, dim[1])
    ne_z = np.linspace(-extent_z, extent_z, dim[2])
    field = m.ScalarDomain(ne_x, ne_y, ne_z, extent_z, probing_direction = 'z')       # modify probing direction HERE
    field.external_ne(ne)
    field.calc_dndr()
    field.clear_memory()
    del ne
    return field, extent_z, extent_x

def system_solve(args): #define a function to give to the pool of workers
    if args == None:
        print('No more rays, terminating.')
        os._exit(0)
    Np, beam_size, divergence, field, ne_extent = args
    ss = s.init_beam(Np=Np, beam_size=beam_size, divergence=divergence, ne_extent=ne_extent, beam_type='circular', probing_direction='z')       # modify probing direction HERE
    rf = field.solve(ss)
    del ss
    rf[0:4:2, :] *= 1e3
    r = rtm.RefractometerRays(rf)
    r.solve()
    r.histogram(bin_scale=1, clear_mem=True)
    sh = rtm.ShadowgraphyRays(rf)
    sh.solve(displacement=0)
    sh.histogram(bin_scale=1, clear_mem=True)
    return sh.H, r.H


if __name__ == '__main__':
    print('Initiating Manager......')

    CustomManager.register('ScalarDomain', s.ScalarDomain) #register the scalar domain objects as objects that can be hosted on the memory manager server
    m = CustomManager()
    m.start()

    file_loc = 'path/to/file.pvti'
    

    field, extent_z, extent_x = calculate_field(file_loc = file_loc, manager = m)


    Np = 1e6

    output_loc = 'path/to/output'

    cpu_count = mp.cpu_count()

    Np_ray_split = int(5e5) #again, split rays, each split is assigned to a worker, so Np is the total number of photons traced

    number_of_splits = Np // Np_ray_split

    remaining_rays = Np % Np_ray_split

    divergence =0.05e-3

    probing_extent = extent_z #change these values accordingly

    beam_size = extent_x

    tasks = [(remaining_rays, beam_size, divergence, field, extent_z)] if remaining_rays > 0 else [] #setup a list of task inputs
    
    


    for i in range(number_of_splits):
        tasks.append((Np_ray_split, beam_size, 0.05e-3, field, extent_z))
    
    print(len(tasks), ' tasks to complete, with ', cpu_count, ' cores')

    with mp.Pool(cpu_count) as pool: #create a pool of workers
        results = pool.map(system_solve, tasks) #assign the function and tasks inputs to workers

    print('results obtained')

    sh_H = np.zeros_like(results[0][0]) #sum the results
    r_H = np.zeros_like(results[0][1])

    for sh_result, r_result in results: #this step can take a while, so ensure the Np_ray_split is not too low
        sh_H += sh_result
        r_H += r_result
    
    print('results summed')

    with open(output_loc + 'shadow.pkl', "wb") as filehandler:
        pickle.dump(sh_H, filehandler)
    with open(output_loc + 'refract.pkl', "wb") as filehandler:
        pickle.dump(r_H, filehandler)

    print('files written to ', output_loc)

