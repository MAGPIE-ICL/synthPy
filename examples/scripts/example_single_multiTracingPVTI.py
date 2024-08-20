'''
Parallelised ray tracer for PVTI files

Author: Louis Evans
Reviewer: Stefano Merlini

Run with the following job submission: 
    #!/bin/sh
    #PBS -l walltime=HH:MM:SS
    #PBS -l select=1:ncpus=N:mpiprocs=N:mem=Mgb
    #PBS -j oe
    cd '/rds/general/user/le322/home/synthPy' #insert path

    module load anaconda3/personal

    source activate MAGPIE_venv #activate your venv

    python path/to/multiprocessing_trace.py <Number of Rays> <pvti/file/path> <output location>
'''

import sys
sys.path.append('../synthPy/')      #import path/to/synthpy
import numpy as np
import pickle
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from multiprocessing import Process
import field_generator.gaussian1D as g1
import field_generator.gaussian2D as g2
import field_generator.gaussian3D as g3
import utils.handle_filetypes as utilIO
import solver.full_solver as s
import solver.rtm_solver as rtm
import matplotlib.pyplot as plt
import gc
import vtk
import os
from vtk.util import numpy_support as vtk_np


class CustomManager(BaseManager):
    pass

def calculate_field(file_loc, manager, probing_direction):
    ne, dim, spacing = utilIO.pvti_readin(file_loc)
    extent_x = ((dim[0]*spacing[0])/2)
    extent_y = ((dim[1]*spacing[1])/2)
    extent_z = ((dim[2]*spacing[2])/2)
    print(f'extents: {extent_x, extent_y, extent_z}')
    print(f'dims: {dim[0], dim[1], dim[2]}')
    print(f'field ranges, max: {np.max(ne)}, min: {np.min(ne)}')
    ne_x = np.linspace(-extent_x, extent_x, dim[0])
    ne_y = np.linspace(-extent_y, extent_y, dim[1])
    ne_z = np.linspace(-extent_z, extent_z, dim[2])
    if probing_direction == 'x':
        extent = extent_x
    elif probing_direction == 'y':
        extent = extent_y
    elif probing_direction == 'z':
        extent = extent_z
    field = m.ScalarDomain(ne_x, ne_y, ne_z, extent = extent, probing_direction = probing_direction)       # modify probing direction HERE
    field.external_ne(ne)
    field.calc_dndr()
    field.clear_memory()
    del ne
    return field, extent

def system_solve(args):
    if args == None:
        print('No more rays, terminating.')
        os._exit(0)
    Np, beam_size, divergence, field, ne_extent = args
    ss = s.init_beam(Np=Np, beam_size=beam_size, divergence=divergence, ne_extent=ne_extent, beam_type='circular', probing_direction='y')       # modify probing direction HERE
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

    CustomManager.register('ScalarDomain', s.ScalarDomain)
    m = CustomManager()
    m.start()

    #get inputted variables 
    Np = int(float(sys.argv[1]))
    file_loc = str(sys.argv[2])
    output_loc = str(sys.argv[3])
    probe = 'z'
    field, probing_extent = calculate_field(file_loc = file_loc, manager = m, probing_direction = probe)
    cpu_count = mp.cpu_count()

    Np_ray_split = int(1e4) #split up rays
    number_of_splits = Np // Np_ray_split
    remaining_rays = Np % Np_ray_split
    #define beam parameters
    divergence =0.05e-3
    beam_size = 5e-3 # 5mm circular beam radius
    tasks = [(remaining_rays, beam_size, divergence, field, probing_extent)] if remaining_rays > 0 else []
    
    for i in range(number_of_splits):
        tasks.append((Np_ray_split, beam_size, divergence, field, probing_extent))
    
    print(len(tasks), ' tasks to complete, with ', cpu_count, ' cores')

    with mp.Pool(cpu_count) as pool:
        results = pool.map(system_solve, tasks)

    print('results obtained')

    sh_H = np.zeros_like(results[0][0])
    r_H = np.zeros_like(results[0][1])

    for sh_result, r_result in results:
        sh_H += sh_result
        r_H += r_result
    
    print('results summed')

    with open(output_loc + 'shadow.pkl', "wb") as filehandler:
        pickle.dump(sh_H, filehandler)
    with open(output_loc + 'refract.pkl', "wb") as filehandler:
        pickle.dump(r_H, filehandler)

    print('files written to ', output_loc)

