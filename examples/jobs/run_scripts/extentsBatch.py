import sys
sys.path.append('/rds/general/user/le322/home/synthPy/')
import numpy as np
import pickle
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from multiprocessing import Process
import field_generator.gaussian1D as g1
import field_generator.gaussian2D as g2
import field_generator.gaussian3D as g3
import solver.full_solver as s
import solver.rtm_solver as rtm
import matplotlib.pyplot as plt
import gc
import vtk
import os
from vtk.util import numpy_support as vtk_np

class CustomManager(BaseManager):
    pass

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
        img = v.reshape(vec, order="F")[0:dim[0]-1, 0:dim[1]-1, 0:dim[2]-1, :]
    else:
        img = v.reshape(vec, order="F")[0:dim[0]-1, 0:dim[1]-1, 0:dim[2]-1]
    dim = img.shape
    return img, dim, spacing

def calculate_field(file_loc, manager):
    ne, dim, spacing = pvti_readin(file_loc)
    extent_x = ((dim[0]*spacing[0])/2)*1e-3
    extent_y = ((dim[1]*spacing[1])/2)*1e-3
    extent_z = ((dim[2]*spacing[2])/2)*1e-3

    print(f'extents: {extent_x, extent_y, extent_z}')
    ne_x = np.linspace(-extent_x, extent_x, dim[0])
    ne_y = np.linspace(-extent_y, extent_y, dim[1])
    ne_z = np.linspace(-extent_z, extent_z, dim[2])
    field = m.ScalarDomain(ne_x, ne_y, ne_z, extent_z, probing_direction = 'z')       # modify probing direction HERE
    field.external_ne(ne)
    # field.test_null()
    field.calc_dndr()
    field.clear_memory()
    del ne
    return field, extent_z, extent_x

def system_solve(args):
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

    CustomManager.register('ScalarDomain', s.ScalarDomain)
    m = CustomManager()
    m.start()




    Np = int(float(sys.argv[1]))

    file_loc = str(sys.argv[2])

    output_loc = str(sys.argv[3])

    index = int(int(sys.argv[4]) - 1)

    extents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.5, 1.7, 1.8, 1.9, 2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]
# 37
    extent = str(extents[index])

    file_loc += extent
    file_loc += '.pvti'

    print('current file location = ', file_loc)

    field, extent_z, extent_x = calculate_field(file_loc = file_loc, manager = m)

    Np_ray_split = int(5e5)

    number_of_splits = Np // Np_ray_split

    remaining_rays = Np % Np_ray_split

    divergence =0.05e-3

    probing_extent = extent_z #change these values accordingly

    beam_size = extent_x

    tasks = [(remaining_rays, beam_size, divergence, field, extent_z)] if remaining_rays > 0 else []
    cpu_count = 80
    


    for i in range(number_of_splits):
        tasks.append((Np_ray_split, beam_size, 0.05e-3, field, probing_extent))
    
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

    with open(output_loc + f'{extent}_shadow.pkl', "wb") as filehandler:
        pickle.dump(sh_H, filehandler)
    with open(output_loc + f'{extent}_refract.pkl', "wb") as filehandler:
        pickle.dump(r_H, filehandler)

    print('files written to ', output_loc)

