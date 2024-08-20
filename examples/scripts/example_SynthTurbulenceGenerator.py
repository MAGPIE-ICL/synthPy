'''
Script to iteratively generate plasmas of varying depths and power spectra.
Run with the following job: recommend 100gb ram, 1 processor
    
Author: Louis Evans
Reviewer: Stefano Merlini

    #!/bin/sh
    #PBS -l walltime=HH:MM:SS
    #PBS -l select=1:ncpus=N:mpiprocs=N:mem=Mgb
    #PBS -j oe
    cd '/rds/general/user/le322/home/synthPy'

    module load anaconda3/personal

    source activate MAGPIE_venv #activate venv

    python path/to/turb_gen.py
'''

import sys
sys.path.append('../synthPy/')              #import path/to/synthpy
import field_generator.gaussian3D as g3
import field_generator.gaussian2D as g2
import utils.power_spectrum as spectrum
import matplotlib.pyplot as plt
import numpy as np
import solver.minimal_solver as s
import sys

ps      =  [-5/3, -11/3, -20/3]     #list of powers
factors =  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]    #list of factors, where integration depth is factor*probing_extent*2

# Define turbulence parameters
l_max  =  1
l_min  =  0.01
extent =  5         # plasma spatial dimensions will be 2*extent x 2*extent x 2*extent*factor
res    =  312       # total dimension will be 2*res x 2*res x 2*res
k_min  =  2 * np.pi / l_max
k_max  =  2 * np.pi / l_min

# Average value and size of maximum perturbation
mean     = 1e25
max_pert = 9e24

def power_spectrum(k,a):
    return k**-a

def k41(k):
    return power_spectrum(k, p)

for p in ps:
    for factor in factors:
        field_3d = g3.gaussian3D(k41)
        ne = field_3d.fft(l_max, l_min, extent, res, factor)
        ne = mean + max_pert*ne 
        print(f'Plasma mean, max, and min: {np.mean(ne), np.max(ne), np.min(ne)}')
        # save as pvti
        x, y, z =  np.linspace(-extent, extent, 2*res), np.linspace(-extent, extent, 2*res), np.linspace(-extent*factor, extent*factor, int(2*res*factor))
        domain  =  s.ScalarDomain(x,y,z)
        domain.external_ne(ne)
        domain.export_scalar_field(fname = f'./Output/Fields/PS_{str(p)}/ne_{str(mean)}_{str(max_pert)}_depth_{factor}')    #insert path to save here
        del x
        del y
        del z
        del ne
