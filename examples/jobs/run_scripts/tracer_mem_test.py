import numpy as np
import matplotlib.pyplot as plt
import vtk
import matplotlib.pyplot as plt
import sys

from vtk.util import numpy_support as vtk_np
#from sys import path.insert as insert_path

sys.path.insert(0, '/rds/general/user/sm5625/home/synthPy/src/simulator')

# cwd is set to synthPy acc. to hpc

import beam as beam_initialiser
import diagnostics as diag
import domain as d
import propagator as p
import utils

extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

probing_extent = extent_z
probing_direction = 'z'

wl = 1064e-9

divergence = 5e-5
beam_size = extent_x
ne_extent = probing_extent
beam_type = 'circular'

parameters = np.array([
    [128, 256, 512, 1024],
    [1, 100, 1e6, 1e7]
], dtype = np.int64)

for i in parameters[0, :]:
    x = np.linspace(-extent_x, extent_x, i)
    y = np.linspace(-extent_y, extent_y, i)
    z = np.linspace(-extent_z, extent_z, i)

    lengths = 2 * np.array([extent_x, extent_y, extent_z])

    domain = d.ScalarDomain(lengths, i)

    domain.test_exponential_cos()

    for j in parameters[1, :]:
        initial_rays = beam_initialiser.Beam(j, beam_size, divergence, ne_extent, probing_direction, wl, beam_type)

        tracer = p.Propagator(domain, initial_rays, inv_brems = False, phaseshift = False)

        tracer.calc_dndr()

        try:
            final_rays = tracer.solve(parallelise = True, jitted = True)

            print("\nCompleted ray trace in", np.round(tracer.duration, 3), "seconds.")

            schlierener = diag.Schlieren(tracer.Beam)
            schlierener.DF_solve()
            schlierener.histogram(bin_scale = 1, clear_mem = True)

            plt.imshow(schlierener.H, cmap = 'hot', interpolation = 'nearest', clim = (0.5, 1))
        except:
            continue