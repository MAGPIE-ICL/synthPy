import numpy as np
import matplotlib.pyplot as plt
import vtk
import matplotlib.pyplot as plt
import sys

from vtk.util import numpy_support as vtk_np
#from sys import path.insert as insert_path

sys.path.insert(0, '/rds/general/user/sm5625/home/synthPy/src/simulator')

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

lwl = 1064e-9

divergence = 5e-5
beam_size = extent_x
ne_extent = probing_extent
beam_type = 'circular'

parameters = np.array([
    [128, 256, 512, 1024],
    [1, 100, 1e6, 1e8]
], dtype = np.int64)

runtime = np.array((2, len(parameters[0, :]), len(parameters[1, :])))

results_count = np.array((2, len(parameters[0, :]), len(parameters[1, :])))

for i in parameters[0, :]:
    x = np.linspace(-extent_x, extent_x, i)
    y = np.linspace(-extent_y, extent_y, i)
    z = np.linspace(-extent_z, extent_z, i)

    lengths = 2 * np.array([extent_x, extent_y, extent_z])

    domain = d.ScalarDomain(lengths, i)

    domain.test_exponential_cos()

    for j in parameters[1, :]:
        beam = beam_initialiser.Beam(j, beam_size, divergence, ne_extent, probing_direction = probing_direction, wavelength = lwl, beam_type = beam_type)

        tracer_serialised = p.Propagator(domain, beam.s0, probing_direction = probing_direction, inv_brems = False, phaseshift = False)
        tracer_serialised.calc_dndr(lwl)

        tracer_parallelised = p.Propagator(domain, beam.s0, probing_direction = probing_direction, inv_brems = False, phaseshift = False)
        tracer_parallelised.calc_dndr(lwl)

        try:
            tracer_serialised.solve(parallelise = False, jitted = False)

            runtime[0, i, j] = tracer_serialised.duration
            print("\nCompleted serialised ray trace in", np.round(tracer_serialised.duration, 3), "seconds.")

            tracer_parallelised.solve(parallelise = True, jitted = True)

            runtime[1, i, j] = tracer_serialised.duration
            print("\nCompleted parallelised ray trace in", np.round(tracer_parallelised.duration, 3), "seconds.")

            results_count[0, i, j] = tracer_serialised.rf[0, :].size
            results_count[1, i, j] = tracer_parallelised.rf[0, :].size

            print("Expected", parameters[1, j], "rays, ended up with:")
            print("solve_ivp result:", results_count[0, i, j])
            print("diffrax result:", results_count[1, i, j])

            diff = np.array(tracer_serialised.rf[:, :] - tracer_parallelised.rf[0, :])
            diff[diff < 1e-7] = None
            print("Difference between solve_ivp and diffrax results (+ means solve_ivp > diffrax):")
            print(diff)

            '''
            schlierener_serialised = diag.Schlieren(tracer_serialised.Beam)
            schlierener_serialised.DF_solve()
            schlierener_serialised.histogram(bin_scale = 1, clear_mem = True)

            schlierener_parallelised = diag.Schlieren(tracer_parallelised.Beam)
            schlierener_parallelised.DF_solve()
            schlierener_parallelised.histogram(bin_scale = 1, clear_mem = True)

            plt.imshow(schlierener_serialised.H, cmap = 'hot', interpolation = 'nearest', clim = (0.5, 1))

            plt.imshow(schlierener_parallelised.H, cmap = 'hot', interpolation = 'nearest', clim = (0.5, 1))
            '''
        except:
            continue

