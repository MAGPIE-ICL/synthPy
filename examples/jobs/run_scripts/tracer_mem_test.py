import numpy as np
import matplotlib.pyplot as plt
import vtk
import matplotlib.pyplot as plt
import sys

from collections import Counter
import linecache
import os
import tracemalloc

from vtk.util import numpy_support as vtk_np
#from sys import path.insert as insert_path

sys.path.insert(0, '/rds/general/user/sm5625/home/synthPy/src/simulator')

import config
config.jax_init()

# cwd is set to synthPy acc. to hpc

import beam as beam_initialiser
import diagnostics as diag
import domain as d
import propagator as p
import utils

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("\n\nTop %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("\n %s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024), end = "\n")

tracemalloc.start()

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
    [1, 100, 1e6, 1e7]
], dtype = np.int64)

count = 0
for i in parameters[0, :]:
    count += 1
    print("\n\n\n", count, "th trial:\n")#

    x = np.linspace(-extent_x, extent_x, i)
    y = np.linspace(-extent_y, extent_y, i)
    z = np.linspace(-extent_z, extent_z, i)

    lengths = 2 * np.array([extent_x, extent_y, extent_z])

    domain = d.ScalarDomain(lengths, np.array([i, i, i]))

    domain.test_exponential_cos()

    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)

    for j in parameters[1, :]:
        print("\n\n\n")
        print("Attempting:", i, "domain indices and", j, "rays.")

        beam = beam_initialiser.Beam(j, beam_size, divergence, ne_extent, probing_direction = probing_direction, wavelength = lwl, beam_type = beam_type)

        tracer = p.Propagator(domain, probing_direction = probing_direction, inv_brems = False, phaseshift = False)

        tracer.calc_dndr(lwl)

        try:
            final_rays = tracer.solve(beam.s0, jitted = True)

            print("\nCompleted ray trace in", np.round(tracer.duration, 3), "seconds.")

            snapshot = tracemalloc.take_snapshot()
            display_top(snapshot)

            '''
            schlierener = diag.Schlieren(tracer.Beam)
            schlierener.DF_solve()
            schlierener.histogram(bin_scale = 1, clear_mem = True)

            plt.imshow(schlierener.H, cmap = 'hot', interpolation = 'nearest', clim = (0.5, 1))
            '''
        except:
            continue