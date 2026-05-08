import sys
import os

try:
    current_file = os.path.realpath(__file__)
except NameError:
    # __file__ not defined
    if sys.argv[0]:  # Might still work in some IDEs
        current_file = os.path.realpath(sys.argv[0])
    else:
        # Fallback to current working directory (e.g. Jupyter)
        current_file = os.getcwd()

#top_level_path = resolve_path(str(os.path.dirname(os.path.realpath(__file__))) + "/../")
top_level_path = os.path.abspath(os.path.join(os.path.dirname(current_file), '../..'))
print("Setting top level path for imports:", top_level_path)

# Ensure top-level path is in sys.path
if top_level_path not in sys.path:
    # makes sure top level directory path is present in system so that relative imports work
    sys.path.insert(0, top_level_path)

import legacy.full_solver as s
import utils.handle_filetypes as utilIO

importlib.reload(fs)
importlib.reload(rtm)
importlib.reload(utilIO)

#load hdf
ne, dim, spacing = utilIO.hdf_readin(str(file_loc))

# multiply domain to match real size experimental target
multi_x = 0 # get multiplier x
multi_y = 0 # get multiplier y
multi_z = 0 # get multiplier z

for i in range(multi_x):
    ne = np.append(ne, ne, axis = 0)
for j in range(multi_y):
    ne = np.append(ne, ne, axis = 1)
for k in range(multi_z):
    ne = np.append(ne, ne, axis = 2)

dim  = ne.shape

extent_x = ((dim[0]*spacing[0].v)/2) * 1e-2
extent_y = ((dim[1]*spacing[1].v)/2) * 1e-2
extent_z = ((dim[2]*spacing[2].v)/2) * 1e-2

ne_x = np.linspace(-extent_x,extent_x, dim[0])
ne_y = np.linspace(-extent_y,extent_y, dim[1])
ne_z = np.linspace(-extent_z,extent_z, dim[2])

print(f'extent x: {extent_x}')
print(f'extent y: {extent_y}')
print(f'extent z: {extent_z}')

print(f'dim x: {dim[0]}')
print(f'dim y: {dim[1]}')
print(f'dim z: {dim[2]}')

# define beam parameters
wl = 1064e-9
probing_direction = 'z'
beam_size = [extent_x, extent_y]    # beam radius
probing_extent = extent_z
ne_extent = probing_extent  # so the beam knows where to initialise initial positions
divergence = 0.05e-3

field = s.ScalarDomain(ne_x, ne_y, ne_z, extent = ne_extent, probing_direction = probing_direction)
field.external_ne(ne.v*1e6)
field.calc_dndr(lwl = wl)
del ne_x
del ne_y
del ne_z
del ne