import sys
#sys.path.insert(0, '/home/administrator/Work/UROP_ICL_Internship/synthPy/src/simulator')
sys.path.insert(0, '/rds/general/user/sm5625/home/synthPy/src/simulator')     # import path/to/synthpy

import config
config.jax_init()

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--domain", type = int)
args = parser.parse_args()

n_cells = 512
if args.domain is not None:
    n_cells = args.domain

print("\nRunning job with a", n_cells, "domain.")
print("Predicted size of domain is:", ((n_cells / 1024)**3) * 32 / 8)

# define some extent, the domain should be distributed as +extent to -extent, does not need to be cubic
extent_x = 5e-3
extent_y = 5e-3
extent_z = 10e-3

lengths = 2 * np.array([extent_x, extent_y, extent_z])

import domain as d
import importlib
importlib.reload(d)

domain = d.ScalarDomain(lengths, n_cells) # B_on = False by default

domain.test_exponential_cos()