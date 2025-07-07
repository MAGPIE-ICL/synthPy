#!/bin/sh

"""
Script to do synthetic interferogram with pvti files

Run with the following job script:
	#!/bin/sh
	#PBS -l walltime=HH:MM:SS
	#PBS -l select=1:ncpus=N:mpiprocs=N:mem=Mgb
	#PBS -j oe
	cd '/rds/general/user/sm5625/home/synthPy'

	module load anaconda3/personal

	source activate MAGPIE_venv #load venv

	mpiexec  -n <n_cpus> python run_scripts/interference_MPI.py <number of rays> <path/to/pvti> <output directory>
"""

#PBS -l walltime=HH:MM:SS
#PBS -l select=1:ncpus=N:mpiprocs=N:mem=Mgb
#PBS -j oe
cd '../'

mpiexec -n 8 python scripts/example_single_InterferogramMPI.py 1000 <path/to/pvti> ../examples_tests/results/