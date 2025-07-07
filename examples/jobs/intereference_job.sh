#!/bin/sh
#PBS -l walltime=7:59:00
#PBS -l select=1:ncpus=120:mpiprocs=120:mem=920gb
#PBS -j oe
cd '/rds/general/user/le322/home/synthPy'

echo 'interference sim'

module load anaconda3/personal

source activate MAGPIE_venv

mpiexec python run_scripts/interference_MPI.py 1e6 1e6 'output/fluid sim/double_mesh/double_mesh_003.pvti' ./output/colliding_mesh_endOnInt003


