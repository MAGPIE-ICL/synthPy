#!/bin/sh
#PBS -l walltime=60:00:00
#PBS -l select=1:ncpus=48:mpiprocs=48:mem=720gb
#PBS -j oe
cd '/rds/general/user/le322/home/synthPy'

echo 'double mesh ray trace side on 1064nm'



source activate MAGPIE_venv

mpiexec python run_scripts/trace_pvti.py 1e7 1064e-9 'z' '/rds/general/user/le322/home/synthPy/output/fluid sim/double_mesh/double_mesh_006.pvti' './output/sideOn_1064_'

