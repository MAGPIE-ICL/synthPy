#!/bin/sh
#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=100:mpiprocs=100:mem=320gb
#PBS -j oe
cd '/rds/general/user/le322/home/synthPy'

echo 'sim'

module load anaconda3/personal

source activate MAGPIE_venv

mpiexec  python run_scripts/external_ray_trace.py 1e6 run_scripts/fields/radcollidingflows_3d_prp_Si.pvti ./output/2D_FlashSim_backup_