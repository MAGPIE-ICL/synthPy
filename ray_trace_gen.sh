#!/bin/sh
#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=100:mpiprocs=100:mem=320gb
#PBS -j oe
cd '/rds/general/user/le322/home/synthPy'

echo '0.1, 2.5, 73'

module load anaconda3/personal

source activate MAGPIE_venv

mpiexec  python run_scripts/external_ray_trace.py 1e6 run_scripts/fields/2D_gen_mod_0.1_2.5_73.pvti ./output/2D_gen_0.1_2.5_73_testing_backup_