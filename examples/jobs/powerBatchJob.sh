#!/bin/sh
#PBS -l walltime=7:59:00
#PBS -l select=1:ncpus=120:mpiprocs=120:mem=920gb
#PBS -j oe
#PBS -J 1-6
cd '/rds/general/user/le322/home/synthPy'

echo 'powers!'

module load anaconda3/personal

source activate MAGPIE_venv


python run_scripts/high_res_trace_batch_spec.py 5e8 '/rds/general/user/le322/home/synthPy/fields/power_spec/power_' './output/power_spec/' ${PBS_ARRAY_INDEX}
