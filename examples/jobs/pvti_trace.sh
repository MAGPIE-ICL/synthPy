#!/bin/sh
#PBS -l walltime=7:59:00
#PBS -l select=1:ncpus=50:mpiprocs=50:mem=512gb
#PBS -j oe
cd '/rds/general/user/le322/home/synthPy'

echo 'jerry: 1e6'

module load anaconda3/personal

source activate MAGPIE_venv


python run_scripts/high_res_trace.py 1e6 '/rds/general/user/le322/home/synthPy/x86_rnec-64.pvti' ./output/jerry_trace
