#!/bin/sh
#PBS -l walltime=7:59:00
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=128gb
#PBS -j oe
cd '/rds/general/user/le322/home/synthPy'

echo 'generating turbulence'

module load anaconda3/personal

source activate MAGPIE_venv


python run_scripts/turb_gen.py 10