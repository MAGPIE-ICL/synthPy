#!/bin/sh
#PBS -l walltime=3:59:00
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -j oe

cd '/rds/general/user/sm5625/home/synthPy/'

echo 'loading packages'

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate testing

echo 'packages loaded successfully'

echo 'starting job'

python -u evaluation/MWE_debugging/complete_MWE_21_7_2025.py