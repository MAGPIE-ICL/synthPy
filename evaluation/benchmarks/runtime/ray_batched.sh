#!/bin/sh
#PBS -l walltime=71:59:00
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1:gpu_type=L40S
#PBS -j oe

cd '/rds/general/user/sm5625/home/synthPy/'

echo 'loading packages'

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate testing

echo 'packages loaded successfully'

echo 'starting job'

python -u evaluation/benchmarks/runtime/runtimes.py