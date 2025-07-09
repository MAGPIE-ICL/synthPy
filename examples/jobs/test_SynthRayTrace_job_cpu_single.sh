#!/bin/sh
#PBS -l walltime=3:59:00
#PBS -l select=1:ncpus=16:mem=40gb
#PBS -j oe

cd '/rds/general/user/sm5625/home/synthPy/'

echo 'loading packages'

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate testing

echo 'packages loaded successfully'

echo 'starting job'

python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 2048 -r 8192