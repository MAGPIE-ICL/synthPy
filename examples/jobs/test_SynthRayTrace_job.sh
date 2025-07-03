#!/bin/sh
#PBS -l walltime=3:59:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -j oe

cd '/rds/general/user/sm5625/home/synthPy/'

echo 'loading packages'

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate testing

echo 'packages loaded successfully'

echo 'starting job'

jupyter nbconvert --execute --to test_SynthRayTrace_job_output.ipynb examples/notebooks/test_SynthRayTracer.ipynb