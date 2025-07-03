#!/bin/sh
#PBS -l walltime=3:59:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -j oe

cd '/rds/general/user/sm5625/home/synthPy/'

echo 'loading packages'

#module load ~/miniforge3/envs/testing
#module load ~miniforge3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate testing
#source activate MAGPIE_venv

echo 'packages loaded successfully'

echo 'starting job'

python -u examples/jobs/run_scripts/tracer_mem_test.py