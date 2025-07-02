#!/bin/sh
#PBS -l walltime=3:59:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=A100
#PBS -j oe

cd '/rds/general/user/sm5625/synthPy'

echo 'loading packages'

module load anaconda3/testing
source activate MAGPIE_venv

echo 'starting job'

python run_scripts/tracer_mem_test.py