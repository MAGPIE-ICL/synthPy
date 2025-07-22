#!/bin/sh
#PBS -l walltime=3:59:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -j oe

cd '/rds/general/user/sm5625/home/synthPy/'

echo 'loading packages'

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate testing

echo 'packages loaded successfully'

echo 'starting job'

python -u evaluation/MWE_debugging/complete_MWE_21_7_2025.py -d 128
python -u evaluation/MWE_debugging/complete_MWE_21_7_2025.py -d 256
python -u evaluation/MWE_debugging/complete_MWE_21_7_2025.py -d 512
python -u evaluation/MWE_debugging/complete_MWE_21_7_2025.py -d 768
python -u evaluation/MWE_debugging/complete_MWE_21_7_2025.py -d 1024