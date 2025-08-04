#!/bin/sh
#PBS -l walltime=3:59:00
#PBS -l select=1:ncpus=8:mem=24gb
#PBS -j oe

cd '/rds/general/user/sm5625/home/synthPy/'

echo 'loading packages'

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate testing

echo 'packages loaded successfully'

echo 'starting job'

python -u evaluation/cpu_sharding_testing/interpolator_testing.py -d 512 -r 8192 -c 8
python -u evaluation/cpu_sharding_testing/interpolator_testing.py -d 640 -r 8192 -c 8
python -u evaluation/cpu_sharding_testing/interpolator_testing.py -d 768 -r 8192 -c 8
python -u evaluation/cpu_sharding_testing/interpolator_testing.py -d 896 -r 8192 -c 8
python -u evaluation/cpu_sharding_testing/interpolator_testing.py -d 1024 -r 8192 -c 8