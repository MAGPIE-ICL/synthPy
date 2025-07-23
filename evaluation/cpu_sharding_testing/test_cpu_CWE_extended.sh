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

python -u evaluation/cpu_sharding_testing/cpu_sharded_testing_23_7_2025.py -d 512 -r 8192 -c 8
python -u evaluation/cpu_sharding_testing/cpu_sharded_testing_23_7_2025.py -d 640 -r 8192 -c 8
python -u evaluation/cpu_sharding_testing/cpu_sharded_testing_23_7_2025.py -d 768 -r 8192 -c 8
python -u evaluation/cpu_sharding_testing/cpu_sharded_testing_23_7_2025.py -d 896 -r 8192 -c 8
python -u evaluation/cpu_sharding_testing/cpu_sharded_testing_23_7_2025.py -d 1024 -r 8192 -c 8
python -u evaluation/cpu_sharding_testing/cpu_sharded_testing_23_7_2025.py -d 1152 -r 8192 -c 8
python -u evaluation/cpu_sharding_testing/cpu_sharded_testing_23_7_2025.py -d 1280 -r 8192 -c 8
python -u evaluation/cpu_sharding_testing/cpu_sharded_testing_23_7_2025.py -d 1408 -r 8192 -c 8
python -u evaluation/cpu_sharding_testing/cpu_sharded_testing_23_7_2025.py -d 1536 -r 8192 -c 8