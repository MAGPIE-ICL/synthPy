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

python -u evaluation/cpu_sharding_testing/cpu_sharded_CWE_25_7_2025.py -d 128 -r 8192 -s "False" -c 8
python -u evaluation/cpu_sharding_testing/cpu_sharded_CWE_25_7_2025.py -d 128 -r 8192 -s "True" -c 8

python -u evaluation/cpu_sharding_testing/cpu_sharded_CWE_25_7_2025.py -d 256 -r 8192 -s "False" -c 8
python -u evaluation/cpu_sharding_testing/cpu_sharded_CWE_25_7_2025.py -d 256 -r 8192 -s "True" -c 8

python -u evaluation/cpu_sharding_testing/cpu_sharded_CWE_25_7_2025.py -d 512 -r 8192 -s "False" -c 8
python -u evaluation/cpu_sharding_testing/cpu_sharded_CWE_25_7_2025.py -d 512 -r 8192 -s "True" -c 8

python -u evaluation/cpu_sharding_testing/cpu_sharded_CWE_25_7_2025.py -d 1024 -r 8192 -s "False" -c 8
python -u evaluation/cpu_sharding_testing/cpu_sharded_CWE_25_7_2025.py -d 1024 -r 8192 -s "True" -c 8