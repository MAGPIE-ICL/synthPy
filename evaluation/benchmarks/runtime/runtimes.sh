#!/bin/sh
#PBS -l walltime=7:59:00
#PBS -l select=1:ncpus=8:mem=16gb:ngpus=1:gpu_type=RTX6000
#PBS -j oe

cd '/rds/general/user/sm5625/home/synthPy/'

echo 'loading packages'

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate testing

echo 'packages loaded successfully'

echo 'starting job'

python -u evaluation/benchmarks/runtime/runtimes.py -d 128 -c 8
python -u evaluation/benchmarks/runtime/runtimes.py -d 256 -c 8
python -u evaluation/benchmarks/runtime/runtimes.py -d 384 -c 8
python -u evaluation/benchmarks/runtime/runtimes.py -d 512 -c 8
python -u evaluation/benchmarks/runtime/runtimes.py -d 640 -c 8
python -u evaluation/benchmarks/runtime/runtimes.py -d 768 -c 8