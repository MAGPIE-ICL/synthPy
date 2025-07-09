#!/bin/sh
#PBS -l walltime=3:59:00
#PBS -l select=1:ncpus=4:mem=96gb:ngpus=1:gpu_type=RTX6000
#PBS -j oe

module load tools/prod
module load jax/0.3.25-foss-2022a-CUDA-11.7.0

python -u examples/jobs/run_scripts/memory_edge_case.py -d 512
python -u examples/jobs/run_scripts/memory_edge_case.py -d 1024
python -u examples/jobs/run_scripts/memory_edge_case.py -d 10000
python -u examples/jobs/run_scripts/memory_edge_case.py -d 100000
python -u examples/jobs/run_scripts/memory_edge_case.py -d 103621
python -u examples/jobs/run_scripts/memory_edge_case.py -d 103622