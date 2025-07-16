#!/bin/sh
#PBS -l walltime=7:59:00
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -j oe

cd '/rds/general/user/sm5625/home/synthPy/'

echo 'loading packages'

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate testing

echo 'packages loaded successfully'

echo 'starting job'

for i in $(seq 1 2048);
do
    echo "Domain of $i"
    python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 512 -r 8192 | grep killed &> output.txt
done