#!/bin/sh
#PBS -l walltime=3:59:00

if [[ $(hostname) == "login-a" ]]; then
  #PBS -l select=1:ncpus=4:mem=16gb:ngpus=1:gpu_type=RTX6000
elif [[ $(hostname) == "login-ai.cx3.hpc.ic.ac.uk" ]]; then
  #PBS -l select=1:ncpus=4:mem=16gb:ngpus=1:gpu_type=a100
fi

#PBS -j oe

cd '/rds/general/user/sm5625/home/synthPy/'

echo 'loading packages'

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate testing

echo 'packages loaded successfully'

echo 'starting job'

python examples/jobs/run_scripts/test_SynthRayTrace.py &> test_SynthRayTrace_output.txt