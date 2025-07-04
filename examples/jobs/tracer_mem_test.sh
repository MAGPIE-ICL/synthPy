#!/bin/sh
#PBS -l walltime=3:59:00

if [[ $(hostname) == "login-a" ]]; then
  #PBS -l select=1:ncpus=4:mem=96gb:ngpus=1:gpu_type=RTX6000
elif [[ $(hostname) == "login-ai.cx3.hpc.ic.ac.uk" ]]; then
  #PBS -l select=1:ncpus=4:mem=96gb:ngpus=1:gpu_type=A100

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

# -u to prevent line buffering?
python examples/jobs/run_scripts/tracer_mem_test.py &> tracer_mem_test_output.txt