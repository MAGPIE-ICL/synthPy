#!/bin/sh
#PBS -l walltime=3:59:00
#PBS -l select=1:ncpus=4:mem=96gb:ngpus=1:gpu_type=RTX6000
#PBS -j oe

cd '/rds/general/user/sm5625/home/synthPy/'

echo 'loading packages'

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate testing

echo 'packages loaded successfully'

echo 'starting job'

#mkdir -p /rds/general/user/sm5625/home/synth_ray_trace
python -u examples/jobs/run_scripts/test_SynthRayTrace.py #&> "synth_ray_trace/test_SynthRayTrace-$(date +"%Y-%m-%d_%I:%M_%p").output"