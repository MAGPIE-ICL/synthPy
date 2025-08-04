#!/bin/sh
#PBS -l walltime=3:59:00
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=RTX6000
#PBS -j oe

cd '/rds/general/user/sm5625/home/synthPy/'

echo 'loading packages'

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate testing

echo 'packages loaded successfully'

echo 'starting job'

#mkdir -p /rds/general/user/sm5625/home/synth_ray_trace
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 896 -r 8192
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 960 -r 8192
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 992 -r 8192
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1008 -r 8192
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1024 -r 8192
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1152 -r 8192
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1280 -r 8192
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1408 -r 8192
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1536 -r 8192