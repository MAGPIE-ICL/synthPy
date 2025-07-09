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
#python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 512 -r 1000
#python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 512 -r 10000000
#python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 513 -r 1000
#python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 528 -r 1000
#python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 544 -r 1000
#python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 576 -r 1000
#python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 640 -r 1000
#python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 704 -r 1000
#python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 768 -r 1000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1024 -r 1 #&> "synth_ray_trace/test_SynthRayTrace-$(date +"%Y-%m-%d_%I:%M_%p").output"
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1023 -r 1
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1025 -r 1