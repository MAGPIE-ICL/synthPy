#!/bin/sh
#PBS -l walltime=7:59:00
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=A100
#PBS -j oe

cd '/rds/general/user/sm5625/home/synthPy/'

echo 'loading packages'

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate testing

echo 'packages loaded successfully'

echo 'starting job'

#mkdir -p /rds/general/user/sm5625/home/synth_ray_trace
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 512 -r 1
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 640 -r 1
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 768 -r 1
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 896 -r 1
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1024 -r 1 #&> "synth_ray_trace/test_SynthRayTrace-$(date +"%Y-%m-%d_%I:%M_%p").output"
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1152 -r 1
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1280 -r 1
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1408 -r 1
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1536 -r 1

python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 512 -r 10000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 640 -r 10000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 768 -r 10000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 896 -r 10000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1024 -r 10000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1152 -r 10000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1280 -r 10000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1408 -r 10000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1536 -r 10000

python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 512 -r 10000000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 640 -r 10000000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 768 -r 10000000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 896 -r 10000000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1024 -r 10000000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1152 -r 10000000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1280 -r 10000000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1408 -r 10000000
python -u examples/jobs/run_scripts/test_SynthRayTrace.py -d 1536 -r 10000000