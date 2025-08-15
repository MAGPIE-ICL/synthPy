#!/bin/bash
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=1:mem=50gb
#PBS -j oe

cd 'synthPy'

module load anaconda3/personal

source activate MAGPIE_venv #activate venv

python examples\notebooks\test_SynthRayTracer.ipynb