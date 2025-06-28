#!/bin/sh
#PBS -l walltime=7:29:00
#PBS -l select=1:ncpus=80:mpiprocs=80:mem=720gb
#PBS -j oe
#PBS -J 1-37
cd '/rds/general/user/le322/home/synthPy'

echo 'extents! 1'

module load anaconda3/personal

source activate MAGPIE_venv


python run_scripts/extentsBatch.py 1e7 '/rds/general/user/le322/home/synthPy/fields/length_scale_full/1/' './output/length_scale/1/' ${PBS_ARRAY_INDEX}
