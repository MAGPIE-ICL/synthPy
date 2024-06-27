#!/bin/sh
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=48:mpiprocs=48:mem=256gb
#PBS -j oe
cd '/rds/general/user/le322/home/synthPy'

pwd

module load anaconda3/personal

source activate MAGPIE_venv


mpiexec python turbulence_MPI.py 1e7