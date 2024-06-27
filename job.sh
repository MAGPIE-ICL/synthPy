#!/bin/sh
#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=4:mpiprocs=4:mem=32gb
#PBS -j oe
cd synthPy

module load anaconda3/personal

source activate MAGPIE_venv


mpiexec python turbulence_MPI.py 1e7