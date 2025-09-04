#!/bin/sh
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=96gb:ngpus=1:gpu_type=A100
#PBS -j oe

cd '/rds/general/user/sm13118/home/dev/synthPy/'

echo 'loading packages'

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate gpu-env

echo 'packages loaded successfully'

echo 'starting job'

python -u evaluation/benchmarks/turbBoxLargeTest/updated.py -p "/rds/general/user/sm13118/home/dev/synthPy/src" -s "/rds/general/user/sm13118/projects/afosrturbulence/live/sm13118/sim/FLASH_4.8/Archive/RadMeshAblation/1200x750x500-double-CH-3rd-HLLC/xyz14_prp_CH_100pts_double_mesh_ug_3rd_5x100_10x75_12x100_cfl_0.5_8eV/radmeshablation_3d_prp_CH_ug_3rd_hdf5_plt_cnt_0228"