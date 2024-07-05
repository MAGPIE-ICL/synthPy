#PBS -l select=1:ncpus=16:mem=64gb
#PBS -l walltime=1:59:0
export DISPLAY=localhost:0.0
echo $DISPLAY
 
module load paraview/5.10.0
mpiexec pvserver --reverse-connection --client-host=172.30.29.210


