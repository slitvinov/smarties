SOCKET=$1
export OMP_NUM_THREADS=24

OPTIONS="-bpdx 32 -bpdy 32 -tdump 0.05 -CFL 0.1 -shape ellipse -semiAxisY 0.005 -semiAxisX 0.05 -rhoS 10.00 -lambda 1e5 -nu 0.001 -ypos 0.4 -xpos 0.6 -tend 0 -Socket ${SOCKET} -nStates 10 -nAction 1"
export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/users/sverma/usr/VTK-7.1.1/lib/:$LD_LIBRARY_PATH

echo $OPTIONS > opts.txt
echo `ls ..`

../CUP2D ${OPTIONS}


