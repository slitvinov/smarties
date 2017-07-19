SOCKET=$1
export OMP_NUM_THREADS=48

OPTIONS="-bpdx 32 -bpdy 32 -tdump 0.05 -CFL 0.1 -shape ellipse -semiAxisY 0.03125 -semiAxisX 0.003125 -rhoS 10.00 -lambda 1e5 -nu 0.001 -ypos 0.3 -xpos 0.5 -tend 0 -Socket ${SOCKET} -nStates 10 -nAction 1"
export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH

echo $OPTIONS > opts.txt
echo `ls ..`

../CUP2D ${OPTIONS}


