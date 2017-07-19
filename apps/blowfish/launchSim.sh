SOCKET=$1
export OMP_NUM_THREADS=12

OPTIONS="-bpdx 16 -bpdy 16 -tdump 0.0 -CFL 0.1 -shape blowfish -radius 0.125 -lambda 1e5 -nu 0.0001 -ypos 0.5 -xpos 0.5 -tend 0 -Socket ${SOCKET} -nStates 8 -nAction 2"

export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH

echo $OPTIONS > opts.txt
echo `ls ..`

../CUP2D ${OPTIONS}


