SOCKET=$1
export OMP_NUM_THREADS=24

OPTIONS="-bpdx 32 -bpdy 32 -tdump 0.1 -CFL 0.2 -shape ellipse -semiAxisY 0.02 -semiAxisX 0.2 -rhoS 10.00 -lambda 1e5 -nu 0.002 -ypos 0.25 -xpos 0.75 -tend 0 -Socket ${SOCKET}"
export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH

../simulation ${OPTIONS}


