SOCKET=$1
export OMP_NUM_THREADS=12
OPTIONS="-bpdx 8 -bpdy 8 -tdump 0.0 -shape blowfish -radius 0.16 -nu 0.0004 -tend 0 -rhoS 0.5"

export LD_LIBRARY_PATH=/cluster/home/novatig/VTK-7.1.0/Build/lib/:$LD_LIBRARY_PATH

echo $OPTIONS > opts.txt
echo `ls ..`

../blowfish ${OPTIONS}


