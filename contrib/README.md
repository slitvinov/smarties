# Fetch

    $ git clone --depth 1 -b f77 git@gitlab.ethz.ch:mavt-cse/smarties.git
    $ cd smarties

# Install

generic

    $ make install
    $ (cd contrib/lib/f77 && make install)
    $ (cd contrib/example/f77 && make)

euler

    $ (env2lmod && module load gcc openmpi && make -j 12 install)
    $ (env2lmod && module load gcc openmpi && cd contrib/lib/f77 && make install)
    $ (env2lmod && module load gcc && cd contrib/lib/gslib && make -j 12 install)
    $ (env2lmod && module load gcc openmpi && cd contrib/example/f77 && make)
    $ (env2lmod && module load gcc && cd contrib/turbChannel/a/lib && make -j 12)
    $ (env2lmod && module load gcc && cd contrib/turbChannel/a && make)

daint

    $ module load daint-gpu
    $ module swap PrgEnv-cray PrgEnv-gnu
    $ make -j 12 install CXX=CC
    $ (cd contrib/lib/f77 && make install CXX=CC)
    $ (cd contrib/example/f77 && make CXX=CC FC=ftn LINK=CC MPI_EXTRA_LIB=)

contrib/turbChannel/dlopen

    $ cd
    $ git clone --depth 1 --recursive git@github.com:Nek5000/Nek5000.git
    $ git clone --depth 1 -b f77 git@github.com:slitvinov/smarties
    $ cd smarties
    $ . contrib/smarties.env
    $ MAKEFLAGS=-j12 ./contrib/install.sh
    $ cd contrib/turbChannel/dlopen
    $ MAKEFLAGS=-j12 MPI=0 FFLAGS='-Ofast -g -fPIC' CFLAGS='-Ofast -g -fPIC' ~/Nek5000/bin/nekconfig  -build-dep
    $ MPI=0 ~/Nek5000/bin/nekconfig
    $ make -f make/lib.mk -j12
    $ make -f make/bin.mk

    On login node
    $ ./main

    To submit
    $ bsub -n 8 -W 00:02 bash -c '. ~/.local/bin/smarties.env && mpirun ./main'
