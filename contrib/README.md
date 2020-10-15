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
    $ (cd contrib/lib/f77 && make install CXX=CC CC=cc)
    $ (cd contrib/example/f77 && make CXX=CC FC=ftn LINK=CC MPI_EXTRA_LIB=)

# contrib/turbChannel/dlopen

    To deploy use contrib/deploy.sh. It can be fetch directly
    $ cd
    $ wget https://raw.githubusercontent.com/slitvinov/smarties/f77/contrib/deploy.sh
    $ rm -rf Nek5000 smarties .local
    $ MAKEFLAGS=-j bash deploy.sh

    To compile:
    $ . ~/.local/bin/smarties.env
    $ make -f make/lib.mk -j
    $ make -f make/bin.mk

    To submit
    $ bsub -n 8 -W 00:02 bash -c '. ~/.local/bin/smarties.env && mpirun ./main'
