# Fetch

    $ git clone --depth 1 -b f77 git@gitlab.ethz.ch:mavt-cse/smarties.git
    $ cd smarties

# Install

generic

    $ make install
    $ (cd contrib/lib/f77 && make install)
    $ (cd contrib/lib/gslib && make INSTALL_ROOT=$HOME/.local)
    $ (cd contrib/cart_pole_f77 && make)

euler

    $ (env2lmod && module load gcc openmpi && make -j 12 install)
    $ (env2lmod && module load gcc openmpi && cd contrib/lib/f77 && make install)
    $ (env2lmod && module load gcc openmpi && cd contrib/cart_pole_f90 && make)
    $ (env2lmod && module load gcc openmpi && cd contrib/cart_pole_f77 && make)

daint

    $ module load daint-gpu
    $ module swap PrgEnv-cray PrgEnv-gnu
    $ make -j 12 install CXX=CC
    $ (cd contrib/lib/f77 && make install CXX=CC)
    $ (cd contrib/cart_pole_f90 && make CXX=CC FC=ftn LINK=CC MPI_EXTRA_LIB=)
    $ (cd contrib/cart_pole_f77 && make CXX=CC FC=ftn LINK=CC MPI_EXTRA_LIB=)
