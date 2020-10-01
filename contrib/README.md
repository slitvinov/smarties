# Install

    $ git clone -b f77 git@gitlab.ethz.ch:mavt-cse/smarties.git --recursive
    $ cd smarties
    $ (env2lmod && module load cmake gcc openmpi openblas python && ./contrib/install.sh )
    $ (env2lmod && module load cmake gcc openmpi openblas python && cd contrib/lib && make install)
    $ (env2lmod && module load cmake gcc openmpi openblas python && cd contrib/cart_pole_f90 && make clean && make)
    $ (env2lmod && module load cmake gcc openmpi openblas python && cd contrib/cart_pole_f77 && make clean && make)