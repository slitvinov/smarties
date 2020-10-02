# Install

euler:

    $ git clone -b f77 git@gitlab.ethz.ch:mavt-cse/smarties.git
    $ cd smarties
    $ (env2lmod && module load cmake gcc openmpi openblas && ./contrib/install.sh )
    $ (env2lmod && module load cmake gcc openmpi openblas && cd contrib/lib && make install)
    $ (env2lmod && module load cmake gcc openmpi openblas && cd contrib/cart_pole_f90 && make clean && make)
    $ (env2lmod && module load cmake gcc openmpi openblas && cd contrib/cart_pole_f77 && make clean && make)