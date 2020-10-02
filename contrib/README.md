# Fetch

   $ git clone -b f77 git@gitlab.ethz.ch:mavt-cse/smarties.git
   $ cd smarties

# Install

generic:

   $ make install
   $ (cd contrib/lib && make install)
   $ (cd contrib/cart_pole_f77 && make)

euler:

    $ (env2lmod && module load cmake gcc openmpi openblas && ./contrib/install.sh )
    $ (env2lmod && module load gcc openmpi openblas && cd contrib/lib && make install)
    $ (env2lmod && module load gcc openmpi openblas && cd contrib/cart_pole_f90 && make)
    $ (env2lmod && module load gcc openmpi openblas && cd contrib/cart_pole_f77 && make)
