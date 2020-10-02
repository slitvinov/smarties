# Fetch

   $ git clone -b f77 git@gitlab.ethz.ch:mavt-cse/smarties.git
   $ cd smarties

# Install

generic:

   $ make install
   $ (cd contrib/lib && make install)
   $ (cd contrib/cart_pole_f77 && make)

euler:

    $ (env2lmod && module load gcc openmpi && make -j 12 install)
    $ (env2lmod && module load gcc openmpi && cd contrib/lib && make install)
    $ (env2lmod && module load gcc openmpi && cd contrib/cart_pole_f90 && make)
    $ (env2lmod && module load gcc openmpi && cd contrib/cart_pole_f77 && make)
