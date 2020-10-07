An example of a turbulent channel flow with smarties using Nek5000

# Install smarties and libsmarties_f77.a

# Install Nek5000
    $ cd ~
    $ git clone --recursive https://github.com/Nek5000/Nek5000.git 
    $ export NEK5000_DIR=$HOME/Nek5000
    $ export PATH=${NEK5000_DIR}/bin:${PATH}
    $ cd ${NEK5000_DIR}/tools; ./maketools genmap; ./maketools genbox 
    $ cd ${NEK5000_DIR}/bin; 
    open makenek and uncomment "MPI=0" (turn off MPI for now) 	
    $ makenek -build-dep 

# Run case
    $ cd $SMARTIES_ROOT/contrib/nek5000 
    open submit.sh and change run directed as needed 
    (will create a directory in ${SCRATCH}/smarties/${RUN_DIR})
    $ ./submit.sh

