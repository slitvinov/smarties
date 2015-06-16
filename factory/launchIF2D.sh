#!/bin/bash

EXECNAME=rl
NPROCESS=$1
NTHREADS=48
NPROCESSORS=$(($NPROCESS*$NTHREADS))
RUNFOLDER=$2
WCLOCK=$3

BASEPATH="/cluster/scratch_xp/public/laurentm/MRAG2D/"
FACTORY="factoryIF2D"  # file in ../factory/ (in the learning part)

SETTINGS=

## ---------- Dmitry setting -------- ##
SETTINGS+=" --config config.ini -- "
## ---------------------------------- ##


# STUDY
#SETTINGS+=" -study FLUID_MEDIATED_INTERACTIONS"
#SETTINGS+=" -study SMART_INTERACTIONS_LAU"
#SETTINGS+=" -study SMART_INTERACTIONS"
SETTINGS+=" -study FLUID_MEDIATED_LAU"
#SETTINGS+=" -study LEARNING_DIPOLE"

# CPU
SETTINGS+=" -nthreads ${NTHREADS}" # number of threads

# FLOW PARAMETERS
SETTINGS+=" -tend 1000.0" # sim time
SETTINGS+=" -lambda 1e4" # penalization factor
SETTINGS+=" -re 550" # Reynolds number
SETTINGS+=" -uinfx 0.0"	# freestream velocity in x
SETTINGS+=" -uinfy 0.0"	# freestream velocity in y

# TIME STEP CONSTRAINTS
SETTINGS+=" -lcfl 0.1" # Lagrangian CFL condition based on strain (advection)
SETTINGS+=" -cfl 0.8" # standard CFL condition on dt, based on max velocity (advection)
SETTINGS+=" -fc 0.5" # Fourier coefficeint condition on dt (diffusion)
SETTINGS+=" -ramp 100" # condition on dt to ramp up at sim initialization

# OBJECT IN FLOW
SETTINGS+=" -mollfactor 2" # number of grid points to smooth the characteristic function
SETTINGS+=" -obstacle heterogeneous" # type of obstacle
SETTINGS+=" -factory ${FACTORY}" # which factory file to use
SETTINGS+=" -sharp 1" # no idea!

# OTHER
SETTINGS+=" -particles 1" # true = use particles
SETTINGS+=" -hilbert 0" # true = use hilbert ordering for grid (?)
SETTINGS+=" -usekillvort 1" # true = kill vorticity at right boundary, fluid mediated interactions only --> other studies: set to 0
SETTINGS+=" -killvort 1" # range for vorticity killing, comment out if usekillvort==0
SETTINGS+=" -useoptimizer 0" # true: fitness function for CMA-ES is calculated, false: fitness function not calculated, fluid mediated interactions only --> other studies: set to 0
SETTINGS+=" -tbound 40" # lower bound of drag coefficient integration (for fitness function), comment out if useoptimizer == 0

# MRAG
SETTINGS+=" -bpd 32" # initial number of blocks per dimension
SETTINGS+=" -lmax 7" # max levels of refinement, N blocks 2^lmax, each block 32x32 grid pts.
SETTINGS+=" -jump 2" # max jump in grid level between any two blocks
SETTINGS+=" -rtol 1e-4" # threshold for refinement, smaller = less
SETTINGS+=" -ctol 1e-6" # threshold for compression, smaller = less, should be less than rtol
SETTINGS+=" -adaptfreq 5" # number of steps between refinement and compression
SETTINGS+=" -refine-omega-only 0" # true = use omega only for refine, false = use velocity too
SETTINGS+=" -rio free_frame" # controls whether refinement occurs everywhere, interior blocks, or at outlet or inlet blocks
SETTINGS+=" -uniform 0" # true = uniform resolution, no refinement/compression

# FMM
SETTINGS+=" -fmm velocity" # fmm (fast multipole method) use to determine velocity
SETTINGS+=" -fmm-potential 0" # fmm used for the potential correction in shape changing geometries 
SETTINGS+=" -fmm-theta 0.5" # fmm factor, 0 = N^2, 1 = fast, inaccurate
SETTINGS+=" -core-fmm sse" # use sse for fmm (?)
SETTINGS+=" -fmm-skip 0" # skip fmm (?)

## FTLE
SETTINGS+=" -tStartFTLE 0.0"
SETTINGS+=" -tEndFTLE 0.0"
SETTINGS+=" -tStartEff 0.0"
SETTINGS+=" -tEndEff 0.0"

# ---- Needed but no USED --- #
# REINFORCEMENT LEARNING
SETTINGS+=" -lr 0.00" # learning rate, 0 = no learning/memory
SETTINGS+=" -gamma 0.95" # future action value, 0 = don't look forward, 1 look infinitely forward
SETTINGS+=" -greedyEps 0.01" # randomness in actions, fraction of actions chosen at random
SETTINGS+=" -shared 1" # true = everyone updates and shares a general polic
# ---- #

# LAUFISH
SETTINGS+=" -adaptvel 1" # true = Set Uinf[0]=-Ufish_x
SETTINGS+=" -learning 1" # true = Learn (act, state, reward)

SETTINGS+=" -xpos 0.5" # grid offset for intialized objects in x
SETTINGS+=" -ypos 0.55" # grid offset for intialized objects
SETTINGS+=" -D 0.025" # body/system size
SETTINGS+=" -d 1.0" # body size wrt system size
SETTINGS+=" -xm 0.0" # body xpos wrt global xpos (dimensionless with d)
SETTINGS+=" -ym 0.0" # body ypos wrt global ypos (dimensionless with d)
SETTINGS+=" -tau 1.0"
SETTINGS+=" -angle 0.0"
SETTINGS+=" -T 1" # swimming frequency

# # REINFORCEMENT LEARNING (Mattia)
# SETTINGS+=" -lr 0.00" # learning rate, 0 = no learning/memory
# SETTINGS+=" -gamma 0.95" # future action value, 0 = don't look forward, 1 look infinitely forward
# SETTINGS+=" -greedyEps 0.01" # randomness in actions, fraction of actions chosen at random
# SETTINGS+=" -shared 1" # true = everyone updates and shares a general policy

# PF
#SETTINGS+=" -isControlled 1" # agents take actions
#SETTINGS+=" -smooth 0" # smooths action over a period of time, otherwise discrete change
##SETTINGS+=" -savefreq 100" # frequency to dump into data (writes a ton of data if state space is large!! for animations i use 100)
#SETTINGS+=" -learnDump 100000" # frequency to dump learning information (reward, etc.)
#SETTINGS+=" -learningTime 100000" # amount of time specified to learn, total learning time is nLearningLevels*learningTime 
#SETTINGS+=" -nLearningLevels 1" # number of levels to progressively halve the lr, e.g. lr = 0.01 for first learning interval, lr = 0.005 for second, lr = 0.0025 for third, ...
#SETTINGS+=" -fitnessTime 0" # time to average fitness function over
#SETTINGS+=" -navg 1" # number of runs to average the fitness over after learningTime*nLearningLevels
#SETTINGS+=" -fitnessSaveFreq 100000" # change to very large if dumping less, then it only saves at the end of each averaging stage
#SETTINGS+=" -fitnessBuffer 1.5" # take (FINTESSBUFFER - 1)% longer to get rid of transient after a refresh
#SETTINGS+=" -individualFitness true" # compute fitness for each dipole
#SETTINGS+=" -exploitation true" # if true, the learning rate is turned to zero after learning time (and possibly greedyEps, check code!)

# OPTIONAL SETTINGS
SETTINGS+=" -fitselection 1"
SETTINGS+=" -islabframe 1"
SETTINGS+=" -isUsingTracers 0"


# I/O
SETTINGS+=" -vtu 0"	# dump vtu/vtk file (slow)
SETTINGS+=" -dumpfreq 250" # (vtu) number of dumps per sim time, dumps are on the dot 
SETTINGS+=" -savefreq 1000"	# (restart) frequency to save a full restart of simulation (super slow)

RESTART=" -restart 0" # restart a simulation stopped 
RESTARTPOLICY=" -restartPolicy 1" # create a new simulation with a given policy (the policy to load has to be in ../factory/policy_backup)
TIMES=1 # Job chaining


OPTIONS=${SETTINGS}${RESTART}${RESTARTPOLICY}


export LD_LIBRARY_PATH=/cluster/work/infk/wvanrees/apps/TBB/tbb42_20140122oss/build/linux_intel64_gcc_cc4.7.2_libc2.12_kernel2.6.32_release/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cluster/work/infk/cconti/VTK5.8_gcc/lib/vtk-5.8/:$LD_LIBRARY_PATH


if [ "${RESTART}" = " -restart 1" ]; then
    echo "---- launch.sh >> Restart ----"
    export OMP_NUM_THREADS=${NTHREADS}
    # rm ../run/*.txt
    # cd ../run
    # cp ../makefiles/config.ini ../run/
    bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=48] -sp 100 -W ${WCLOCK} mpirun -np ${NPROCESS} --mca btl tcp,self -pernode  ./${EXECNAME} ${OPTIONS}
fi

if [ "${RESTART}" = " -restart 0" ]; then
    if [[ "$(hostname)" == *hpc-net.ethz.ch ]]; then
        echo "---- Interactive (for development)----"
        export OMP_NUM_THREADS=${NTHREADS}
        rm -fr ../run
        mkdir ../run
        cp ../makefiles/${EXECNAME} ../run/executable  
        cp ../makefiles/config.ini ../run/
        cp ../factory/factoryIF2D ../run/
        cp ../factory/factoryFish ../run/
        cd ../run
        mpirun -np 2 ./executable ${OPTIONS}
    fi


    if [[ "$(hostname)" == brutus* ]]; then
        export OMP_NUM_THREADS=${NTHREADS}
        mkdir -p ${BASEPATH}${RUNFOLDER}
        if [ "${RESTARTPOLICY}" = " -restartPolicy 1" ]; then
            echo "---- launch.sh >> Restart Policy ----"
            mkdir -p ${BASEPATH}${RUNFOLDER}/res 
            cp ../factory/policy_backup ${BASEPATH}${RUNFOLDER}/res/ 
        fi
        cp ../factory/factoryIF2D ${BASEPATH}${RUNFOLDER}/
        cp ../factory/factoryFish ${BASEPATH}${RUNFOLDER}/
        cp ../makefiles/config.ini ${BASEPATH}${RUNFOLDER}/
        cp ../makefiles/${EXECNAME} ${BASEPATH}${RUNFOLDER}/
        cp ${0} ${BASEPATH}${RUNFOLDER}/
        cd ${BASEPATH}${RUNFOLDER}
        echo "Submission 0..."
        bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=48] -sp 100 -W ${WCLOCK} mpirun -np ${NPROCESS} --mca btl tcp,self -pernode  ./${EXECNAME} ${OPTIONS}
        # module load valgrind
        # bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=48] -sp 100 -W ${WCLOCK} mpirun -np ${NPROCESS} --mca btl tcp,self -pernode valgrind  --num-callers=100  --tool=memcheck  --leak-check=yes  --track-origins=yes --show-reachable=yes  --log-file=totoValgrind   ./${EXECNAME} ${OPTIONS}
        #valgrind --tool=memcheck --leak-check=yes --log-file=toto%p 
        
        # Job Chaining
        RESTART=" -restart 1"
        OPTIONS=${SETTINGS}${RESTART}${RESTARTPOLICY}
        for (( c=1; c<=${TIMES}-1; c++ ))
        do
            echo "Submission $c..."
            bsub -J ${RUNFOLDER} -n ${NPROCESSORS} -R span[ptile=48] -sp 100 -w "ended(${RUNFOLDER})" -W ${WCLOCK} mpirun -np ${NPROCESS} --mca btl tcp,self -pernode  ./${EXECNAME} ${OPTIONS}
        done

    fi
fi
