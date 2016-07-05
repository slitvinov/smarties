# build the settings
SETTINGS=

# setup settings
SETTINGS+=" -study learning"
SETTINGS+=" -factory factory2F_Learn"
SETTINGS+=" -length 0.1"
SETTINGS+=" -nu 2e-6"
SETTINGS+=" -lmax 7"
SETTINGS+=" -restartpath restart"

SETTINGS+=" -lambda 1e6"
SETTINGS+=" -Uinfx 0.0"
SETTINGS+=" -Uinfy 0.0"
SETTINGS+=" -Nagents 1"

SETTINGS+=" -Tstartlearn 1"
SETTINGS+=" -LearnFreq 2"
SETTINGS+=" -GoalDX 2"

# grid settings
SETTINGS+=" -bpd 16"
SETTINGS+=" -uniform 0"
SETTINGS+=" -particles 1"
SETTINGS+=" -jump 2"
SETTINGS+=" -rtol 1e-3"
SETTINGS+=" -ctol 1e-5"
SETTINGS+=" -adaptfreq 20"
SETTINGS+=" -boundary-adaptation free_frame"
SETTINGS+=" -refine-omega-only 1"
SETTINGS+=" -compute-efficiency 0"
SETTINGS+=" -compute-pressure 1"
SETTINGS+=" -killvort 1"

# timestepping settings
SETTINGS+=" -cfl 0.95"
SETTINGS+=" -lcfl 0.1"
SETTINGS+=" -fc 0.25"
SETTINGS+=" -ramp 1000"
SETTINGS+=" -nsteps 100000000"
SETTINGS+=" -tmax 80000.0"

# velocity solver settings
SETTINGS+=" -fmm velocity-diego"
SETTINGS+=" -obstacle-potential 1"
SETTINGS+=" -fmm-theta 0.5"
SETTINGS+=" -f2z 1"
SETTINGS+=" -core-fmm sse"

# pressure solver settings
SETTINGS+=" -fmm-theta-pressure 0.5"
SETTINGS+=" -rho 1.0"

# postprocess settings
SETTINGS+=" -dumpfreq 0"
SETTINGS+=" -savefreq 0"

SETTINGS+=" -DumpTFreq 30"
SETTINGS+=" -SaveTFreq 2"
#SETTINGS+=" -vtu 1"