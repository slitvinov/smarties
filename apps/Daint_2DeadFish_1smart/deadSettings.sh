# build the settings
SETTINGS=

# setup settings
SETTINGS+=" -study learning"
SETTINGS+=" -factory dead2Factory"
SETTINGS+=" -lambda 1e6"
SETTINGS+=" -nu 2e-5"
SETTINGS+=" -length 0.1"
SETTINGS+=" -Uinfx 0.3"
SETTINGS+=" -Uinfy 0.0"
SETTINGS+=" -compute-efficiency 0"
SETTINGS+=" -compute-pressure 1"
SETTINGS+=" -restartpath restart"

SETTINGS+=" -Tstartlearn 0.0"
SETTINGS+=" -nActions 1"

# grid settings
SETTINGS+=" -bpd 16"
SETTINGS+=" -uniform 0"
SETTINGS+=" -particles 1"
SETTINGS+=" -jump 2"
SETTINGS+=" -lmax 5"
SETTINGS+=" -rtol 1e-1"
SETTINGS+=" -ctol 1e-2"
SETTINGS+=" -adaptfreq 5"
SETTINGS+=" -boundary-adaptation free_frame"
SETTINGS+=" -refine-omega-only 1"
SETTINGS+=" -killvort 3"

# timestepping settings
SETTINGS+=" -cfl 0.9"
SETTINGS+=" -lcfl 0.1"
SETTINGS+=" -fc 0.25"
SETTINGS+=" -ramp 1000"
SETTINGS+=" -nsteps 100000000"
SETTINGS+=" -tmax 100000"

# velocity solver settings
SETTINGS+=" -fmm velocity-diego"
SETTINGS+=" -obstacle-potential 1"
SETTINGS+=" -fmm-theta 0.5"
SETTINGS+=" -core-fmm sse"
SETTINGS+=" -f2z 1"

# pressure solver settings
SETTINGS+=" -fmm-theta-pressure 0.5"
SETTINGS+=" -rho 1.0"
SETTINGS+=" -Usolidx 0.0"
SETTINGS+=" -Usolidy 0.0"

# postprocess settings
SETTINGS+=" -dumpfreq 0"
SETTINGS+=" -savefreq 0"

SETTINGS+=" -DumpTFreq 0"
SETTINGS+=" -SaveTFreq 0"
#SETTINGS+=" -DumpTFreq 32"
#SETTINGS+=" -SaveTFreq 4"
#SETTINGS+=" -vtu 1"
