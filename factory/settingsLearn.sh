# build the settings
SETTINGS=

# setup settings
SETTINGS+=" -study learning"
SETTINGS+=" -LearnFreq 2"
SETTINGS+=" -factory factoryStefans"
SETTINGS+=" -lambda 1e4"
SETTINGS+=" -nu 4e-5"

# grid settings
SETTINGS+=" -bpd 16"
SETTINGS+=" -uniform 0"
SETTINGS+=" -particles 1"
SETTINGS+=" -jump 2"
SETTINGS+=" -lmax 5"
SETTINGS+=" -rtol 1e-2"
SETTINGS+=" -ctol 1e-4"
SETTINGS+=" -adaptfreq 20"
SETTINGS+=" -boundary-adaptation free_frame"
SETTINGS+=" -refine-omega-only 0"
SETTINGS+=" -killvort 0"

# timestepping settings
SETTINGS+=" -cfl 0.5"
SETTINGS+=" -lcfl 0.1"
SETTINGS+=" -fc 0.25"
SETTINGS+=" -ramp 100"
SETTINGS+=" -nsteps 100000000"
SETTINGS+=" -tmax 15.0"

# velocity solver settings
SETTINGS+=" -fmm velocity"
SETTINGS+=" -obstacle-potential 1"
SETTINGS+=" -fmm-theta 0.8"
SETTINGS+=" -core-fmm sse"
SETTINGS+=" -f2z 1"

# postprocess settings
SETTINGS+=" -dumpfreq 0"
SETTINGS+=" -savefreq 1000"
SETTINGS+=" -DumpTFreq 32"
SETTINGS+=" -vtu 1"
