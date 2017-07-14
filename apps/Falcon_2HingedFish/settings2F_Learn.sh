# build the settings
SETTINGS=

# setup settings
SETTINGS+="\n-study\nlearning"
SETTINGS+="\n-factory\n../factory2F_Learn"
SETTINGS+="\n-length\n0.2"
SETTINGS+="\n-nu\n8e-6"
SETTINGS+="\n-lmax\n5"
SETTINGS+="\n-restartpath\nrestart"

SETTINGS+="\n-lambda\n1e5"
SETTINGS+="\n-Uinfx\n0.0"
SETTINGS+="\n-Uinfy\n0.0"

SETTINGS+="\n-Tstartlearn\n1"
SETTINGS+="\n-NpLatLine\n10"
SETTINGS+="\n-nActions\n2"
SETTINGS+="\n-nStates\n4"
SETTINGS+="\n-GoalDX\n2"

# grid settings
SETTINGS+="\n-bpd\n4"
SETTINGS+="\n-uniform\n0"
SETTINGS+="\n-particles\n1"
SETTINGS+="\n-jump\n2"
SETTINGS+="\n-rtol\n1e-1"
SETTINGS+="\n-ctol\n1e-2"
SETTINGS+="\n-adaptfreq\n20"
SETTINGS+="\n-boundary-adaptation\nfree_frame"
SETTINGS+="\n-refine-omega-only\n1"
SETTINGS+="\n-compute-efficiency\n0"
SETTINGS+="\n-compute-pressure\n1"
SETTINGS+="\n-killvort\n1"

# timestepping settings
SETTINGS+="\n-cfl\n0.95"
SETTINGS+="\n-lcfl\n0.1"
SETTINGS+="\n-fc\n0.25"
SETTINGS+="\n-ramp\n100"
SETTINGS+="\n-nsteps\n100000000"
SETTINGS+="\n-tmax\n80000.0"

# velocity solver settings
SETTINGS+="\n-fmm\nvelocity-diego"
SETTINGS+="\n-obstacle-potential\n1"
SETTINGS+="\n-fmm-theta\n0.5"
SETTINGS+="\n-f2z\n1"
SETTINGS+="\n-core-fmm\nsse"

# pressure solver settings
SETTINGS+="\n-fmm-theta-pressure\n0.5"
SETTINGS+="\n-rho\n1.0"

# postprocess settings
SETTINGS+="\n-dumpfreq\n0"
SETTINGS+="\n-savefreq\n0"

SETTINGS+="\n-DumpTFreq\n30"
SETTINGS+="\n-SaveTFreq\n2"
#SETTINGS+=" -vtu 1"
