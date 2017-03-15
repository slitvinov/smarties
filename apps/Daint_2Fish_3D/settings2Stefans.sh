#!/bin/bash
OPTIONS=
OPTIONS+=" -nActions 2"
#OPTIONS+=" -bpdx 64 -bpdy 32 -bpdz 8"
OPTIONS+=" -bpdx 32 -bpdy 16 -bpdz 4"
OPTIONS+=" -2Ddump 1 -restart 0"
OPTIONS+=" -nprocsz 1"
OPTIONS+=" -CFL 0.1"
OPTIONS+=" -length 0.2"
OPTIONS+=" -lambda 1e5"
OPTIONS+=" -nu 0.000008"
OPTIONS+=" -tend 40 -tdump 0.25"
