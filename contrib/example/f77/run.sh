#!/bin/sh

. /cluster/apps/local/env2lmod.sh
module load openblas
mpirun ./app_main
