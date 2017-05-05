SOCK=$1

export LD_LIBRARY_PATH=${HOME}/smarties/apps/alebot/Arcade-Learning-Environment/:$LD_LIBRARY_PATH
../alebot ${SOCK} ${HOME}/ROMS/Boxing.bin
