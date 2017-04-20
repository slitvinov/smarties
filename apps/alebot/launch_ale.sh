SOCK=$1
ROM=$2

export LD_LIBRARY_PATH=${HOME}/smarties/apps/alebot/Arcade-Learning-Environment/:$LD_LIBRARY_PATH
../alebot ${SOCK} ${ROM}
