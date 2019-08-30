#!/bin/bash

DIR=$1
if [ -f ${DIR}.tar.gz ]; then
  echo "Stopped from overwriting" $RETURNPATH
  exit -1
fi
scp novatig@daint:/scratch/snx3000/novatig/smarties/${DIR}.tar.gz .
tar -zxvf ${DIR}.tar.gz
rm ${DIR}.tar.gz
