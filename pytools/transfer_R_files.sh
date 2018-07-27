#!/bin/bash
if [ $# -lt 3 ] ; then
echo "Usage: ./transfer_R_files.sh BASEPATH SHAREDPATTERN RETURNPATH"
exit 1
fi

BASEPATH=$1
SHAREDPATTERN=$2
RETURNPATH=$3
if [ -d $RETURNPATH ]; then
  echo "Stopped from overwriting" $RETURNPATH
  exit -1
fi
mkdir -p $RETURNPATH

for f in ${BASEPATH}*${SHAREDPATTERN}*; do
  echo $f
  dir=${f/${BASEPATH}}
  echo "Creating" ${RETURNPATH}/${dir}
  mkdir ${RETURNPATH}/${dir}
  cp ${f}/*cumulative_rewards*.dat ${RETURNPATH}/${dir}/
done

tar -zcvf ${RETURNPATH}.tar.gz ${RETURNPATH}
rm -rf ${RETURNPATH}
