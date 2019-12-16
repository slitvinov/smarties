#!/usr/bin/env python3

import re, argparse, numpy as np, glob, subprocess
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def findAllParams(path):
    NUs, EPSs = set(), set()
    alldirs = glob.glob(path+'/scalars_*')
    for dirn in alldirs:
        EPSs.add(re.findall('EPS\d.\d\d\d',  dirn)[0][3:])
        NUs. add(re.findall('NU\d.\d\d\d\d', dirn)[0][2:])
    NUs, EPSs  = list( NUs), list(EPSs)
    NUs.sort()
    EPSs.sort()
    for i in range(len(NUs)): NUs[i] = float(NUs[i])
    for i in range(len(EPSs)): EPSs[i] = float(EPSs[i])
    return NUs, EPSs

def runspec(nu, eps, cs, run):
    return "HIT_BOX2_LES_EXT2pi_EPS%.03f_NU%.04f_CS%.03f_RUN%d" \
               % (eps, nu, cs, run)

def getSettings(nu, eps, cs):
    options = '-sgs SSM -cs %f -bpdx 4 -bpdy 4 -bpdz 4 -CFL 0.1 ' % cs
    tAnalysis = np.sqrt(nu / eps)
    return options + '-extentx 6.2831853072 -dump2D 0 -dump3D 0 ' \
       '-tdump 1 -BC_x periodic -BC_y periodic -BC_z periodic ' \
       '-spectralIC fromFit -initCond HITurbulence -tAnalysis %f ' \
       '-compute-dissipation 1 -nprocsx 1 -nprocsy 1 -nprocsz 1 ' \
       '-spectralForcing 1 -tend 200  -keepMomentumConstant 1 ' \
       '-analysis HIT -nu %f -energyInjectionRate %f ' \
       % (tAnalysis, nu, eps)

def launchEuler(nu, eps, cs, run):
    cmd = "FOLDER=~/CubismUP_3D/runs/%s\n " \
          "mkdir -p ${FOLDER}\n" \
          "cp ~/CubismUP_3D/bin/simulation ${FOLDER}/\n" \
          "export OMP_NUM_THREADS=12\n" \
          "cd $FOLDER\n" \
          "mpirun -np 1 ./simulation %s\n" \
          % (runspec(nu, eps, cs, run), getSettings(nu, eps, cs))
    subprocess.run(cmd, executable=parsed.shell, shell=True)


def launchDaint(nCases, les):
    SCRATCH = os.getenv('SCRATCH')
    HOME = os.getenv('HOME')

    f = open('HIT_sbatch','w')
    f.write('#!/bin/bash -l \n')
    if les:
      f.write('#SBATCH --job-name=LES_HIT \n')
      f.write('#SBATCH --time=01:00:00 \n')
    else:
      f.write('#SBATCH --job-name=DNS_HIT \n')
      f.write('#SBATCH --time=24:00:00 \n')
    f.write('#SBATCH --output=out.%j.%a.txt \n')
    f.write('#SBATCH --error=err.%j.%a.txt \n')
    f.write('#SBATCH --constraint=gpu \n')
    f.write('#SBATCH --account=s929 \n')
    f.write('#SBATCH --array=0-%d \n' % (nCases-1))
    #f.write('#SBATCH --partition=normal \n')
    #f.write('#SBATCH --ntasks-per-node=1 \n')

    f.write('ind=$SLURM_ARRAY_TASK_ID \n')
    if les:
      f.write('RUNDIRN=`./launchLESHIT.py --LES --case ${ind} --printName` \n')
      f.write('OPTIONS=`./launchLESHIT.py --LES --case ${ind} --printOptions` \n')
    else:
      f.write('RUNDIRN=`./launchLESHIT.py --case ${ind} --printName` \n')
      f.write('OPTIONS=`./launchLESHIT.py --case ${ind} --printOptions` \n')

    f.write('mkdir -p %s/CubismUP3D/${RUNDIRN} \n' % SCRATCH)
    f.write('cd %s/CubismUP3D/${RUNDIRN} \n' % SCRATCH)
    f.write('cp %s/CubismUP_3D/bin/simulation ./exec \n' % HOME)

    f.write('export OMP_NUM_THREADS=12 \n')
    f.write('export CRAY_CUDA_MPS=1 \n')
    f.write('srun --ntasks 1 --ntasks-per-node=1 ./exec ${OPTIONS} \n')
    f.close()
    os.system('sbatch HIT_sbatch')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")

    parser.add_argument('--path', default='target', help="Simulation case.")

    args = parser.parse_args()

    NUs, EPSs = findAllParams(args.path)
    nNus, nEps, nCss = len(NUs), len(EPSs), 9
    LESs = np.linspace(0.16, 0.24, nCss)

    for ni in range(nNus):
        for ei in range(nEps):
            for li in range(nCss):
                for ri in [0, 1, 2, 3, 4]:
                    launchEuler(NUS[ni], EPS[ei], LESs[li], ri)

