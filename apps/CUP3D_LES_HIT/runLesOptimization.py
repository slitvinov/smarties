#!/usr/bin/env python3

import re, argparse, numpy as np, glob, subprocess
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def    tkeFit(nu, eps): return 2.86302040 * np.power(eps, 2/3.0)
#def    relFit(nu, eps): return 7.33972668 * np.power(eps, 1/6.0) / np.sqrt(nu)
def    relFit(nu, eps):
    tke = tkeFit(nu,eps)
    uprime = np.sqrt(2.0/3.0 * tke);
    lambd = np.sqrt(15 * nu / eps) * uprime;
    return uprime * lambd / nu;

def epsNuFromRe(Re, uEta = 1.0):
    C = 2.87657077
    K = 2/3.0 * C * np.sqrt(15)
    eps = np.power(uEta*uEta * Re / K, 3.0/2.0)
    nu = np.power(uEta, 4) / eps
    return eps, nu

def runspec(re, cs, run):
    return "HIT_BOX2_LES_EXT2pi_RE%03d_CS%.03f_RUN%d" % (re, cs, run)

def getSettings(nu, eps, cs):
    options = '-sgs SSM -cs %f -bpdx 4 -bpdy 4 -bpdz 4 -CFL 0.1 ' % cs
    tAnalysis = np.sqrt(nu / eps)
    tEnd = 1000 * tAnalysis
    return options + '-extentx 6.2831 -dump2D 0 -dump3D 0 ' \
       '-tdump 0 -BC_x periodic -BC_y periodic -BC_z periodic ' \
       '-spectralIC fromFile -initCond HITurbulence -tAnalysis %f ' \
       '-compute-dissipation 1 -nprocsx 1 -nprocsy 1 -nprocsz 1 ' \
       '-spectralForcing 1 -tend %f -keepMomentumConstant 1 ' \
       '-analysis HIT -nu %f -energyInjectionRate %f ' \
       % (tAnalysis, tEnd, nu, eps)

def launchEuler(tpath, nu, eps, re, cs, run):
    scalname = "%s/scalars_RE%03d" % (tpath, re)
    logEname = "%s/spectrumLogE_RE%03d" % (tpath, re)
    iCovname = "%s/invCovLogE_RE%03d" % (tpath, re)
    runname  = runspec(re, cs, run)
    cmd = "export LD_LIBRARY_PATH=/cluster/home/novatig/hdf5-1.10.1/gcc_6.3.0_openmpi_2.1/lib/:$LD_LIBRARY_PATH\n" \
      "FOLDER=/cluster/scratch/novatig/CubismUP_3D/%s\n " \
      "mkdir -p ${FOLDER}\n" \
      "cp ~/CubismUP_3D/bin/simulation ${FOLDER}/\n" \
      "cp %s ${FOLDER}/scalars_target\n" \
      "cp %s ${FOLDER}/spectrumLogE_target\n" \
      "cp %s ${FOLDER}/invCovLogE_target\n" \
      "export OMP_NUM_THREADS=18\n" \
      "cd $FOLDER\n" \
      "bsub -n 18 -J %s -W 04:00 -R \"select[model==XeonGold_6150] span[ptile=18]\" mpirun -n 1 ./simulation %s\n" \
      % (runname, scalname, logEname, iCovname, runname, getSettings(nu, eps, cs))
    subprocess.run(cmd, shell=True)


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

    nCss = 13

    for re in np.linspace(60, 240, 19):
      for cs in np.linspace(0.0, 0.24, nCss):
        for ri in [3]:
          eps, nu = epsNuFromRe(re)
          launchEuler(args.path, nu, eps, re, cs, ri)

