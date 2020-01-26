#!/usr/bin/env python3.6
import re, argparse, numpy as np, glob
from os import path

nBins = 16 * 16//2 - 1

def tkeFit(nu, eps): return 2.87657077 * np.power(eps, 2/3.0)
def relFit(nu, eps):
    tke = tkeFit(nu,eps)
    uprime = np.sqrt(2.0/3.0 * tke);
    lambd = np.sqrt(15 * nu / eps) * uprime;
    return uprime * lambd / nu;

def epsNuFromRe(Re, uEta = 1.0):
    C = np.sqrt(196.0/20.0)
    K = 2/3.0 * C * np.sqrt(15)
    eps = np.power(uEta*uEta * Re / K, 3.0/2.0)
    nu = np.power(uEta, 4) / eps
    return eps, nu

def findAllParams(path):
    REs = set()
    alldirs = glob.glob(path+'*')
    for dirn in alldirs: REs.add(re.findall('RE\d\d\d\d', dirn)[0][2:])
    REs  = list( REs);  REs.sort()
    for i in range(len(REs)): REs[i] = float(REs[i])
    return REs

def computeIntTimeScale(tau_integral):
    tau_int = 0.0
    N = len(tau_integral)
    for i in range(N):
        if(tau_integral[i]<1e16):  tau_int += tau_integral[i]/N
        else: tau_int += 1e16/N
    return tau_int

def getAllData(dirn, eps, nu, fSkip=1):
    fname = dirn + '/spectralAnalysis.raw'
    if  path.exists(fname) :
      f = np.fromfile(fname, dtype=np.float64)
    else :
      f = np.zeros([nBins + 13])
      print('File %s does not exist. Simulation did not start?' % fname)
    f = f.reshape([f.size//(nBins + 13), nBins + 13])
    nSamples = f.shape[0]
    tIntegral = computeIntTimeScale(f[:,8])
    tAnalysis = np.sqrt(nu / eps) # time space between data files
    ind0 = int(5 * tIntegral / tAnalysis) # skip initial integral times
    if ind0 == 0 or ind0 > nSamples: ind0 = nSamples
    data = {
        'dt'           : f[ind0:, 1], 'tke'          : f[ind0:, 3],
        'tke_filtered' : f[ind0:, 4], 'dissip_visc'  : f[ind0:, 5],
        'dissip_tot'   : f[ind0:, 6], 'l_integral'   : f[ind0:, 7],
        't_integral'   : f[ind0:, 8], 'grad_mean'    : f[ind0:,11],
        'grad_std'     : f[ind0:,12], 'spectra'      : f[ind0:,13:]
    }
    return data

def main(path, fSkip, nBlocksRL=4):
  REs = findAllParams(path)
  #REs =  [60, 70, 82, 95, 110, 130, 150, 176, 206, 240, 280, 325, 380]
  EPSs, NUs = len(REs) * [0], len(REs) * [0] # will be overwritten
  for j in range(len(REs)):
    EPSs[j], NUs[j] = epsNuFromRe(REs[j])
    print('Re %e nu %e eps %e' % (REs[j], NUs[j], EPSs[j]))
    data = None
    for run in [0, 1, 2, 3, 4, 5, 6, 7]:
      dirn = '%sRE%04d_RUN%d' % (path, REs[j], run)
      runData = getAllData(dirn, EPSs[j], NUs[j], fSkip)
      if data is None: data = runData
      else:
        for key in runData:
          data[key] = np.append(data[key], runData[key], 0)

    if data == None or data['dt'].size < 2:
      print('skipped eps:%f nu:%f' % (EPSs[j], NUs[j]))
      continue

    nBinsTgt = nBlocksRL * 16 // 2 - 1
    logE = np.log(data['spectra'])
    #print(logE.shape)
    avgLogSpec = np.mean(logE, axis=0)
    logE = np.log(data['spectra'][:, 0:nBinsTgt])
    stdLogSpec = np.std(logE, axis=0)
    covLogSpec = np.cov(logE, rowvar=False)
    #print(covLogSpec.shape)
    modes = np.arange(1, nBins+1) # assumes box is 2 pi
    avgTke, avgDissip = np.mean(data['tke']), np.mean(data['dissip_tot'])
    #reLambda = np.sqrt(20/3) * avgTke / np.sqrt(NUs[j] * avgDissip)
    reLambda = np.sqrt(20/3) * avgTke / np.sqrt(NUs[j] * EPSs[j])

    logCov2pi = np.power( np.linalg.det(2 * np.pi * covLogSpec), 0.5/nBinsTgt)
    print(-np.log(logCov2pi))

    fout = open('scalars_RE%03d' % int(REs[j]), "w")
    fout.write("eps %e\n" % (EPSs[j]) )
    fout.write("nu %e\n" % (NUs[j]) )
    fout.write("dt %e %e\n"     % (np.mean(data['dt']),
                                   np.std( data['dt']) ) )
    fout.write("tKinEn %e %e\n" % (np.mean(data['tke']),
                                   np.std( data['tke']) ) )
    fout.write("epsVis %e %e\n" % (np.mean(data['dissip_visc']),
                                   np.std( data['dissip_visc']) ) )
    fout.write("epsTot %e %e\n" % (np.mean(data['dissip_tot']),
                                   np.std( data['dissip_tot']) ) )
    fout.write("lInteg %e %e\n" % (np.mean(data['l_integral']),
                                   np.std( data['l_integral']) ) )
    fout.write("tInteg %e %e\n" % (np.mean(data['t_integral']),
                                   np.std( data['t_integral']) ) )
    fout.write("avg_Du %e %e\n" % (np.mean(data['grad_mean']),
                                   np.std( data['grad_mean']) ) )
    fout.write("std_Du %e %e\n" % (np.mean(data['grad_std']),
                                   np.std( data['grad_std']) ) )
    fout.write("ReLamd %e\n"    % reLambda)
    fout.write("logPdenom %e\n"   % -np.log(logCov2pi) )

    #print(modes.shape, avgLogSpec.shape)
    ary = np.append(     modes.reshape([nBins,1]),
                    avgLogSpec.reshape([nBins,1]), 1)
    np.savetxt('spectrumLogE_RE%03d' % int(REs[j]), ary, delimiter=', ')
    invCovLogSpec = np.linalg.inv(covLogSpec)
    np.savetxt('invCovLogE_RE%03d' % int(REs[j]), invCovLogSpec, delimiter=', ')
    np.savetxt('stdevLogE_RE%03d' % int(REs[j]), stdLogSpec, delimiter=', ')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('simdir',
      help="Simulation directory containing the 'Analysis' folder")
  parser.add_argument('--fSkip', type=int, default=1,
    help="Sampling frequency for analysis files. If 1, take all. If 2, take 1 skip 1, If 3, take 1, skip 2, and so on.")
  parser.add_argument('--nBlocksRL', type=int, default=4,
    help="Number of CubismUP 3D blocks in the training runs.")
  args = parser.parse_args()

  main(args.simdir, args.fSkip, args.nBlocksRL)

