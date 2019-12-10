#!/usr/bin/env python3.6
import re, argparse, numpy as np, glob
nBins = 16 * 16//2 - 1

def findAllParams(path):
    NUs, EPSs, EXTs = set(), set(), set()
    alldirs = glob.glob(path+'*')
    for dirn in alldirs:
        EPSs.add(re.findall('EPS\d.\d\d\d',  dirn)[0][3:])
        NUs. add(re.findall('NU\d.\d\d\d\d', dirn)[0][2:])
    NUs  = list( NUs);  NUs.sort()
    EPSs = list(EPSs); EPSs.sort()
    return NUs, EPSs

def computeIntTimeScale(tau_integral):
    tau_int = 0.0
    N = len(tau_integral)
    for i in range(N):
        if(tau_integral[i]<100): tau_int += tau_integral[i]
        else: tau_int += 100000
    return tau_int/N

def getAllData(dirn, eps, nu, fSkip=1):
    f = np.fromfile(dirn + '/spectralAnalysis.raw', dtype=np.float64)
    f = f.reshape([f.size//(nBins + 13), nBins + 13])
    nSamples = f.shape[0]
    tIntegral = computeIntTimeScale(f[:,8])
    tAnalysis = np.sqrt(nu / eps) # time space between data files
    ind0 = int(5 * tIntegral / tAnalysis) # skip initial integral times
    if ind0 == 0 or ind0 > nSamples: ind0 = nSamples
    data = {
        'dt'           : f[ind0:, 1],  'tke'          : f[ind0:, 3],
        'tke_filtered' : f[ind0:, 4],  'dissip_visc'  : f[ind0:, 5],
        'dissip_tot'   : f[ind0:, 6],  'l_integral'   : f[ind0:, 7],
        't_integral'   : f[ind0:, 8],  'grad_mean'    : f[ind0:, 11],
        'grad_std'     : f[ind0:, 12], 'spectra'      : f[ind0:, 13:]
    }
    return data

def main(path, fSkip):
  NUs, EPSs = findAllParams(path)

  for ei in range(len(EPSs)):
    EPSs[ei] = float(EPSs[ei])
    for ni in range(len(NUs)):
      NUs[ni] = float(NUs[ni])

      data = None
      for run in [0, 1, 2, 3, 4]:
          dirn = '%sEXT2pi_EPS%.03f_NU%.04f_RUN%d' \
                 % (path, EPSs[ei], NUs[ni], run)
          runData = getAllData(dirn, EPSs[ei], NUs[ni], fSkip)
          if data is None: data = runData
          else:
            for key in runData:
              data[key] = np.append(data[key], runData[key], 0)

      if data == None or data['dt'].size < 2:
        print('skipped eps:%f nu:%f' % (EPSs[ei], NUs[ni]))
        continue

      logE = np.log(data['spectra'])
      avgLogSpec = np.mean(logE, axis=0)
      stdLogSpec = np.std(logE, axis=0)
      covLogSpec = np.cov(logE, rowvar=False)
      modes = np.arange(1, nBins+1) # assumes box is 2 pi
      reLambda = np.sqrt(20/3) * np.mean(data['tke']) / np.sqrt(NUs[ni] * np.mean(data['dissip_tot']))

      fout = open('scalars_EPS%.03f_NU%.04f' % (EPSs[ei], NUs[ni]), "w")
      fout.write("eps %e\n" % (EPSs[ei]) )
      fout.write("nu %e\n" % (NUs[ni]) )
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

      print(modes.shape, avgLogSpec.shape)
      ary = np.append(modes.reshape([nBins,1]), avgLogSpec.reshape([nBins,1]), 1)
      np.savetxt('spectrumLogE_EPS%.03f_NU%.04f' % (EPSs[ei], NUs[ni]), \
                 ary, delimiter=', ')
      invCovLogSpec = np.linalg.inv(covLogSpec)
      np.savetxt('invCovLogE_EPS%.03f_NU%.04f' % (EPSs[ei], NUs[ni]), \
                 invCovLogSpec, delimiter=', ')
      np.savetxt('stdevLogE_EPS%.03f_NU%.04f' % (EPSs[ei], NUs[ni]), \
                 stdLogSpec, delimiter=', ')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('simdir',
      help="Simulation directory containing the 'Analysis' folder")
  parser.add_argument('--fSkip', type=int, default=1,
    help="Sampling frequency for analysis files. If 1, take all. If 2, take 1 skip 1, If 3, take 1, skip 2, and so on.")
  args = parser.parse_args()

  main(args.simdir, args.fSkip)

