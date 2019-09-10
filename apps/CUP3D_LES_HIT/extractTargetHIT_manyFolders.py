#!/usr/bin/env python3
import re, argparse, numpy as np, glob
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
linest = ['solid', 'dashed', 'dashdot', 'dotted']

def computeAverages(scalars):
    tkes = [scalars[i]['tke']          for i in range(len(scalars))]
    lmbd = [scalars[i]['lambda']       for i in range(len(scalars))]
    rels = [scalars[i]['Re_lambda']    for i in range(len(scalars))]
    tint = [scalars[i]['tau_integral'] for i in range(len(scalars))]
    lint = [scalars[i]['l_integral']   for i in range(len(scalars))]
    grad = [scalars[i]['mean_grad']    for i in range(len(scalars))]
    return [np.mean(tkes), np.mean(lmbd), np.mean(rels),    \
            np.mean(tint), np.mean(lint), np.mean(grad)],   \
           [ np.std(tkes),  np.std(lmbd),  np.std(rels),    \
             np.std(tint),  np.std(lint),  np.std(grad)],

def computeIntTimeScale(scalars):
    tau_integral = 0.0
    for i in range(len(scalars)):
      if(scalars[i]['tau_integral'] < 100): tau_integral += scalars[i]['tau_integral']
      else: tau_integral += 100000
    return tau_integral/len(scalars)

def computeViscTimeScale(scalars):
    tau_eta = 0.0
    for i in range(len(scalars)):
      if(scalars[i]['tau_eta'] < 100): tau_eta += scalars[i]['tau_eta']
      else: tau_eta += 100000
    return tau_eta/len(scalars)

def computeIntLengthScale(scalars):
    l_integral = 0.0
    for i in range(len(scalars)):
      if(scalars[i]['l_integral'] < 100): l_integral += scalars[i]['l_integral']
      else: l_integral += 100000
    return l_integral/len(scalars)

def computeViscLengthScale(scalars):
    eta = 0.0
    for i in range(len(scalars)):
      if(scalars[i]['eta'] < 100): eta += scalars[i]['eta']
      else: eta += 100000
    return eta/len(scalars)

def getAllFiles(dirsname, nSkip):
    dirs = glob.glob(dirsname+'*')
    allFiles = []
    for path in dirs:
        files = glob.glob(path+'/analysis/spectralAnalysis_*')
        files.sort()
        allFiles = allFiles + files[nSkip:]
    return allFiles

def findAllParams(path):
    NUs, EPSs, EXTs = set(), set(), set()
    alldirs = glob.glob(path+'*')
    for dirn in alldirs:
        EPSs.add(re.findall('EPS\d.\d\d',  dirn)[0][3:])
        NUs. add(re.findall('NU\d.\d\d\d', dirn)[0][2:])
        EXTs.add(re.findall('EXT\dpi',     dirn)[0][3] )
    NUs  = list( NUs);  NUs.sort()
    EPSs = list(EPSs); EPSs.sort()
    EXTs = list(EXTs); EXTs.sort()
    return EXTs, NUs, EPSs

def getDataScalar(files):
    nFiles = len(files)
    scalars = [dict() for i in range(nFiles)]

    for i in range(nFiles):
        f = open(files[i], 'r')
        line = f.readline() # skip
        for nLine in range(15):
            line = f.readline()
            line = line.split()
            newKey = {line[0] : float(line[1])}
            scalars[i].update(newKey)
    return scalars

def getDataSpectrum(files):
    nFiles = len(files)
    nData = nFiles

    modes, energy = np.loadtxt(files[0], unpack=True, skiprows=18)
    nModes = len(modes)

    spectrum = np.ndarray(shape=(nData, nModes), dtype=float)
    spectrum[0,:] = energy


    for i in range(1, nData):
        modes, energy = np.loadtxt(files[i], unpack=True, skiprows=18)
        spectrum[i,:] = energy

    return modes, spectrum

def exportTarget(nu, spectrum, modes, scalars, nSample):
    shape = spectrum.shape

    nModes = shape[1]
    nData  = shape[0]

    modes *= 2*np.pi / scalars[0]['lBox']

    mean = np.array([np.mean(spectrum[:,i]) for i in range(nModes)])
    std  = np.array([np.std(spectrum[:,i])  for i in range(nModes)])

    with open('meanTarget.dat', 'w') as f:
        for i in range(nModes):
            f.write('{:.4f}\t{:0.7e}\t{:0.7e}\n'.format(modes[i], mean[i], std[i]))


    max_rew = 0
    with open('kdeTarget.dat', 'w') as f:
        f.write('{}\t#nSamplePerMode\n\n'.format(nSample))

        for i in range(nModes):
            f.write('{:.4f}\n'.format(modes[i]))
            e_in = spectrum[:,i].reshape(-1,1)

            mi = e_in.min()
            ma = e_in.max()

            e_out = np.linspace(mi, ma, num=nSample).reshape(-1,1)

            bandwidth = std[i]
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(e_in)

            zz = np.exp(kde.score_samples(e_out))
            zz = np.reshape(nSample*zz/np.sum(zz), e_out.shape)

            if (i<32):
                max_rew += max(zz)

            for n in range(nSample):
                f.write('{:.7e}\t{:.7e}\n'.format(e_out[n][0], zz[n][0]))
            f.write('\n')

    tau_integral=0
    tau_eta=0
    for i in range(nData):
        tau_integral += scalars[i]['tau_integral']
        tau_eta      += scalars[i]['tau_eta']

    tau_integral /= 1.0*nData
    tau_eta      /= 1.0*nData

    with open('scaleTarget.dat', 'w') as f:
        f.write('{:.4f}\t#tau_integral\n'.format(tau_integral))
        f.write('{:.4f}\t#tau_eta\n'.format(tau_eta))
        f.write('{:.4f}\t#max_rew with 32 modes\n'.format(max_rew[0]))
        f.write('{:.4f}\t#viscosity'.format(nu))

def getLogSpectrumStats(spectrum, modes, scalars):
    modes *= 2*np.pi / scalars[0]['lBox']
    logE = np.log(np.asarray(spectrum))
    #print(logE.shape, spectrum.shape)
    mean  = np.array([np.mean(logE[:,i]) for i in range(spectrum.shape[1])])
    stdev = np.array([np.std( logE[:,i]) for i in range(spectrum.shape[1])])
    for i in range(1,len(mean)):
      if mean[i] < -36:
        mean[i]  = ( mean[i] +  mean[i-1])/2
        stdev[i] = max(stdev[i], stdev[i-1])
    #print(logE.shape, spectrum.shape, mean.shape, stdev.shape)
    return modes, mean, stdev

def main(dirsname, nu, nSkip, nSample):
    files = getAllFiles(dirsname, nSkip)
    scalars = getDataScalar(files)
    modes, spectrum = getDataSpectrum(files)

    exportTarget(nu, spectrum, modes, scalars, nSample)

def fitFunction(exts, nus, epss, dataM, dataV, ind, func):
    data = []
    stdvs = []
    inputs = []
    for li in range(len(exts)):
     for ni in range(len(nus)):
      for ei in range(len(epss)):
       if(dataM[li,ni,ei,ind]<1000 and dataM[li,ni,ei,ind]>-1000):
        inputs = inputs + [[exts[li], nus[ni], epss[ei]]]
        stdvs = stdvs + [dataV[li, ni, ei, ind]]
        data = data + [dataM[li, ni, ei, ind]]
    data, stdvs= np.asarray(data), np.asarray(stdvs)
    inputs = np.asarray(inputs).transpose()
    print(data.shape, inputs.shape)
    popt, pcov = curve_fit(func, inputs, data, sigma=stdvs)
    return popt

def main_integral(path):
    EXTs, NUs, EPSs = findAllParams(path)
    print(EXTs, NUs, EPSs)
    # average/variance tke, lambda, re, tauint, lint, l2grad
    dataM = np.zeros([len(EXTs), len(NUs), len(EPSs), 6])
    dataV = np.zeros([len(EXTs), len(NUs), len(EPSs), 6])
    for ei in range(len(EPSs)):
     EPSs[ei] = float(EPSs[ei])
     for ni in range(len(NUs)):
      NUs[ni] = float(NUs[ni])
      for li in range(len(EXTs)):
       EXTs[li] = float(EXTs[li])
       allScalars = []
       for run in [0, 1, 2, 3, 4]:
        dirn = '%sEXT%dpi_NU%.03f_EPS%.02f_RUN%d' \
               % (path, EXTs[li], NUs[ni], EPSs[ei], run)
        files = getAllFiles(dirn, 0)
        scalars = getDataScalar(files)
        tint = computeIntTimeScale(scalars)
        #print(tint)
        allScalars = allScalars + scalars[int(tint/0.1):]
       dataM[li,ni,ei,:], dataV[li,ni,ei,:] = computeAverages(allScalars)

    plt.figure()
    axes = [ plt.subplot(2,3,1), plt.subplot(2,3,2), plt.subplot(2,3,3), \
             plt.subplot(2,3,4), plt.subplot(2,3,5), plt.subplot(2,3,6) ]
    axes[0].set_ylabel('Turbulent Kinetic Energy')
    axes[1].set_ylabel('Taylor Microscale')
    axes[2].set_ylabel('Reynolds Number')
    axes[3].set_ylabel('Integral Time Scale')
    axes[4].set_ylabel('Integral Length Scale')
    axes[5].set_ylabel('Vel Gradient Magnitude')
    for ax in axes     : ax.grid()
    for ax in axes[:3] : ax.set_xticklabels([])
    for ax in axes[3:] : ax.set_xlabel('Energy Injection Rate')

    for ai in range(6):
     for ni in range(len(NUs)):
      for li in range(len(EXTs)):
        Y  = dataM[li,ni,:,ai]
        Yb, Yt = Y - dataV[li,ni,:,ai], Y + dataV[li,ni,:,ai]
        axes[ai].fill_between(EPSs, Yb, Yt, facecolor=colors[ni], alpha=.5)
        axes[ai].plot(EPSs, Y, color=colors[ni], linestyle=linest[li])

    def fitTKE(x, A):    return A * np.power(x[2], 2/3.0)
    def fitLAMBDA(x, A): return A * np.power(x[2],-1/6.0) * np.power(x[1], 0.5)
    def fitREL(x, A):    return A * np.power(x[2], 1/6.0) * np.power(x[1],-0.5)
    def fitFun(x, A, B, C): return A * np.power(x[2], B) * np.power(x[1], C)

    popt = fitFunction(EXTs, NUs, EPSs, dataM, dataV, 0, fitTKE)
    print('tke fit:', popt)
    popt = fitFunction(EXTs, NUs, EPSs, dataM, dataV, 1, fitLAMBDA)
    print('lambda fit:', popt)
    popt = fitFunction(EXTs, NUs, EPSs, dataM, dataV, 2, fitREL)
    print('Re_lambda fit:', popt)
    popt = fitFunction(EXTs, NUs, EPSs, dataM, dataV, 3, fitFun)
    print('tau_integral fit:', popt)
    popt = fitFunction(EXTs, NUs, EPSs, dataM, dataV, 4, fitFun)
    print('l_integral fit:', popt)
    popt = fitFunction(EXTs, NUs, EPSs, dataM, dataV, 5, fitFun)
    print('mean_grad fit:', popt)
    #files = getAllFiles(dirsname, nSkip)
    #scalars = getDataScalar(files)
    plt.show()

def main_spectral(path):
    def EkFunc(x, C, CI, CE, BETA, P0):
      if(C   <1e-16): C   =1e-16
      #if(CI<0): CI=0
      if(CE  <1e-16): CE  =1e-16
      if(BETA<1e-16): BETA=1e-16
      if(P0  <1e-16): P0  =1e-16
      k, eps, leta, lint = x[0], x[1], x[2], x[3]
      FL = np.power( k*lint / (np.abs(k*lint) + CI), 5/3.0 + P0 )
      #FL = np.power( k*lint / np.sqrt((k*lint)**2 + CI), 5/3.0 + P0 )
      FE = np.exp( -BETA * ( np.power( (k*leta)**4 + CE**4, 0.25 ) - CE ) )
      return C * np.power(eps, 2/3.0) * np.power(k, -5/3.0) * FL * FE
    def logEkFunc(x, C, CI, CE, BETA, P0):
      return np.log(EkFunc(x, C, CI, CE, BETA, P0))

    EXTs, NUs, EPSs = findAllParams(path)
    allSpectra = None
    allStdevs = None
    allModes = None
    inptdata = None
    ekdata = None
    for ei in range(len(EPSs)):
      EPSs[ei] = float(EPSs[ei])
      for ni in range(len(NUs)):
        NUs[ni] = float(NUs[ni])
        for li in range(1):
          EXTs[li] = float(EXTs[li])
          scalars, spectra, modes = [], None, None
          for run in [0, 1, 2, 3, 4]:
            dirn = '%sEXT%dpi_NU%.03f_EPS%.02f_RUN%d' \
                   % (path, EXTs[li], NUs[ni], EPSs[ei], run)
            files = getAllFiles(dirn, 0)
            runscalars = getDataScalar(files)
            runmodes, runspectrum = getDataSpectrum(files)
            tint = computeIntTimeScale(runscalars)
            if(tint>=10): continue
            #print(runmodes)
            scalars = scalars + runscalars[int(tint/0.1):]
            modes = runmodes
            if spectra is None:
              spectra = runspectrum[int(tint/0.1):]
              #modes   = runmodes[int(tint/0.1):]
            else:
              spectra = np.append(spectra, runspectrum[int(tint/0.1):], 0)
              #modes   = np.append(  modes,    runmodes[int(tint/0.1):], 0)

          if len(scalars) == 0: continue

          modes,avgLogSpec,stdLogSpec = getLogSpectrumStats(spectra,modes,scalars)

          leta = computeViscLengthScale(scalars)
          lint = computeIntLengthScale(scalars)
          if(allModes is None) :
            allModes = np.zeros([0, len(modes)])
            allStdevs = np.zeros([0, len(modes)])
            allSpectra = np.zeros([0, len(modes)])
            inptdata = np.zeros([0, len(modes), 5])
          allSpectra = np.append(allSpectra, avgLogSpec.reshape(1,len(modes)), 0)
          allStdevs = np.append(allStdevs, stdLogSpec.reshape(1,len(modes)), 0)
          allModes = np.append(allModes, modes.reshape(1,len(modes)), 0)
          inptdata = np.append(inptdata, np.zeros([1, len(modes), 5]), 0)

          for k in range(len(modes)):
            inptdata[-1, k, 0] = modes[k]
            inptdata[-1, k, 1] = EPSs[ei]
            inptdata[-1, k, 2] = leta
            inptdata[-1, k, 3] = lint
            inptdata[-1, k, 4] = NUs[ni]

    ekdata = allSpectra.flatten()
    eksigma = allSpectra.flatten()
    kdata = inptdata.reshape(inptdata.shape[0]*inptdata.shape[1], 5)
    kdata = kdata.transpose()
    popt, pcov = curve_fit(logEkFunc, kdata, ekdata, sigma=eksigma, maxfev=100000, \
                 p0=[3.626, 0.000159, 0.178, 5.24, 1e3])
    # 3.62590946e+00 1.58981122e-04 1.78026894e-01 5.24030627e+00 6.69070846e+03
    C,CI,CE,BETA,P0 = popt[0], popt[1], popt[2], popt[3], popt[4]
    print(popt)

    plt.figure()
    axes = []
    for ni in range(len(NUs)):
      axes = axes + [plt.subplot(1, len(NUs), 1+ni)]
      axes[-1].set_xlabel(r'$k \eta$')
      axes[-1].grid()
    axes[0].set_ylabel(r'$E(k) / (\eta u^2_\eta)$')

    for i in range(allSpectra.shape[0]):
      eps,leta,lint,nu = inptdata[i,0,1],inptdata[i,0,2],inptdata[i,0,3],inptdata[i,0,4]
      ni = np.argmin(np.abs( NUs - nu))
      ei = np.argmin(np.abs(EPSs - eps))
      Ekscal = np.power(nu**5 * eps, 0.25)
      X, Y = allModes[i,:], np.exp(allSpectra[i,:])/Ekscal
      Yfit = np.array([EkFunc([k,eps,leta,lint], C,CI,CE,BETA,P0) for k in X])
      Yb = np.exp(allSpectra[i,:] - allStdevs[i,:])/Ekscal
      Yt = np.exp(allSpectra[i,:] + allStdevs[i,:])/Ekscal
      axes[ni].fill_between(X*leta, Yb, Yt, facecolor=colors[ei], alpha=.5)
      axes[ni].plot(X*leta, Yfit/Ekscal, color=colors[ei], linestyle='--')
      label = r'$\nu=%.03f\quad\epsilon=%.02f$' %(nu, eps)
      axes[ni].plot(X*leta, Y, color=colors[ei], label=label)

    for ax in axes:
      ax.set_yscale("log")
      ax.set_xscale("log")
      ax.legend(loc='lower left')
    plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('simdir',
    help="Simulation directory containing the 'Analysis' folder")
  parser.add_argument('nu',
    help="Viscosity of the simulation.")
  parser.add_argument('nSkip',
    help="Skip the n first analysis files.")
  parser.add_argument('nSamp',
    help="Nb. of KDE sample")
  args = parser.parse_args()

  #main(args.simdir, float(args.nu), int(args.nSkip), int(args.nSamp))
  #main_integral(args.simdir)
  main_spectral(args.simdir)