#!/usr/bin/env python3
import re, argparse, numpy as np, glob
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt

colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
linest = ['solid', 'dashed', 'dashdot', 'dotted']

def computeAverages(scalars):
    nData = len(scalars)
    tkes = [scalars[i]['tke']          for i in range(nData)]
    lmbd = [scalars[i]['lambda']       for i in range(nData)]
    rels = [scalars[i]['Re_lambda']    for i in range(nData)]
    tint = [scalars[i]['tau_integral'] for i in range(nData)]
    lint = [scalars[i]['l_integral']   for i in range(nData)]
    grad = [scalars[i]['mean_grad']    for i in range(nData)]
    return [np.mean(tkes), np.mean(lmbd), np.mean(rels),    \
            np.mean(tint), np.mean(lint), np.mean(grad)],   \
           [ np.std(tkes),  np.std(lmbd),  np.std(rels),    \
             np.std(tint),  np.std(lint),  np.std(grad)],

def computeIntTimeScale(scalars):
    nData = len(scalars)
    tau_integral = 0.0
    for i in range(len(scalars)):
      if(scalars[i]['tau_integral'] < 100):
        tau_integral += scalars[i]['tau_integral']
    return tau_integral/nData

def getAllFiles(dirsname, nSkip):
    dirs = glob.glob(dirsname+'*')
    allFiles = []
    for path in dirs:
        files = glob.glob(path+'/analysis/spectralAnalysis_*')
        files.sort()
        allFiles = allFiles + files[nSkip:]
    return allFiles

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

def getMeanSpectrum(nu, spectrum, modes, scalars, nSample):
    shape = spectrum.shape

    nModes = shape[1]
    nData  = shape[0]
    #modes *= 2*np.pi / scalars[0]['lBox']
    #std  = np.array([np.std( spectrum[:,i]) for i in range(nModes)])
    return np.array([np.mean(spectrum[:,i]) for i in range(nModes)])

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
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(func, inputs, data, sigma=stdvs)
    return popt

def main_integral(path):
    #path = path[:path.find('EXT')] #cut param spec from common path
    NUs, EPSs, EXTs = set(), set(), set()
    alldirs = glob.glob(path+'*')
    for dirn in alldirs:
        EPSs.add(re.findall('EPS\d.\d\d',  dirn)[0][3:])
        NUs. add(re.findall('NU\d.\d\d\d', dirn)[0][2:])
        EXTs.add(re.findall('EXT\dpi',     dirn)[0][3] )
    NUs  = list( NUs);  NUs.sort()
    EPSs = list(EPSs); EPSs.sort()
    EXTs = list(EXTs); EXTs.sort()
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
  main_integral(args.simdir)
