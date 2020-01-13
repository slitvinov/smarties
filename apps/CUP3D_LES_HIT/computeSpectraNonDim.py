#!/usr/bin/env python3
import re, argparse, numpy as np, glob, os
#from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from computeMeanIntegralQuantitiesNonDim import findAllParams
from computeMeanIntegralQuantitiesNonDim import readAllFiles
colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
nQoI = 8
h = 2 * np.pi / (16*16)
QoI = [ 'Time Step Size',
        'Turbulent Kinetic Energy',
        'Velocity Gradient',
        'Velocity Gradient Stdev',
        'Integral Length Scale',
]

def EkFunc(x, C, CI, CE, BETA, P0):
    if(C   <1e-16): C   =1e-16
    #if(CI<0): CI=0
    if(CI  <1e-16): CI  =1e-16
    if(CE  <1e-16): CE  =1e-16
    if(BETA<1e-16): BETA=1e-16
    if(P0  <1e-16): P0  =1e-16
    CI, BETA, P0 = 0, 5, 500
    k, eps, leta, lint, nu = x[0], x[1], x[2], x[3], x[4]
    #print(x.shape)
    #lint =  0.74885397 * np.power(eps, -0.0233311) * np.power(nu, 0.07192009)
    #leta = np.power(eps, -0.25) * np.power(nu, 0.75)
    #FL = 1 # np.power( k*lint / (np.abs(k*lint) + CI), 5/3.0 + P0 )
    FL = np.power( k*lint / np.sqrt((k*lint)**2 + CI), 5/3.0 + P0 )
    FE = np.exp( - BETA * ( np.power( (k*leta)**4 + CE**4, 0.25 ) - CE ) )
    ret = 3 * np.power(eps, 2/3.0) * np.power(k, -5/3.0) * FL * FE
    print(np.min(FE))
    #print(C, CI, CE, BETA, P0)
    return ret

def logEkFunc(x, C, CI, CE, BETA, P0):
    return np.log(EkFunc(x, C, CI, CE, BETA, P0))
def EkBrief(x, popt):
    return EkFunc(x, popt[0], popt[1], popt[2], popt[3], popt[4])

def readAllSpectra(path, REs):
    nRes = len(REs)
    allStdevs, allSpectra = None, None

    ind = 0
    for ei in range(nRes):
        ename = '%s/spectrumLogE_RE%03d' % (path, REs[ei])
        sname = '%s/stdevLogE_RE%03d' % (path, REs[ei])
        if os.path.isfile(ename) == False : continue
        if os.path.isfile(sname) == False : continue
        modes, stdevs = np.loadtxt(ename, delimiter=','), np.loadtxt(sname)
        nyquist = stdevs.size
        modes = modes[:nyquist,1].reshape(nyquist,1)
        stdevs = stdevs.reshape(nyquist,1)
        if allSpectra is None :
            allStdevs, allSpectra = np.zeros([nyquist,0]), np.zeros([nyquist,0])
        allStdevs  = np.append(allStdevs , stdevs, axis=1)
        allSpectra = np.append(allSpectra, modes , axis=1)

    return allSpectra, allStdevs

def fitFunction(inps, dataM, dataV, row, func):
    if dataV is None :
      popt, pcov = curve_fit(func, inps, dataM[row,:])
    else:
      popt, pcov = curve_fit(func, inps, dataM[row,:], sigma = dataV[row,:])
    return popt

def fitSpectrum(vecParams, vecMean, vecSpectra, vecEnStdev):
    assert(vecSpectra.shape[1] == vecEnStdev.shape[1])
    assert(vecSpectra.shape[1] == vecParams.shape[1])
    assert(vecSpectra.shape[1] == vecMean.shape[1])
    nyquist, nruns = vecSpectra.shape[0], vecSpectra.shape[1]
    kdata = np.zeros([nruns, nyquist, 5])
    for i in range(nruns):
        for j in range(nyquist):
            kdata[i, j, 0] = 0.5 + j
            kdata[i, j, 1] = vecParams[0,i]
            kdata[i, j, 2] = np.power(vecParams[1,i]**3 / vecParams[0,i], 0.25)
            kdata[i, j, 3] = vecMean[4,i]
            kdata[i, j, 4] = vecParams[1,i]
    #prepare vectors so that they are compatible with curve fit:
    ekdata, eksigma = vecSpectra.flatten(), vecEnStdev.flatten()
    kdata = kdata.reshape(nyquist*nruns, 5).transpose()
    bounds = [[ 1e-16,  1e-16, -np.inf,  1e-16,  1e-16],
              [np.inf, np.inf,  np.inf, np.inf, np.inf]]
    popt, pcov = curve_fit(logEkFunc, kdata, ekdata, sigma=eksigma,
        maxfev=100000, p0=[6, 1, 0, 5.24, 2], bounds=bounds)
    return popt, pcov

def main_integral(path):
    REs = findAllParams(path)
    nRes = len(REs)
    vecParams, vecMean, vecStd = readAllFiles(path, REs)
    vecSpectra, vecEnStdev = readAllSpectra(path, REs)
    popt, pcov = fitSpectrum(vecParams, vecMean, vecSpectra, vecEnStdev)
    C,CI,CE,BETA,P0 = popt[0], popt[1], popt[2], popt[3], popt[4]
    print(popt)

    plt.figure()
    axes = []
    nPlots = 1
    for i in range(nPlots):
      axes = axes + [plt.subplot(1, nPlots, 1+i)]
      axes[-1].set_xlabel(r'$k \eta$')
      axes[-1].grid()
    axes[0].set_ylabel(r'$E(k) / (\eta u^2_\eta)$')

    ci = 0
    nyquist, nruns = vecSpectra.shape[0], vecSpectra.shape[1]
    for i in range(1, nruns, 2):
        eps, nu, re = vecParams[0,i], vecParams[1,i], vecParams[2,i]
        leta = np.power(vecParams[1,i]**3 / vecParams[0,i], 0.25)
        lint = vecMean[4,i]
        ri = np.argmin(np.abs(REs - re))
        print(ri, i)

        Ekscal = np.power(nu**5 * eps, 0.25)
        K = np.arange(1, nyquist+1)
        X, Y = K * leta, np.exp(vecSpectra[:,i]) / Ekscal
        fit = np.array([EkBrief([k, eps,leta,lint,nu], popt) for k in K])/Ekscal
        Yb = np.exp(vecSpectra[:,i] - vecEnStdev[:,i])/Ekscal
        Yt = np.exp(vecSpectra[:,i] + vecEnStdev[:,i])/Ekscal

        idx = 0
        label = r'$Re=%f$' % re
        color = colors[ ci ]
        ci += 1
        axes[idx].fill_between(X, Yb, Yt, facecolor=color, alpha=.5)
        axes[idx].plot(X, fit, 'o', color=color)
        axes[idx].plot(X, Y, color=color, label=label)

    for ax in axes:
      ax.set_yscale("log")
      ax.set_xscale("log")
      ax.legend(loc='lower left')
    plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('--targets',
    help="Simulation directory containing the 'Analysis' folder")
  args = parser.parse_args()

  main_integral(args.targets)