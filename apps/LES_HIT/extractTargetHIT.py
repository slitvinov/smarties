#!/usr/bin/env python3.6
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

from sklearn.neighbors.kde import KernelDensity

def getDataScalar(path, files, nSkip):
    nFiles = len(files)
    scalars = [dict() for i in range(nSkip, nFiles)]

    print(len(scalars))

    for i in range(nFiles-nSkip):
        f = open(path+files[i+nSkip], 'r')
        line = f.readline()
        for nLine in range(12):
            line = f.readline()
            line = line.split()
            newKey = {line[0] : float(line[1])}
            scalars[i].update(newKey)
    return scalars

def getDataSpectrum(path, files, nSkip):
    nFiles = len(files)
    nData = nFiles - nSkip

    modes, energy = np.loadtxt(path+files[nSkip], unpack=True, skiprows=15)
    nModes = len(modes)

    spectrum = np.ndarray(shape=(nData, nModes), dtype=float)
    spectrum[0,:] = energy


    for i in range(1, nData):
        modes, energy = np.loadtxt(path+files[i+nSkip], unpack=True, skiprows=15)
        spectrum[i,:] = energy

    return modes, spectrum

def exportTarget(spectrum, modes, scalars, nSample):
    shape = spectrum.shape

    nModes = shape[1]
    nData  = shape[0]

    modes *= 2*np.pi / scalars[0]['lBox']

    mean = np.array([np.mean(spectrum[:,i]) for i in range(nModes)])
    std  = np.array([np.std(spectrum[:,i])  for i in range(nModes)])

    with open('meanTarget.dat', 'w') as f:
        for i in range(nModes):
            f.write('{:.4f}\t{:0.7e}\t{:0.7e}\n'.format(modes[i], mean[i], std[i]))


    maxRew = 0
    with open('kdeTarget.dat', 'w') as f:
        f.write('{}\t#nSamplePerMode\n\n'.format(nSample))

        for i in range(nModes):
            f.write('{:.4f}\n'.format(modes[i]))
            e_in = spectrum[:,i].reshape(-1,1)

            mi = 0.8*e_in.min()
            ma = 1.2*e_in.max()

            e_out = np.linspace(mi, ma, num=nSample).reshape(-1,1)

            bandwidth = std[i]
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(e_in)

            zz = np.exp(kde.score_samples(e_out))
            zz = np.reshape(nSample*zz/np.sum(zz), e_out.shape)

            if (i<=32):
                maxRew += max(zz)

            for n in range(nSample):
                f.write('{:.7e}\t{:.7e}\n'.format(e_out[n][0], zz[n][0]))
            f.write('\n')

    print('max={}'.format(maxRew))
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


def main(simdir, nSkip, nSample):
    path = simdir + "/analysis/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files.sort()

    nData = len(files) - nSkip
    scalars = getDataScalar(path, files, nSkip)
    modes, spectrum = getDataSpectrum(path, files, nSkip)

    exportTarget(spectrum, modes, scalars, nSample)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('simdir', help="Simulation directory containing the 'Analysis' files")
  parser.add_argument('nSkip',  help="Skip the n first analysis files.")
  parser.add_argument('nSamp',  help="Nb. of KDE sample")
  args = parser.parse_args()

  main(args.simdir, int(args.nSkip), int(args.nSamp))

