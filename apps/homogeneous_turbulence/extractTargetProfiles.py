#!/usr/bin/env python3.6
import argparse
import os
import numpy as np
import scipy.interpolate

def uniformMesh(nCells, l):
    res = np.zeros(nCells)
    for i in range(nCells):
        res[i] = l * ((i+0.5)/nCells - 0.5)
    return res

def sinusoidalMesh(nCells, l, eta):
    nHalf = int(nCells/2) + nCells%2
    res = np.zeros(nCells)
    spacing = np.zeros(nCells)
    h = 2.0/nCells
    ducky = 0.0
    for i in range(nHalf):
        x = h * (i + 0.5) - 1
        dh = np.sin(0.5*eta*x*np.pi)/np.sin(0.5*eta*np.pi) + 1
        spacing[i] = dh
        spacing[nCells-1-i] = dh
        ducky += 2*dh

    scale = l/ducky
    res[0] = spacing[0]/2*scale - l/2
    for i in range(1, nCells):
        res[i] = res[i-1] + (spacing[i-1]+spacing[i])/2*scale

    return res

# Get some paths
dirname = os.path.dirname(os.path.realpath(__file__))

def main(inputdir, bpd_y, l_y, eta_y):
    uy_file = dirname + '/' + inputdir + '/u_y'
    ky_file = dirname + '/' + inputdir + '/k_y'
    exists = os.path.isfile(uy_file) and os.path.isfile(ky_file)
    if not os.path.isfile(uy_file):
        print('Failed : file does not exist :', uy_file)
        return
    if not os.path.isfile(ky_file):
        print('Failed : file does not exists :', ky_file)
        return
    # Read data from input dir
    y, U = np.loadtxt(uy_file, unpack=True)
    y, k = np.loadtxt(ky_file, unpack=True)

    # Compte the position of mesh generating points for the simulation
    nCells = 16*bpd_y
    if eta_y > 0:
        y_tgt = sinusoidalMesh(nCells, l_y, eta_y)
    else:
        y_tgt = uniformMesh(nCells, l_y)

    spline_U = scipy.interpolate.splrep(y, U, s=0)
    spline_k = scipy.interpolate.splrep(y, k, s=0)

    U_tgt = scipy.interpolate.splev(y_tgt, spline_U, der=0)
    k_tgt = scipy.interpolate.splev(y_tgt, spline_k, der=0)

    # Export in text file
    outpath = dirname + '/target.dat'
    data = np.array([y_tgt, U_tgt, k_tgt])
    data = data.T
    np.savetxt(outpath, data)

    # Compute u_avg
    u_avg = 0.5*scipy.interpolate.splint(-1,1,spline_U)
    print(u_avg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Export mean Ux and kx target profiles for RL agent.')
    parser.add_argument('inputdir', help="Input directory containing files 'u_y' and 'k_y'")
    parser.add_argument('bpd_y', help="Number of blocks (of 16 cells) in the y direction")
    parser.add_argument('l_y', help="Extent of the simulation in y direction")
    parser.add_argument('eta_y', help="Sinusoidal mapping control. 0=uniform mesh")
    args = parser.parse_args()

    main(args.inputdir, int(args.bpd_y), float(args.l_y), float(args.eta_y))
