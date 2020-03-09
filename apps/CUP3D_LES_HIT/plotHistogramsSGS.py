import argparse, matplotlib.pyplot as plt, numpy as np
nBINS = 900

def plot(ax, dirname):
  fRelPath='/simulation_000_00000/run_00000000/sgsAnalysis.raw'
  f = np.fromfile(dirname + fRelPath, dtype=np.float64)
  nSAMPS = f.size // (nBINS + 4)
  f = f.reshape([nSAMPS, nBINS + 4])
  x = (np.arange(nBINS) + 0.5)/nBINS * 0.09
  #P =  np.zeros(nBINS)
  #for i in range(nSAMPS):
  #  MCS, VCS = f[i,0], f[i,2]
  #  denom = 1.0 / np.sqrt(2*np.pi*VCS) / nSAMPS
  #  P += denom * np.exp( -(x-MCS)**2 / (2*VCS) )
  #plt.plot(x, P)
  line = ax.plot(x, np.mean(f[:,4:], axis=0), label=dirname[-5:])
  return line

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "CSS plotter.")
  parser.add_argument('--path', nargs='+', help="Simulation case.")
  parsed = parser.parse_args()
  fig = plt.figure()
  ax = plt.subplot(1, 1, 1)
  lines = []
  for PATH in parsed.path:
    lines.append( plot(ax, PATH) )
  ax.legend(loc='upper left')
  plt.show()
