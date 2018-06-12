#/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from scipy.stats.stats import pearsonr

XCOL = int(sys.argv[1])
YCOL = int(sys.argv[2])
NFILES = len(range(3, len(sys.argv)))


for j in range(3, len(sys.argv)):
  DATA = np.fromfile(sys.argv[j], dtype=np.float32)
  print (DATA.size // 4)
  DATA = DATA.reshape([DATA.size // 4, 4])
  start = 3000000
  print(np.corrcoef(DATA[start:, XCOL], DATA[start:, YCOL]))
  plt.semilogx(DATA[start:, XCOL], DATA[start:, YCOL], '.')
plt.show()
