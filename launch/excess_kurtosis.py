import sys
import numpy as np
import matplotlib.pyplot as plt
FILE=    sys.argv[1]


#np.savetxt(sys.stdout, np.fromfile(sys.argv[1], dtype='i4').reshape(2,10).transpose())
DATA = np.fromfile(FILE, dtype=np.float32)
DATA = DATA.reshape([DATA.size//4, 4])
N = DATA.shape[0]
M = 100000
for i in range(0, (N//M)*M, M):
  start = i
  end = start + M
  x = DATA[start:end, 2]
  mean_1 = np.mean(x)
  numer =  np.sum((x-mean_1)**4 /M)  
  denom = (np.sum((x-mean_1)**2 /M) )**2
  print numer/denom- 3
  

