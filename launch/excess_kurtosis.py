import sys
import numpy as np
import matplotlib.pyplot as plt
FILE =     sys.argv[1]
M    = int(sys.argv[2])
ICOL = int(sys.argv[3])


#np.savetxt(sys.stdout, np.fromfile(sys.argv[1], dtype='i4').reshape(2,10).transpose())
DATA = np.fromfile(FILE, dtype=np.float32)
DATA = DATA.reshape([DATA.size//4, 4])
N = DATA.shape[0]
norm1 = M ** 0.50
norm2 = M ** 0.25
for i in range(0, (N//M)*M, M):
  start = i
  end = start + M
  x = DATA[start:end, ICOL]
  mean_1 = np.mean(x)
  numer =  np.sum( ((x-mean_1)/norm2)**4 )  
  denom = (np.sum( ((x-mean_1)/norm1)**2 ) )**2
  print numer/denom- 3
  

