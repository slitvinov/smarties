import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
M    = int(sys.argv[1])

means = np.zeros([M, len(range(2,len(sys.argv))) ])
stdev = np.zeros([M, len(range(2,len(sys.argv))) ])
exkrt = np.zeros([M, len(range(2,len(sys.argv))) ])

for j in range(2, len(sys.argv)):
  DATA = np.fromfile(sys.argv[j], dtype=np.float32)
  L = DATA.shape[0] // M
  DATA = DATA[DATA.shape[0] - M*L : -1]
  norm1, norm2 = L ** 0.50, L ** 0.25
  for i in range(M):
    start, end = i*L, (i+1)*L
    x = DATA[start:end] 
    #x = 1000*np.random.randn(L) 
    mean_1 = np.mean(x)
    stdv_1 = np.std(x)
    numer =  np.sum( ((x-mean_1)/norm2)**4 )  
    denom = (np.sum( ((x-mean_1)/norm1)**2 ) )**2
    exkr_1 = numer/denom- 3
    means[i,j-2], stdev[i,j-2], exkrt[i,j-2] = mean_1, stdv_1, exkr_1
    #print (exkr_1, stdv_1)
ret=np.append(np.mean(means, axis=1).reshape(M,1),np.mean(stdev, axis=1).reshape(M,1),axis=1)
ret=np.append(ret,                                np.mean(exkrt, axis=1).reshape(M,1),axis=1)
print(ret)

