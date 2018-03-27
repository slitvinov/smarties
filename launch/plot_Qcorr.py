import numpy as np, matplotlib.pyplot as plt, sys

PATH = sys.argv[1]
A = np.fromfile(PATH+'/onpolQdist.raw',dtype=np.float32)
A = A.reshape([A.size//3,3])
plt.plot(abs(A[:,0]-A[:,1]),'.')
plt.show()

