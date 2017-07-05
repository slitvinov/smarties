import sys
import numpy as np
import matplotlib.pyplot as plt

NS  =int(sys.argv[1])
NA  =int(sys.argv[2])
FILE=    sys.argv[3]
ICOL=int(sys.argv[4])

NL=(NA*NA+NA)/2
NCOL=3+NS+NA+2*NA

#np.savetxt(sys.stdout, np.fromfile(sys.argv[1], dtype='i4').reshape(2,10).transpose())
DATA = np.fromfile(FILE, dtype=np.float32)
NROW = DATA.size / NCOL
DATA = DATA.reshape(NROW, NCOL)
print(NROW, NCOL,ICOL)

plt.plot(np.arange(0,NROW), DATA[:,ICOL])
#plt.savefig('prova.png', dpi=100)
