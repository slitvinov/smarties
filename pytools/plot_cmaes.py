#/usr/bin/env python
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
# SIMPLE PYTHON SCRIPT TO PLOT .raw weight storage files. Input is path to
# weight files up to the specifier of whether they are weights or adam params.
# (eg /path/to/dir/agent_00_net_)

import sys
import numpy as np
import matplotlib.pyplot as plt
FILE=    sys.argv[1]

W  = np.fromfile(FILE+"weights.raw", dtype=np.float64)
M1 = np.fromfile(FILE+"pathSig.raw", dtype=np.float64)
M2 = np.fromfile(FILE+"pathCov.raw", dtype=np.float64)
M3 = np.fromfile(FILE+"diagCov.raw", dtype=np.float64)

plt.subplot(221)
plt.semilogy(abs(W),'b.')
plt.title('Weights')
#plt.plot(abs(W),'b.')

plt.subplot(222)
plt.semilogy(abs(M1),'k.')
plt.title('pathSig')

plt.subplot(223)
plt.semilogy(abs(M2), 'g.')
#plt.semilogy(abs(M)/S,'g.')
plt.title('pathCov')

plt.subplot(224)
plt.semilogy(abs(M3), 'g.')
plt.title('diagCov')

plt.show()
