#/usr/bin/env python
# SIMPLE PYTHON SCRIPT TO PLOT .raw AGENT OBS FILES
#
# usage:
# python python_plot_obs.py len_state_vec len_action_vec path/to/file.raw column_ID_to_plot
# also, optional: ( number_of_elements_in_policy_vector )
# otherwise, assumed continuous pol with 2*NA components (mean, precision of gaussian)
#
# structure of .raw files is:
# transition_id [0/1/2] [state] [action] [reward] [policy]
# (second column is 1 for first observation of an episode, 2 for last)

import sys
import numpy as np
import matplotlib.pyplot as plt
FILE=    sys.argv[1]

W  = np.fromfile(FILE+"weights.raw", dtype=np.float64)
M1 = np.fromfile(FILE +"1stMom.raw", dtype=np.float64)
M2 = np.fromfile(FILE +"2ndMom.raw", dtype=np.float64)
INDS = np.where(M2>1e-16)
#INDS = np.arange(64768,73984)
#INDS = np.reshape(INDS, [128,72])
#INDS = INDS[:, 1:36]
#INDS = INDS[:, 37:54]
#INDS = INDS[:, 54:71]
#INDS = INDS[:]
W  = W[INDS];
M1 = M1[INDS];
M2 = M2[INDS];
assert(not np.isnan(W).any())
assert(not np.isnan(M1).any())
assert(not np.isnan(M2).any())

plt.subplot(231)
plt.semilogy(1e-6 + abs(W),'b.')
plt.title('Weights')
#plt.plot(abs(W),'b.')

plt.subplot(232)
plt.semilogy(abs(M1) + 1e-12,'k.')
plt.title('Abs M1')

plt.subplot(233)
plt.semilogy(1e-6 + np.sqrt(M2), 'g.')
#plt.semilogy(abs(M)/S,'g.')
plt.title('Sqrt M2')

plt.subplot(234)
plt.semilogy(abs(M1)/(1e-7 + np.sqrt(M2)) + 1e-6,'g.')
plt.title('Abs Grad')

plt.subplot(235)
plt.semilogy((abs(W)*2.2e-16)/(np.sqrt(M2)),'k.')


plt.subplot(236)
#plt.semilogy(abs(W)*1.19209e-07 + 1e-12,'r.',alpha=0.2)
plt.semilogy((abs(W)*2.2e-16)/(abs(M1)),'k.')

plt.show()
