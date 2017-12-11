#
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

plt.subplot(121)
plt.semilogy(abs(W),'b.')
#plt.plot(abs(M),'r.')
plt.subplot(122)
plt.semilogy(abs(M1)/np.sqrt(M2),'g.')
#plt.semilogy(abs(M)/S,'g.')
plt.show()
