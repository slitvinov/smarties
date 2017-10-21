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


#np.savetxt(sys.stdout, np.fromfile(sys.argv[1], dtype='i4').reshape(2,10).transpose())
DATA = np.loadtxt(FILE, delimiter=" ")
print(DATA.shape)
NOUTS = DATA.shape[1]/2
#DATA = DATA.reshape(NROW, NCOL)

for ind in range(0, NOUTS-2):
  plt.subplot(121)
  plt.plot(DATA[:,ind]/DATA[:,NOUTS+ind], label=str(ind))
  #plt.semilogy(abs(DATA[:,ind]),label=str(ind))
  plt.subplot(122)
  plt.semilogy(abs(DATA[:,NOUTS+ind]),'--',  label=str(ind))
  
plt.legend(loc=5, bbox_to_anchor=(1.2, 0.5))
#plt.savefig('prova.png', dpi=100)
plt.show()
