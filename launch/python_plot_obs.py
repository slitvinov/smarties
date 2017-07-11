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
NS  =int(sys.argv[1])
NA  =int(sys.argv[2])
FILE=    sys.argv[3]
ICOL=int(sys.argv[4])
NL=(NA*NA+NA)/2

if len(sys.argv) > 5: NP=int(sys.argv[5])
else: NP = 2*NA

NCOL=3+NS+NA+NP

#np.savetxt(sys.stdout, np.fromfile(sys.argv[1], dtype='i4').reshape(2,10).transpose())
DATA = np.fromfile(FILE, dtype=np.float32)
NROW = DATA.size / NCOL
DATA = DATA.reshape(NROW, NCOL)
print(NROW, NCOL,ICOL)

plt.plot(np.arange(0,NROW), DATA[:,ICOL])
#plt.savefig('prova.png', dpi=100)
plt.show()
