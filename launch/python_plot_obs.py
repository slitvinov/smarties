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

terminals = np.argwhere(abs(DATA[:,0]-2.1)<0.1)
initials  = np.argwhere(abs(DATA[:,0]-1.1)<0.1)
inds = np.arange(0,NROW)

for ind in range(0, len(terminals), 10):
  term = terminals[ind]; term = term[0]
  init =  initials[ind]; init = init[0]
  span = range(init, term) 
  print(init,term)
  plt.plot(inds[span], DATA[span,ICOL])
  #plt.plot(inds, DATA[:,ICOL])
  if ICOL >= NCOL-2:
    plt.plot(inds[term], DATA[term-1,ICOL], 'ro')
  else:
    plt.plot(inds[term], DATA[term,  ICOL], 'ro')

#plt.semilogy(inds, 1/np.sqrt(DATA[:,ICOL]))
#plt.semilogy(inds[terminals], 1/np.sqrt(DATA[terminals-1,ICOL]), 'ro')

#plt.savefig('prova.png', dpi=100)
plt.show()
