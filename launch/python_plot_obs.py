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

if len(sys.argv) > 6: SKIP=int(sys.argv[6])
else: SKIP = 10

if len(sys.argv) > 7: XAXIS=int(sys.argv[7])
else: XAXIS = -1

if len(sys.argv) > 8: IND0=int(sys.argv[8])
else: IND0 = 0

NREW=3+NS+NA
NCOL=3+NS+NA+NP

#np.savetxt(sys.stdout, np.fromfile(sys.argv[1], dtype='i4').reshape(2,10).transpose())
DATA = np.fromfile(FILE, dtype=np.float32)
NROW = DATA.size / NCOL
DATA = DATA.reshape(NROW, NCOL)

terminals = np.argwhere(abs(DATA[:,0]-2.1)<0.1)
initials  = np.argwhere(abs(DATA[:,0]-1.1)<0.1)
print(NROW, NCOL,ICOL,len(terminals))
inds = np.arange(0,NROW)

for ind in range(IND0, len(terminals), SKIP):
  term = terminals[ind]; term = term[0]
  init =  initials[ind]; init = init[0]
  span = range(init, term) 
  print(init,term)
  if XAXIS>=0:
    xes, xtrm = DATA[span,XAXIS], DATA[term,XAXIS]
  else:
    xes, xtrm = inds[span], inds[term]
  plt.plot(xes, DATA[span,ICOL])
  #plt.plot(inds, DATA[:,ICOL])
  if ICOL >= NREW:
    plt.plot(xtrm, DATA[term-1,ICOL], 'ro')
  else:
    plt.plot(xtrm, DATA[term,  ICOL], 'ro')

#plt.semilogy(inds, 1/np.sqrt(DATA[:,ICOL]))
#plt.semilogy(inds[terminals], 1/np.sqrt(DATA[terminals-1,ICOL]), 'ro')

#plt.savefig('prova.png', dpi=100)
plt.show()
