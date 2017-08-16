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
#COLMAX = 1e7
#COLMAX = 8e6
#COLMAX = 1e6
COLMAX = -1

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
print(np.mean(DATA[terminals,NREW-1]))
print(NROW, NCOL,ICOL,len(terminals))
inds = np.arange(0,NROW)

for ind in range(IND0, len(terminals), SKIP):
  term = terminals[ind]; term = term[0]
  init =  initials[ind]; init = init[0]
  if COLMAX>0 and term>COLMAX: break; 
  span = range(init, term, 10) 
  print(init,term)
  if XAXIS>=0:
    xes, xtrm = DATA[span,XAXIS], DATA[term,XAXIS]
  else:
    xes, xtrm = inds[span]      , inds[term]
  
  if (ind % 10) == 0:   
    if ind==IND0:
      plt.plot(xes, DATA[span,ICOL], 'b-', label='x-trajectory')
    else:
      plt.plot(xes, DATA[span,ICOL], 'b-')

  #plt.plot(inds, DATA[:,ICOL])
  if ICOL >= NREW: plottrm = term-1
  else: plottrm = term

  if ind==IND0:
    plt.plot(xtrm, DATA[plottrm, ICOL], 'ro', label='terminal x')
  else:
    plt.plot(xtrm, DATA[plottrm,  ICOL], 'ro')
#plt.legend(loc=4)
#plt.ylabel('x',fontsize=16)
#plt.xlabel('t',fontsize=16)
if COLMAX>0:plt.axis([0, COLMAX, -50, 150])
#plt.semilogy(inds, 1/np.sqrt(DATA[:,ICOL]))
plt.tight_layout()
#plt.semilogy(inds[terminals], 1/np.sqrt(DATA[terminals-1,ICOL]), 'ro')

#plt.savefig('prova.png', dpi=100)
plt.show()
