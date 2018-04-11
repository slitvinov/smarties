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

import sys, time
import numpy as np
import matplotlib.pyplot as plt
L = 50
colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
nP = len(sys.argv)
lines = [None] * nP
fills = [None] * nP
#plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

while 1:
  for i in range(1, len(sys.argv)):
    PATH =  sys.argv[i]
    FILE = "%s/cumulative_rewards_rank00.dat" % (PATH)
    DATA = np.fromfile(FILE, sep=' ')
    DATA = DATA.reshape(DATA.size // 5, 5)
    N = (DATA.shape[0] // L)*L
    span = DATA.shape[0]-N + np.arange(0, N)
    #print(N, L, DATA.shape[0], span.size) 
    X = DATA[span, 1] - DATA[0,1] 
    Y = DATA[span, 4]
    X = X.reshape(X.size//L, L) 
    Y = Y.reshape(Y.size//L, L) 
    X = X.mean(1)
    Yb = np.percentile(Y, 20, axis=1)
    #Ym = np.percentile(Y, 50, axis=1)
    Ym = np.mean(Y, axis=1)
    Yt = np.percentile(Y, 80, axis=1)
    if fills[i]: a=1#fills[i].set_data(X,Yb,Yt)
    else:
       fills[i]  = ax.fill_between(X,Yb,Yt,facecolor=colors[i-1],alpha=.5)
    if lines[i]: lines[i].set_data(X,Ym)
    else:
       lines[i], = ax.plot(X,Ym, label=PATH, color=colors[i-1])
    fig.canvas.draw()
  plt.show()
  time.sleep(10)

#plt.tight_layout()
#plt.show()
