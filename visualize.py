"""
Visualization for Yvette (:-))
Author: Mojmir 

Usage: 
>>> python visualize.py output.bin

Respond to prompt requests afterwards. 

"""


import sys
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt 
import time 
from matplotlib import animation


rewards = [[]]
coordinates = [[]]
speeds=[[]]
reward = 0.0
suppress=0.1

arguments = []
for arg in sys.argv:
	arguments.append(arg)
file_name = arguments[1]
max_epochs = 10

inp = open(file_name, "r")

epoch = 0
for line in inp:
	line2 = line.strip().split(" ")

	# if epoch>max_epochs:
	# 	break
	# else:
	# 	continue

	if len(line2)>2:
		if line2[1]=="state":
			
			theta = float(line2[2])
			phi = float(line2[5])
			
			x0 = 0.0
			y0 = 0.0
			
			x1 = np.sin(theta)
			y1 = -np.cos(theta)

			x2 = np.sin(phi+theta) + x1
			y2 = -np.cos(phi+theta) + y1

			v1 = float(line2[3])*suppress
			v2 = float(line2[4])*suppress

			#coordinates[epoch].append([x0,y0,x1,y1,x2,y2])
			coordinates[epoch].append([(x0,x1,x2),(y0,y1,y2)])
			#speeds[epoch].append([(x1,y1),(x1+v1*np.cos(theta),y1+v1*np.sin(theta)),(x2,y2),(x2+v2*np.cos(theta+phi),y2+v2*np.sin(theta+phi))])
                        speeds[epoch].append([(x1,x1+v1*np.cos(theta)),(y1,y1+v1*np.sin(theta)),(x2,x2+v2*np.cos(theta+phi)),(y2,y2+v2*np.sin(theta+phi))])

		elif line2[1]=="reward":
			reward = reward + float(line2[2])
			rewards[epoch].append(float(reward))

		elif line2[1]=="term":
			epoch = epoch + 1 
			coordinates.append([])
			rewards.append([])
			speeds.append([])

			reward = 0 


def visualize_epoch(epoch,data,animation_speed=50, V=None):

	fig, ax = plt.subplots()
	plt.grid(True)
	plt.plot([-3,3],[0,0],'k',lw=5)
	im, = ax.plot(0,0,marker="o",markersize=20, animated=True)
	ax.text(-1.5, 2, "Run: "+str(epoch),	bbox={'facecolor':'blue', 'alpha':0.5, 'pad':5})
        im2 = ax.text(-1.5, 1.5, "Iteration: ", bbox={'facecolor':'blue', 'alpha':0.5, 'pad':5}, animated=True)
        im3 = ax.text(-1.5, 2.5, "Cumulative reward: ", bbox={'facecolor':'blue', 'alpha':0.5, 'pad':5}, animated=True)
	
	if V is not None:
		#im4 = ax.arrow(x1, y1, x1+v1*np.cos(theta), y1+v1*np.sin(theta), head_width=0.10, head_length=0.1, fc='r', ec='r')
		#im5 = ax.arrow(x2, y2, x2+v2*np.cos(theta+phi), y2+v2*np.sin(theta+phi), head_width=0.05, head_length=0.1, fc='r', ec='r')

		im4, = ax.plot(V[epoch][0][0],V[epoch][0][1],'r',lw=4.0, animated=True)
		im5, = ax.plot(V[epoch][0][2],V[epoch][0][3],'r',lw=4.0, animated=True)
	
	def animate(i):
		plt.xlim([-3,3])
		plt.ylim([-3,3])
		xdata = data[epoch][i][0]
		ydata = data[epoch][i][1]

		im.set_data(xdata, ydata)

		im2.set_text("Iteration: "+str(i)+ "/" + str(data[epoch].shape[0]))
		im3.set_text("Cumulative reward: "+str(rewards[epoch][i])+ "/" + str(rewards[epoch][data[epoch].shape[0]-1]))
		if V is not None:

			im4.set_data(V[epoch][i][0],V[epoch][i][1])
			im5.set_data(V[epoch][i][2],V[epoch][i][3])

			return [im,im2,im3,im4,im5]
			#return [im,im2,im3]

		else:
			return [im,im2,im3]

	anim = animation.FuncAnimation(fig, animate, frames=data[epoch].shape[0], interval=animation_speed, blit=True )
	plt.show()


X = []
V = []
for epoch in coordinates:
	X.append(np.array(epoch))
for epoch in speeds:
	V.append(np.array(epoch))


print "Number of runs loaded:", len(X)
e = int(raw_input("What run to simulate? [0 - " + str(len(X)-1) + "]"))
simulation_speed = int(raw_input('Speed of simulation [1 - 200] (fast - slow) ?'))
visualize_epoch(e,X,V=V,animation_speed = simulation_speed )
