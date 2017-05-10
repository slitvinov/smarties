#!/usr/bin/env python
import gym
import sys
import socket
import os, os.path
import time
import numpy as np

sockid = sys.argv[1]
server_address = "/tmp/smarties_sock_"+str(sockid)


try:
    os.unlink(server_address)
except OSError:
    if os.path.exists(server_address):
        raise

server = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(server_address)

server.listen(1)
conn, addr = server.accept()
print( 'Connected by', addr )

print("openAI environment: ")
print (sys.argv[2])
env = gym.make(sys.argv[2])

nActions=1
print("nepisodes:")
print(sys.argv[3])
print("ntimesteps:")
print(sys.argv[4])
while True:
	status=1
	agent_id=0 #assume only 1 exist for now
	reward=0
	observation=np.zeros_like(env.reset(), dtype=np.float64)
	state=np.zeros(observation.size+3)
	for i_episode in range(int(float(sys.argv[3]))):
		print("env.reset()")
		observation = env.reset()
		for t in range(int(float(sys.argv[4]))):
			state[0]=agent_id
			state[1]=status
			state[2:observation.size+2]
			state[observation.size+2]=reward
			conn.send(state.tobytes())
			status=0
			buf=conn.recv(nActions*8)
			action=np.frombuffer(buf,dtype=np.float64)
			print(action)
			observation, reward, done, info = env.step(int(action[0]))
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				break
		info=2
		state[0]=agent_id
		state[1]=status
		state[2:observation.size+2]
		state[observation.size+2]=reward
		conn.send(state.tobytes())
		info=1
		reward=0




