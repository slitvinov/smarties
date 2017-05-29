#!/usr/bin/env python
import gym
import sys
import socket
import os, os.path
import time
import numpy as np
sockid = sys.argv[1]
server_address = "/tmp/smarties_sock"+str(sockid)

try:
    os.unlink(server_address)
except OSError:
    if os.path.exists(server_address):
        raise

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(server_address)
server.listen(1)
conn, addr = server.accept()
#time.sleep(1)
#conn = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
#conn.connect( server_address )

print("openAI environment: ")
print (sys.argv[2])
env = gym.make(sys.argv[2])

nActions=1
actionValues = np.zeros([0],dtype=np.float64)
actionOptions = np.zeros([0],dtype=np.float64)
state_bounds = np.zeros([0],dtype=np.float64)

if hasattr(env.action_space, 'spaces'):
    nActions = len(env.action_space.spaces)
    for i in range(nActions):
        nActions_i = env.action_space.spaces[i].n
        actionOptions = np.append(actionOptions, nActions_i+.1)
        actionValues = np.append(actionValues,np.arange(0,nActions_i)+.1)
elif hasattr(env.action_space, 'shape'):
    nActions = env.action_space.shape[0]
    for i in range(nActions):
        actionOptions = np.append(actionOptions, 2.1)
        actionValues = np.append(actionValues,env.action_space.low[i])
        actionValues = np.append(actionValues,env.action_space.high[i])
elif hasattr(env.action_space, 'n'):
    nActions_i = env.action_space.n
    actionOptions = np.append(actionOptions, nActions_i)
    actionValues = np.append(actionValues,np.arange(0,nActions_i)+.1)
else: assert(False)

if hasattr(env.observation_space, 'shape'):
    nStates = 1
    for i in range(len(env.observation_space.shape)):
        nStates *= env.observation_space.shape[i]
    for i in range(nStates):
        if(env.observation_space.high[i]<1e3 and env.observation_space.low[i]>-1e3):
            state_bounds = np.append(state_bounds, env.observation_space.high[i])
            state_bounds = np.append(state_bounds, env.observation_space.low[i])
        else: #no scaling
            state_bounds = np.append(state_bounds, 1)
            state_bounds = np.append(state_bounds, -1)
elif hasattr(env.observation_space, 'n'):
    nStates = 1
    state_bounds = np.append(state_bounds, env.observation_space.n)
    state_bounds = np.append(state_bounds, 0)
else: assert(False)

conn.send(np.array([nStates+.1,nActions+.1],np.float64).tobytes())
conn.send(state_bounds.tobytes())
conn.send(actionOptions.tobytes())
conn.send(actionValues.tobytes())

state=np.zeros(nStates+3)
while True:
    info,reward,status=1,0,1
    observation = env.reset()
    while True:
    	state[0]=0
    	state[1]=status
        if hasattr(env.observation_space, 'shape'):
    	       state[2:nStates+2]=observation.ravel()
        else: state[2] = observation
    	state[nStates+2]=reward
    	conn.send(state.tobytes())
    	status=0
        env.render()
    	buf = np.frombuffer(conn.recv(nActions*8),dtype=np.float64)

        if hasattr(env.action_space, 'shape'):
            action = buf
        elif hasattr(env.action_space, 'spaces'):
            action = [int(buf[0])]
            for i in range(1, nActions): action = action + [int(buf[i])]
        else: action = int(buf[0])

        for i in range(4):
            observation, reward, done, info = env.step(action)
            if done: break
        if done: break
    state[0]=0
    state[1]=2
    if hasattr(env.observation_space, 'shape'):
           state[2:nStates+2]=observation.ravel()
    else: state[2] = observation
    state[nStates+2]=reward
    conn.send(state.tobytes())
    buf = np.frombuffer(conn.recv(nActions*8),dtype=np.float64)
    if(buf[0]<0):
        print("Received end of training signal. Aborting...");
        break
conn.close()
