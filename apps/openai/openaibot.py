#!/usr/bin/env python
import gym
from gym import wrappers
import matplotlib.pyplot as plt
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
        actionOptions = np.append(actionOptions, [nActions_i+.1, 1])
        actionValues = np.append(actionValues,np.arange(0,nActions_i)+.1)
elif hasattr(env.action_space, 'shape'):
    nActions = env.action_space.shape[0]
    for i in range(nActions):
        bounded = 0 #figure out if environment is strict about the bounds on action:
        test = env.reset()
        test_act = 0.5*(env.action_space.low + env.action_space.high)
        test_act[i] = env.action_space.high[i]+1
        try: test = env.step(test_act)
        except: bounded = 1.1
        env.reset()
        actionOptions = np.append(actionOptions, [2.1, bounded])
        actionValues = np.append(actionValues,max(env.action_space.low[i],-1e3))
        actionValues = np.append(actionValues,min(env.action_space.high[i],1e3))
elif hasattr(env.action_space, 'n'):
    nActions_i = env.action_space.n
    actionOptions = np.append(actionOptions, [nActions_i, 1])
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
bRender = int(round(np.frombuffer(conn.recv(8), dtype=np.float64)[0]))
#print(bRender)
sys.stdout.flush()
if bRender==3: env = gym.wrappers.Monitor(env, './', force=True)

seq_id, frame_id = 0, 0
state=np.zeros(nStates+3, dtype=np.float64)
while True:
    seq_id += 1
    reward, status, frame_id = 0, 1, 0
    observation = env.reset()
    while True:
        state[0]=0
        state[1]=status
        if hasattr(env.observation_space, 'shape'):
            state[2:nStates+2]=observation.ravel()
        else: state[2] = observation
        state[nStates+2]=reward
        for i in range(nStates+2): assert(not np.isnan(state[i]))
        #print(state)

        conn.send(state.tobytes())
        status=0
        if bRender==1: env.render()
        if bRender==2:
            fname = 'state_seq%04d_frame%07d' % (seq_id, frame_id)
            plt.imshow(env.render(mode='rgb_array'))
            plt.savefig(fname, dpi=100)

        buf = np.frombuffer(conn.recv(nActions*8),dtype=np.float64)
        for i in range(nActions): assert(not np.isnan(buf[i]))
        if hasattr(env.action_space, 'shape'):
            action = buf
        elif hasattr(env.action_space, 'spaces'):
            action = [int(buf[0])]
            for i in range(1, nActions): action = action + [int(buf[i])]
        else: action = int(buf[0])
        #print(action)
        reward = 0
        for i in range(1):
            observation, instreward, done, info = env.step(action)
            reward += instreward
            if done: break
        if done:
            if bRender==1: env.render()
            if bRender==2:
                fname = 'state_seq%04d_frame%07d' % (seq_id, frame_id)
                plt.imshow(env.render(mode='rgb_array'))
                plt.savefig(fname, dpi=100)
            break
        frame_id += 1

    state[0]=0
    state[1]=2
    if hasattr(env.observation_space, 'shape'):
        state[2:nStates+2]=observation.ravel()
    else: state[2] = observation
    state[nStates+2]=reward
    for i in range(nStates+2): assert(not np.isnan(state[i]))
    #print(state)
    conn.send(state.tobytes())
    buf = np.frombuffer(conn.recv(nActions*8),dtype=np.float64)
    #print(buf)
    if(buf[0]<0):
        print("Received end of training signal. Aborting...");
        break

conn.close()
