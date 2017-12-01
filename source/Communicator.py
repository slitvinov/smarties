#!/usr/bin/env python
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import sys
import socket
import os, os.path
import time
import numpy as np

class Communicator:
    def start_server(self):
        #read from argv identifier for communication:
        sockid = sys.argv[1]
        server_address = "/tmp/smarties_sock"+str(sockid)
        try: os.unlink(server_address)
        except OSError:
            if os.path.exists(server_address): raise

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(server_address)
        server.listen(1)
        self.conn, addr = server.accept()
        #time.sleep(1)
        #conn = socket.socket( socket.AF_UNIX, socket.SOCK_STREAM )
        #conn.connect( server_address )

    def set_action_scales(self, upper, lower, bounded=False):
        actVals = np.zeros([0],dtype=np.float64)
        actOpts = np.zeros([0],dtype=np.float64)
        if bounded: bound=1.0
        else: bound=0.0;
        assert(len(upper) == self.nActions and len(lower) == self.nActions)
        for i in range(self.nActions):
            actOpts = np.append(actOpts, [2.1, bound])
            actVals = np.append(actVals, [lower[i], upper[i]])
        self.action_options = actOpts
        self.action_bounds = actVals

    def set_state_scales(self, upper, lower):
        obsBnds = np.zeros([0],dtype=np.float64)
        assert(len(upper) == self.nStates and len(lower) == self.nStates)
        for i in range(self.nStates):
            obsBnds = np.append(obsBnds, [upper[i], lower[i]])
        self.observation_bounds = obsBnds

    def set_state_observable(self, observable):
        self.obs_in_use = np.zeros([self.nStates],dtype=np.float64)
        assert(len(observable) == self.nStates)
        for i in range(self.nStates): self.obs_in_use[i] = observable[i]

    def __init__(self, state_components = 0, action_components = 0, number_of_agents = 1):
        self.start_server()
        self.sent_stateaction_info = False
        self.discrete_actions = False
        self.number_of_agents = number_of_agents
        if(state_components==0 or action_components==0):
            self.gym = self.read_gym_env()
        else:
            self.nActions, self.nStates = action_components, state_components
            actVals = np.zeros([0],dtype=np.float64)
            actOpts = np.zeros([0],dtype=np.float64)
            obsBnds = np.zeros([0],dtype=np.float64)
            for i in range(action_components):
                #tell smarties unbounded (0) continuous actions
                actOpts = np.append(actOpts, [2.1, 0.])
                #tell smarties non-rescaling bounds -1 : 1
                actVals = np.append(actVals, [1, -1])
            for i in range(state_components):
                #tell smarties non-rescaling bounds -1 : 1.
                obsBnds = np.append(obsBnds, [1, -1])
            self.obs_in_use = np.ones(state_components, dtype=np.float64)
            self.observation_bounds = obsBnds
            self.action_options = actOpts
            self.action_bounds = actVals
            self.gym = None
        self.seq_id, self.frame_id = 0, 0

    def __del__(self):
        self.conn.close()

    def read_gym_env(self):
        print("openAI environment: ", sys.argv[2])
        env = gym.make(sys.argv[2])
        nAct, nObs = 1, 1
        actVals = np.zeros([0],dtype=np.float64)
        actOpts = np.zeros([0],dtype=np.float64)
        obsBnds = np.zeros([0],dtype=np.float64)
        if hasattr(env.action_space, 'spaces'):
            nAct = len(env.action_space.spaces)
            self.discrete_actions = True
            for i in range(nAct):
                nActions_i = env.action_space.spaces[i].n
                actOpts = np.append(actOpts, [nActions_i+.1, 1])
                actVals = np.append(actVals, np.arange(0,nActions_i)+.1)
        elif hasattr(env.action_space, 'n'):
            nActions_i = env.action_space.n
            self.discrete_actions = True
            actOpts = np.append(actOpts, [nActions_i, 1])
            actVals = np.append(actVals, np.arange(0,nActions_i)+.1)
        elif hasattr(env.action_space, 'shape'):
            nAct = env.action_space.shape[0]
            for i in range(nAct):
                bounded = 0 #figure out if environment is strict about the bounds on action:
                test = env.reset()
                test_act = 0.5*(env.action_space.low + env.action_space.high)
                test_act[i] = env.action_space.high[i]+1
                try: test = env.step(test_act)
                except: bounded = 1.1
                env.reset()
                actOpts = np.append(actOpts, [2.1, bounded])
                actVals = np.append(actVals, max(env.action_space.low[i],-1e3))
                actVals = np.append(actVals, min(env.action_space.high[i],1e3))
        else: assert(False)

        if hasattr(env.observation_space, 'shape'):
            for i in range(len(env.observation_space.shape)):
                nObs *= env.observation_space.shape[i]

            for i in range(nObs):
                if(env.observation_space.high[i]<1e3 and env.observation_space.low[i]>-1e3):
                    obsBnds = np.append(obsBnds, env.observation_space.high[i])
                    obsBnds = np.append(obsBnds, env.observation_space.low[i])
                else: #no scaling
                    obsBnds = np.append(obsBnds, [1, -1])
        elif hasattr(env.observation_space, 'n'):
            obsBnds = np.append(obsBnds, env.observation_space.n)
            obsBnds = np.append(obsBnds, 0)
        else: assert(False)

        self.obs_in_use = np.ones(nObs, dtype=np.float64)
        self.nActions, self.nStates = nAct, nObs
        self.observation_bounds = obsBnds
        self.action_options = actOpts
        self.action_bounds = actVals
        self.send_stateaction_info()
        if self.bRender==3:
            env = gym.wrappers.Monitor(env, './', force=True)
        return env

    def send_stateaction_info(self):
        if(not self.sent_stateaction_info):
            sizes_ary = np.array([self.nStates+.1, self.nActions+.1, self.discrete_actions+.1, self.number_of_agents+.1],np.float64)
            print(sizes_ary); sys.stdout.flush()
            self.conn.send(sizes_ary.tobytes())
            self.conn.send(self.obs_in_use.tobytes())
            self.conn.send(self.observation_bounds.tobytes())
            self.conn.send(self.action_options.tobytes())
            self.conn.send(self.action_bounds.tobytes())
            self.bRender = np.frombuffer(self.conn.recv(8), dtype=np.float64)
            self.bRender = int(round(self.bRender[0]))
            #print(bRender); sys.stdout.flush()
            self.sent_stateaction_info = True

    def send_state(self, observation, reward=0, terminal=False, initial=False, agent_id = 0):
        if initial: self.seq_id, self.frame_id = self.seq_id+1, 0
        self.frame_id = self.frame_id + 1
        self.send_stateaction_info()
        assert(agent_id<self.number_of_agents)
        state = np.zeros(self.nStates+3, dtype=np.float64)
        state[0] = agent_id
        if terminal:  state[1] = 2.1
        elif initial: state[1] = 1.1
        else:         state[1] = 0.1
        if hasattr(observation, 'shape'):
            assert( self.nStates == observation.size )
            state[2:self.nStates+2] = observation.ravel()
        else:
            assert( self.nStates == 1 )
            state[2] = observation
        state[self.nStates+2] = reward
        for i in range(self.nStates+2): assert(not np.isnan(state[i]))
        #print(state); sys.stdout.flush()
        self.conn.send(state.tobytes())
        #if self.bRender==1 and self.gym is not None: self.gym.render()
        #if self.bRender==2 and self.gym is not None:
        #    seq_id, frame_id = 0, 0
        #    fname = 'state_seq%04d_frame%07d' % (seq_id, frame_id)
        #    plt.imshow(self.gym.render(mode='rgb_array'))
        #    plt.savefig(fname, dpi=100)

    def recv_action(self):
        buf = np.frombuffer(self.conn.recv(self.nActions*8), dtype=np.float64)
        for i in range(self.nActions): assert(not np.isnan(buf[i]))

        if self.gym is not None:
            if hasattr(self.gym.action_space, 'n'):
                action = int(buf[0])
            elif hasattr(self.gym.action_space, 'shape'):
                action = buf
            elif hasattr(self.gym.action_space, 'spaces'):
                action = [int(buf[0])]
                for i in range(1, self.nActions): action = action+[int(buf[i])]
            else: assert(False)
        else: action = buf
        if abs(buf[0]+256)<2.2e-16: quit()
        return action

    def get_gym_env(self):
        assert(self.gym is not None)
        return self.gym

if __name__ == '__main__':
    comm = Communicator() # create communicator with smarties
    env = comm.get_gym_env()

    while True: #training loop
        observation = env.reset()
        #send initial state
        comm.send_state(observation, initial=True)

        while True: # simulation loop
            #receive action from smarties
            action = comm.recv_action()
            #advance the environment
            observation, reward, done, info = env.step(action)
            #send the observation to smarties
            comm.send_state(observation, reward=reward, terminal=done)
            if done: break
