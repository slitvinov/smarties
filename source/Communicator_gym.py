#!/usr/bin/env python
import gym, sys, socket, os, os.path, time
from gym import wrappers
import numpy as np
from Communicator import Communicator

class Communicator_gym(Communicator):
    def get_env(self):
        assert(self.gym is not None)
        return self.gym

    def __init__(self):
        self.start_server()
        self.sent_stateaction_info = False
        self.discrete_actions = False
        self.number_of_agents = 1
        print("openAI environment: ", sys.argv[2])
        env = gym.make(sys.argv[2])
        nAct, nObs = 1, 1
        actVals = np.zeros([0], dtype=np.float64)
        actOpts = np.zeros([0], dtype=np.float64)
        obsBnds = np.zeros([0], dtype=np.float64)
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
        #if self.bRender==3:
        #    env = gym.wrappers.Monitor(env, './', force=True)
        self.gym = env
        self.seq_id, self.frame_id = 0, 0


if __name__ == '__main__':
    comm = Communicator_gym() # create communicator with smarties
    env = comm.get_env()

    while True: #training loop
        observation = env.reset()
        #send initial state
        comm.send_state(observation, initial=True)

        while True: # simulation loop
            #receive action from smarties
            buf = comm.recv_action()
            if hasattr(env.action_space, 'n'):        action = int(buf[0])
            elif hasattr(env.action_space, 'shape'):  action = buf
            elif hasattr(env.action_space, 'spaces'):
                action = [int(buf[0])]
                for i in range(1, comm.nActions): action = action+[int(buf[i])]
            else: assert(False)

            #advance the environment
            observation, reward, done, info = env.step(action)
            #send the observation to smarties
            comm.send_state(observation, reward=reward, terminal=done)
            if done: break