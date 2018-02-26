#!/usr/bin/env python
import gym, sys, socket, os, os.path, time
from gym import wrappers
import numpy as np
from Communicator import Communicator
import cv2
cv2.ocl.setUseOpenCL(False)

class Communicator_atari(Communicator):
    def get_env(self):
        assert(self.gym is not None)
        return self.gym

    def base_step(self, action):
        total_reward, done = 0.0, None
        for i in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            if i == self.frame_skip - 1: self.buffer[0] = obs
            if i == self.frame_skip - 2: self.buffer[1] = obs
            #if i == self.frame_skip - 3: self.buffer[2] = obs
            total_reward += reward
            if done: break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self.buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def noop_reset(self):
        self.env.reset()
        obs, noops = None, np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0) # 0 is noop action
            if done: obs = self.env.reset()
        return obs

    def life_reset(self):
        if self.was_real_done: obs = self.noop_reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.ale.lives()
        return obs

    def fire_reset(self):
        self.env.reset()
        if(self.env.get_action_meanings()[1] == 'FIRE'):
            obs, _, done, _ = self.env.step(1)
            if done: self.env.reset()
            obs, _, done, _ = self.env.step(2)
            if done: self.env.reset()

        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def life_step(self, action):
        obs, reward, done, info = self.base_step(action)
        self.was_real_done = done
        lives = self.env.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def final_step(self, action):
        obs, reward, done, info = self.life_step(action)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs[:, :, None], reward, done, info

    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    def get_frame(self):
        total_reward = 0.0
        done = None
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            self.was_real_done = done
            # check current lives, make loss of life terminal,
            # then update lives to handle bonus lives
            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives and lives > 0:
                # for Qbert sometimes we stay in lives == 0 condtion for a few frames
                # so its important to keep lives > 0, so that we only reset once
                # the environment advertises done.
                done = True
            self.lives = lives
            total_reward += np.sign(reward)
            if done: break

        return obs, reward, done, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def __init__(self):
        self.start_server()
        self.sent_stateaction_info = False
        self.discrete_actions = False
        self.number_of_agents = 1
        self.frame_skip = 4
        self.stack_curr = 0
        self.noop_max = 30
        self.lives = 0
        self.was_real_done = True
        print("openAI environment: ", sys.argv[2])
        env = gym.make(sys.argv[2])
        self.buffer = np.zeros((2,)+env.observation_space.shape, type=np.uint8)
        nAct, nObs = 1, 84 * 84
        actVals = np.zeros([0], dtype=np.float64)
        actOpts = np.zeros([0], dtype=np.float64)
        obsBnds = np.zeros([0], dtype=np.float64)

        assert( hasattr(env.action_space, 'n') )
        assert( env.unwrapped.get_action_meanings()[0] == 'NOOP' )
        nActions_i = env.action_space.n
        self.discrete_actions = True
        actOpts = np.append(actOpts, [nActions_i, 1])
        actVals = np.append(actVals, np.arange(0,nActions_i)+.1)

        for i in range(nObs):
            obsBnds = np.append(obsBnds, [255, 0])

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
