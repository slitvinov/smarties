#!/usr/bin/env python3
##
##  smarties
##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.
##
##  Created by Guido Novati (novatig@ethz.ch).
##

import sys, socket, os, os.path, time, numpy as np
os.environ['DISABLE_MUJOCO_RENDERING'] = '1'
from dm_control import suite
import smarties as rl

def app_main(comm):
    print("DeepMind Control Suite environment: ",
           sys.argv[1], "task: ", sys.argv[2])
    env = suite.load(domain_name=sys.argv[1], task_name=sys.argv[2])
    act_spec, obs_spec = env.action_spec(), env.observation_spec()
    dimState, dimAction = 0, act_spec.shape[0]
    print(dimState, dimAction)
    sys.stdout.flush()
    for component in obs_spec.values():
        if len(component.shape): dimState = dimState + component.shape[0]
        else: dimState = dimState + 1
    upprActScale, lowrActScale = dimAction * [ 1.0], dimAction * [-1.0]
    isBounded = True # all bounded in DMC
    comm.setStateActionDims(dimState, dimAction, 0) # 1 agent
    comm.setActionScales(upprActScale, lowrActScale, True, 0)

    while True: #training loop
        t = env.reset()
        obsVec = np.zeros([0], dtype=np.float64)
        for oi in t.observation.values(): obsVec = np.append(obsVec, oi)
        comm.sendInitState(obsVec.ravel()) #send initial state

        while True: # simulation loop
            action = comm.recvAction() #receive action from smarties
            t = env.step(action) #advance the environment
            obs, rew, step = t.observation, t.reward, t.step_type.value
            obsVec = np.zeros([0], dtype=np.float64)
            for oi in obs.values(): obsVec = np.append(obsVec, oi)
            #send the observation to smarties
            if t.last(): # DMC does not have term condition, just truncated seqs
                comm.sendLastState(obsVec.ravel(), rew)
                break
            else: comm.sendState(obsVec.ravel(), rew)

if __name__ == '__main__':
  e = rl.Engine(sys.argv)
  if( e.parse() ): exit()
  e.run( app_main )
