#!/usr/bin/env python3
##
##  smarties
##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.
##
##  Created by Guido Novati (novatig@ethz.ch).
##

import numpy as np, sys
import smarties as rl

class Brachistochrone:
  def __init__(self, L, N):
    self.N = N
    self.L = L
    self.step = 1
    self.r = 0
    self.y = 0

  def reset(self):
    self.step = 1
    self.r = 0
    self.y = 0

  def computeDT(self, ynext):
    dx = self.L / self.N
    tanth = (ynext - self.y) / dx
    costh = np.cos(np.arctan(tanth))
    C = 1/(2 * costh * np.sqrt(9.80665) )
    if np.abs(tanth) > 1e-16:
      return C * 2/tanth * (np.sqrt(ynext) - np.sqrt(self.y))
    else:
      return C * dx / np.sqrt(self.y)
    
  def advance(self, ynext):
    if ynext < 1e-16 : ynext = 1e-16
    self.r = - self.computeDT(ynext)
    self.step = self.step + 1
    self.y = ynext
    if self.step >= self.N:
      self.r = self.r - self.computeDT(1)
      self.step = self.step + 1
      self.y = 1
      return 1
    else: return 0

  def getState(self):  return [self.step * self.L/self.N]

  def getReward(self): return self.r

def app_main(comm):
  env = Brachistochrone(np.pi, 50)
  comm.set_state_action_dims(1, 1)
  comm.set_action_scales([2.0], [0.0], areBounds=False)

  while 1: #train loop, each new episode starts here
    env.reset()
    comm.sendInitState(env.getState());
    while 1: #simulation loop
      action = comm.recvAction();
      terminated = env.advance(action[0]);
      if terminated:
        comm.sendTermState(env.getState(), env.getReward());
        break
      else: comm.sendState(env.getState(), env.getReward());

if __name__ == '__main__':
  e = rl.Engine(sys.argv)
  if( e.parse() ): exit()
  e.run( app_main )
