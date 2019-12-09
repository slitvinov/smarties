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

  def computeDT(self, ynext): #const
    dx = self.L / self.N
    tanth = (ynext - self.y) / dx
    costh = np.cos(np.arctan(tanth))
    C = 1/(2 * costh * np.sqrt(9.80665) )
    if np.abs(tanth) > 1e-16:
      return C * 2/tanth * (np.sqrt(ynext) - np.sqrt(self.y))
    else:
      return C * dx / np.sqrt(self.y)

  def optimalY(self): #const
    x = self.step * self.L / self.N
    R = ( 1 + self.L*self.L ) / (2*self.L)
    return np.sqrt(2*x*R - x*x)

  def advance(self, ynext):
    if ynext < 1e-2 : ynext = 1e-2
    self.step = self.step + 1 # advance step to get correct optimalY
    # option 1) minimize sum dt (classic approach, difficult)
    self.r = - self.computeDT(ynext)
    # option 2) minimize distance from dt for optimal action, somehow easier
    optimalDT = self.computeDT(self.optimalY())
    r = - np.abs(optimalDT - self.computeDT(ynext))
    # option 3) minimize distance between action and optimal action
    #self.r = - np.abs(ynext - self.optimalY())
    self.y = ynext
    if self.step >= self.N:
      # if option 3) add cost for last DT (would be 0 for other 2 options)
      #self.r = self.r - self.computeDT(1)
      self.step = self.step + 1
      self.y = 1
      return 1
    else: return 0

  def getState(self): return [self.step * self.L/self.N]

  def getReward(self): return self.r

def app_main(comm):
  env = Brachistochrone(np.pi, 100)
  comm.setStateActionDims(1, 1)
  comm.setActionScales([2.0], [0.0], areBounds=False)

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
