/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "BlowfishEnvironment.h"

BlowfishEnvironment::BlowfishEnvironment(const Uint _nAgents, const string _execpath, Settings & _s) : Environment(_nAgents, _execpath, _s) { }

void BlowfishEnvironment::setDims() //this environment is for the cart pole test
{
  sI.inUse.clear();
  sI.scale.clear();
  sI.mean.clear();
  //blowfish receives 8 dimensional observation vector:
  // {U, V, omega, angle, flapAng_R, flapAng_L, flapAngVel_R, flapAngVel_L}
  sI.inUse.resize(8,true);
  sI.scale.resize(8,1);
  sI.mean.resize(8,0);

  //blowfish sets angular velocity for 2 fins
  aI.values.clear();
  aI.bounded.clear();
  aI.dim = 2;
  aI.values.resize(2);
  aI.values[0] = vector<Real>{-1., 1.}; //scale of the two actions is the same
  aI.values[1] = vector<Real>{-1., 1.}; //-1 to 1, so no scaling
  aI.bounded.resize(2,false); //unbounded action space

  commonSetup(); //required
}
