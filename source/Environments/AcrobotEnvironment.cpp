/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016, modified by Iveta Rott on January 8,2017
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "AcrobotEnvironment.h"

AcrobotEnvironment::AcrobotEnvironment(const Uint _nAgents, const string _execpath, Settings & _s) :
Environment(_nAgents, _execpath, _s) {}


void AcrobotEnvironment::setDims() //this environment is for the cart pole test
{
  {
    sI.inUse.clear();
    sI.inUse.push_back(true);
		sI.inUse.push_back(true);
		sI.inUse.push_back(true);
		sI.inUse.push_back(true);
		sI.inUse.push_back(true);
		sI.inUse.push_back(true);
  }
  {
    aI.dim = 1;
    aI.values.resize(1);
    aI.values[0].push_back(-2.); //here the app accepts real numbers
    aI.values[0].push_back( 2.);
		aI.bounded.push_back(1);
  }
  commonSetup(); //required
}

bool AcrobotEnvironment::pickReward(const State & t_sO, const Action & t_a,
				const State& t_sN, Real& reward,const int info)
{
  return info==2; 
}
