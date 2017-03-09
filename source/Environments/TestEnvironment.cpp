/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "TestEnvironment.h"
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <cstdio>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <iostream>
#include <algorithm>
#include <stdio.h>

using namespace std;

TestEnvironment::TestEnvironment(const int _nAgents, const string _execpath,
																 const int _rank, Settings & settings) :
Environment(_nAgents, _execpath, _rank, settings)
{
}

void TestEnvironment::setDims() //this environment is for the cart pole test
{
		mpi_ranks_per_env = 2;
		paramsfile="params.txt";
    {
      sI.inUse.clear();
      //for each state variable:
      // State: coordinate...
      sI.inUse.push_back(true); //ignore, leave as is

      // ...velocity...
			sI.inUse.push_back(true); //ignore, leave as is

      // ...and angular velocity
			sI.inUse.push_back(true); //ignore, leave as is

      // ...angle...
			sI.inUse.push_back(true); //ignore, leave as is

    }
    {
      aI.dim = 2; //number of action that agent can perform per turn: usually 1 (eg DQN)
      aI.values.resize(aI.dim);
      for (int i=0; i<aI.dim; i++) {
      	//used if discrete actions: options available to agent for acting
				//otherwise can be used for scaling
          aI.values[i].push_back(-1.);
          aI.values[i].push_back(1.0);
      }
    }
    commonSetup(); //required
}
