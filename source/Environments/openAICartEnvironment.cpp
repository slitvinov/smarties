/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "openAICartEnvironment.h"
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

openAICartEnvironment::openAICartEnvironment(const int _nAgents, const string _execpath, Settings & _s) :
Environment(_nAgents, _execpath, _s), allSenses(_s.senses==0)
{
//   cheaperThanNetwork=false;
}

void openAICartEnvironment::setDims() //this environment is for the cart pole test
{
    comm_ptr->getStateActionShape(aI.values, sI.mean, sI.scale);
    vector<Real> upper = sI.mean;
    vector<Real> lower = sI.scale;
    for (unsigned i=0; i<upper.size(); i++) {
      sI.mean[i]  = 0.5*(upper[i]+lower[i]);
      sI.scale[i] = 0.5*(upper[i]-lower[i]);
      assert(sI.scale[i]>0);
    }
    aI.dim = aI.values.size();
    sI.inUse.resize(sI.mean.size(), true);
    aI.bounded.resize(aI.dim,1);
    commonSetup(); //required
}
