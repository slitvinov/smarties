/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "CartEnvironment.h"
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

GliderEnvironment::GliderEnvironment(const int _nAgents, const string _execpath,
																 const int _rank, Settings & settings) :
Environment(_nAgents, _execpath, _rank, settings) { }

void GliderEnvironment::setDims() //this environment is for the cart pole test
{
		sI.inUse.clear();
		for (int i=0; i<9; i++) sI.inUse.push_back(true);

    aI.dim = 1; //number of action that agent can perform per turn: usually 1 (eg DQN)
    aI.values.resize(aI.dim);
    for (int i=0; i<aI.dim; i++) {
        aI.values[i].push_back(-1.); //here the app accepts real numbers
        aI.values[i].push_back(1.);
    }

    commonSetup(); //required
}

bool GliderEnvironment::pickReward(const State & t_sO, const Action & t_a,
																 const State& t_sN, Real& reward,const int info)
{
    const bool new_sample = info==2;
    if (new_sample) reward *= 1./(1.-gamma); // = - max cumulative reward
    return new_sample; //cart pole has failed if r = -1, need to clean this shit and rely only on info
}
