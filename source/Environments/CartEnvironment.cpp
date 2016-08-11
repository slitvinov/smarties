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

CartEnvironment::CartEnvironment(const int nAgents, const string execpath, const int _rank, Settings & settings) :
Environment(nAgents, execpath, _rank, settings)
{
}

bool Environment::predefinedNetwork(Network* const net) const
{
	//this function can be used if environment requires particular network settings
	//i.e. not fully connected LSTM/FF network
	//i.e. if you want to use convolutions
	return false;
}

void Environment::setDims() //this environment is for the cart pole test
{
    {
        sI.bounds.clear(); sI.top.clear(); sI.bottom.clear(); sI.isLabel.clear(); sI.inUse.clear();
        // State: coordinate...
        sI.bounds.push_back(12);
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.isLabel.push_back(false); sI.inUse.push_back(true);
        
        // ...velocity...
        sI.bounds.push_back(6);
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.isLabel.push_back(false); sI.inUse.push_back(true);
        
        // ...and angular velocity
        sI.bounds.push_back(6);
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.isLabel.push_back(false); sI.inUse.push_back(true);
        
        // ...angle...
        sI.bounds.push_back(16);
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.isLabel.push_back(false); sI.inUse.push_back(true);
    }
    {
        aI.realValues = false;
        aI.dim = 1;
        aI.zeroact = 2;
        aI.values.resize(aI.dim);
        
        for (int i=0; i<aI.dim; i++) {
            aI.bounds.push_back(7);
            aI.upperBounds.push_back( 1.);
            aI.lowerBounds.push_back(-1.);
            
            aI.values[i].push_back(-20.);
            aI.values[i].push_back(-5.);
            aI.values[i].push_back(-1.);
            aI.values[i].push_back(0.0);
            aI.values[i].push_back(1.);
            aI.values[i].push_back(5.);
            aI.values[i].push_back(20.);
        }
    }
    commonSetup();
}

bool Environment::pickReward(const State & t_sO, const Action & t_a, const State & t_sN, Real & reward)
{
    bool new_sample(false);
    if (reward<-0.9) new_sample=true;
#ifndef _scaleR_
    reward = 1. - fabs(t_sN.vals[3])/0.2;            //max cumulative reward = sum gamma^t r < 1/(1-gamma)
    if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
#else
    reward = (1. - fabs(t_sN.vals[3])/0.1)*(1.-gamma); //max cumulative reward = sum gamma^t r < 1/(1-gamma) = 1
    if (new_sample) reward = -1.;  // = - max cumulative reward
#endif
    return new_sample; //cart pole has failed if r = -1, need to clean this shit and rely only on info
}
