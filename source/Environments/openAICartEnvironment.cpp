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
#ifndef GYM_RENDEROPT
#define GYM_RENDEROPT 1
#endif
using namespace std;

openAICartEnvironment::openAICartEnvironment(const int _nAgents, const string _execpath, Settings & _s) :
Environment(_nAgents, _execpath, _s), allSenses(_s.senses==0)
{
//   cheaperThanNetwork=false;
}

void openAICartEnvironment::setDims() //this environment is for the cart pole test
{
  comm_ptr->getStateActionShape(aI.values, sI.mean, sI.scale, aI.bounded);
  vector<Real> upper = sI.mean;
  vector<Real> lower = sI.scale;
	bool scaleState = true;
	if(!settings.world_rank) printf("State dim:");
  for (unsigned i=0; i<upper.size(); i++) {
    sI.mean[i]  = 0.5*(upper[i]+lower[i]);
    sI.scale[i] = 0.5*(upper[i]-lower[i])/std::sqrt(3.); //approximate std=1
    assert(sI.scale[i]>0);
		if(sI.scale[i]>=1e3) scaleState = false;
		if(!settings.world_rank) printf(" [%u: %f +/- %f]", i, sI.mean[i], sI.scale[i]);
  }
	if(!settings.world_rank) printf("\n");
	if(!scaleState) { //if empty then mean and scale computed from data
		sI.scale = vector<Real>();
		sI.mean = vector<Real>();
	}

	sI.inUse.resize(sI.mean.size(), 1);
  aI.dim = aI.values.size();
	if(!settings.world_rank) printf("Action dim:");
	for (Uint i=0; i<aI.dim; i++) {
		const Real amax = aI.getActMaxVal(i), amin = aI.getActMinVal(i);
		const Real scale = 0.5*(amax - amin), mean = 0.5*(amax + amin);
		if(scale>=1e3) aI.bounded[i] = 0;
		if(!settings.world_rank)
		printf(" [%u: %f +/- %f%s]", i, mean, scale, aI.bounded[i]?" (bounded)":"");
	}
	if(!settings.world_rank) printf("\n");
  commonSetup(); //required

	if(settings.slaves_rank==0) return;

	#if   GYM_RENDEROPT==0
		double bRender[1] = {settings.bTrain||settings.slaves_rank>1 ? -1.:1.};
		comm_ptr->send_buffer_to_app(bRender, sizeof(double));
	#elif GYM_RENDEROPT==1
		double bRender[1] = {settings.slaves_rank>1 ? -1. : 1.};
		comm_ptr->send_buffer_to_app(bRender, sizeof(double));
	#elif GYM_RENDEROPT==2
		double bRender[1] = {settings.slaves_rank>1 ? -1. : 2.};
		comm_ptr->send_buffer_to_app(bRender, sizeof(double));
	#else
		double bRender[1] = {1.};
		comm_ptr->send_buffer_to_app(bRender, sizeof(double));
	#endif
}