/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "GliderEnvironment.h"

GliderEnvironment::GliderEnvironment(const Uint _nAgents, const string _execpath, Settings & _s) :
Environment(_nAgents, _execpath, _s), sensevel(_s.senses==0) {
	//cheaperThanNetwork=false;
	settings.saveFreq = 1e5;
}

void GliderEnvironment::setDims() //this environment is for the cart pole test
{
	sI.inUse.clear();
	//if !sensevel then skip u v omega
	for (Uint i=0; i<7; i++) 
		sI.inUse.push_back(sensevel || i>2);

	sI.inUse.push_back(true);
	//sI.inUse.push_back(false);
	sI.inUse.push_back(false);
	sI.inUse.push_back(false);
	{
		sI.mean.push_back(0); //u
		sI.mean.push_back(0); //v
		sI.mean.push_back(0); //omega
		sI.mean.push_back(50); //x
		sI.mean.push_back(-25); //y
		sI.mean.push_back(0); //cos theta
		sI.mean.push_back(0); //sin theta
		sI.mean.push_back(0); //T
		sI.mean.push_back(0); //vx
		sI.mean.push_back(0); //vy
	}
	{
		sI.scale.push_back(1); //u
		sI.scale.push_back(1); //v
		sI.scale.push_back(1); //omega
		sI.scale.push_back(50); //x
		sI.scale.push_back(25); //y
		sI.scale.push_back(1); //cos theta
		sI.scale.push_back(1); //sin theta
		sI.scale.push_back(1); //T
		sI.scale.push_back(1); //vx
		sI.scale.push_back(1); //vy
	}
	aI.dim = 1; //number of action that agent can perform per turn: usually 1 (eg DQN)
	aI.values.resize(aI.dim);
	aI.values[0].push_back(-1.); //here the app accepts real numbers
	aI.values[0].push_back(1.);
	aI.bounded.push_back(1);
	commonSetup(); //required
}

bool GliderEnvironment::pickReward(const State & t_sO, const Action & t_a,
																 const State& t_sN, Real& reward,const int info)
{
    const bool new_sample = info==2;
    //if (new_sample) reward *= 1./(1.-gamma); // = - max cumulative reward
    return new_sample; //cart pole has failed if r = -1, need to clean this shit and rely only on info
}

vector<Real> GliderEnvironment::stateDumpUpperBound() {return vector<Real>{ 1, 1, .5,125,  0,2*M_PI};}
vector<Real> GliderEnvironment::stateDumpLowerBound() {return vector<Real>{-1,-1,-.5, -5,-50,     0};}
vector<Uint> GliderEnvironment::stateDumpNBins() {return vector<Uint> { 9, 9,  9, 53, 21,     9};}
