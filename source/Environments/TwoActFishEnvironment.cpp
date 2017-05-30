/*
 *  TwoFishEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "TwoActFishEnvironment.h"
//#define __Cubism3D

TwoActFishEnvironment::TwoActFishEnvironment(const Uint _nAgents, const string _execpath, Settings & _s) :
Environment(_nAgents, _execpath, _s),
sight( _s.senses    ==0 ||  _s.senses==8),
rcast( _s.senses    % 2), //if eq {1,  3,  5,  7}
lline((_s.senses/2) % 2), //if eq {  2,3,    6,7}
press((_s.senses/4) % 2 ||  _s.senses==8), //if eq {      4,5,6,7}
study(_s.rType), goalDY((_s.goalDY>1.)? 1.-_s.goalDY : _s.goalDY)
{
  cheaperThanNetwork = false; //this environment is more expensive to simulate than updating net. todo: think it over?
#ifdef __Cubism3D
	//mpi_ranks_per_env = 1;
	mpi_ranks_per_env = 8;
#else
	mpi_ranks_per_env = 0;
#endif
	//paramsfile="settings_32.txt";
  paramsfile="settings_64.txt";

  assert(settings.senses<=8);
}

void TwoActFishEnvironment::setDims()
{
	sI.inUse.clear();
	{
		// State: Horizontal distance from goal point...
		sI.inUse.push_back(sight);
		// ...vertical distance...
		sI.inUse.push_back(sight);
		// ...inclination of1the fish...
		sI.inUse.push_back(sight);
		// ..time % Tperiod (phase of the motion, maybe also some info on what is the incoming vortex?)...
		sI.inUse.push_back(true);
		// ...last action (HAX!)
		sI.inUse.push_back(true);
		// ...second last action (HAX!)
		sI.inUse.push_back(false); //if l_line i have curvature info
	}
	{
		//New T period
		sI.inUse.push_back(true);

		//Phase Shift
		sI.inUse.push_back(true);

		// VxInst
		sI.inUse.push_back(false);
		//sI.inUse.push_back(true);

		// VyInst
		sI.inUse.push_back(false);

		// AvInst
		sI.inUse.push_back(false);
	}
	#if 0
		//Xabs 6
		sI.inUse.push_back(false);

		//Yabs 7
		sI.inUse.push_back(false);
	#endif
	{
		//Dist 6
		sI.inUse.push_back(false);

		//Quad 7
		sI.inUse.push_back(false);

		// VxAvg 8
		sI.inUse.push_back(false);

		// VyAvg 9
		sI.inUse.push_back(false);

		// AvAvg 10
		sI.inUse.push_back(false);

		//Pout 11
		sI.inUse.push_back(false);

		//defPower 12
		sI.inUse.push_back(false);

		// EffPDef 13
		sI.inUse.push_back(false);

		// PoutBnd 14
		sI.inUse.push_back(false);

		// defPowerBnd 15
		sI.inUse.push_back(false);

		// EffPDefBnd 16
		sI.inUse.push_back(false);

		// Pthrust 17
		sI.inUse.push_back(false);

		// Pdrag 18
		sI.inUse.push_back(false);

		// ToD 19
		sI.inUse.push_back(false);
	}
	const Uint nSensors = 10;
	for (Uint i=0; i<nSensors; i++) {
		// (VelNAbove  ) x 5 [20]
		sI.inUse.push_back(lline && i<4);
	}
	for (Uint i=0; i<nSensors; i++) {
		// (VelTAbove  ) x 5 [25]
		sI.inUse.push_back(lline && i<4);
	}
	for (Uint i=0; i<nSensors; i++) {
		// (VelNBelow  ) x 5 [30]
		sI.inUse.push_back(lline && i<4);
	}
	for (Uint i=0; i<nSensors; i++) {
		// (VelTBelow  ) x 5 [35]
		sI.inUse.push_back(lline && i<4);
	}
	for (Uint i=0; i<nSensors; i++) {
		// (FPAbove  ) x 5 [40]
		sI.inUse.push_back(press && i<4);
	}
	for (Uint i=0; i<nSensors; i++) {
		//	 (FVAbove  ) x 5 [45]
		sI.inUse.push_back(press && i<4);
	}
	for (Uint i=0; i<nSensors; i++) {
		// (FPBelow  ) x 5 [50]
		sI.inUse.push_back(press && i<4);
	}
	for (Uint i=0; i<nSensors; i++) {
		// (FVBelow ) x 5 [55]
		sI.inUse.push_back(press && i<4);
	}
	for (Uint i=0; i<2*nSensors; i++) {
		// (FVBelow ) x 5 [55]
		sI.inUse.push_back(rcast);
	}
	/*
	sI.values.push_back(-.50);
	sI.values.push_back(-.25);
	sI.values.push_back(0.00);
	sI.values.push_back(0.25);
	sI.values.push_back(0.50);
	*/
	{
		aI.dim = 2;
		aI.values.resize(aI.dim);
		//curavture
		aI.bounded.push_back(1);
		aI.values[0].push_back(-.75);
		aI.values[0].push_back(0.75);
		//period:
		aI.bounded.push_back(1);
		aI.values[1].push_back(-.5);
		aI.values[1].push_back(0.5);
	}
	resetAll=false;
	commonSetup();
}


bool TwoActFishEnvironment::pickReward(const State& t_sO, const Action& t_a,
                                const State& t_sN, Real& reward, const int info)
{
  if(info!=1)
  if (fabs(t_sO.vals[4] -t_sN.vals[5])>0.00001) {
      printf("Mismatch state two states!!! [%s] === [%s]\n",
       t_sN._print().c_str(),t_a._print().c_str());
      abort();
  }

  bool new_sample(false);
  if (reward<-9.9) new_sample=true;
  if(new_sample) assert(info==2);

  if (study == 0) {
      #ifdef __Cubism3D
        reward = (t_sN.vals[18]-.3)/(.8-.6);
        if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
      #else
        reward = (t_sN.vals[18]-.3)/(1.-.3);
        if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
      #endif
  }
  else if (study == 1) {
      #ifdef __Cubism3D
      	reward = (t_sN.vals[21]-.3)/(.6-.3);
      #else
      	reward = (t_sN.vals[21]-.3)/(.6-.3);
      #endif

      if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
  }
  else if (study == 2) {
      reward =  1.-2*sqrt(fabs(t_sN.vals[1])); //-goalDY
      if (new_sample) reward = -2./(1.-gamma);
  }
  else if (study == 4) {
    const Real x=t_sN.vals[0], y=t_sN.vals[1];
    //Fish should stay 1.5 body lengths behind leader at y=0
    reward = 1 - std::sqrt((x-1.5)*(x-1.5) + y*y);
    if (new_sample) reward = -1./(1.-gamma);
  }
  else if (study == 5) {
    reward = (t_sN.vals[18]-.4)/.5;
    if (t_sN.vals[0] > 0.5) reward = std::min(0.,reward);
    if (new_sample) reward = -2./(1.-gamma);
  }
  else if (new_sample) reward = -10.;

  //gently push sim away from extreme curvature: not kosher
  if(std::fabs(t_a.vals[0])>0.74)
    reward = std::min((Real)0.,reward);
  if(std::fabs(t_a.vals[1])>0.49)
    reward = std::min((Real)0.,reward);

  return new_sample;
}

#ifdef __DBG_CNN
bool TwoActFishEnvironment::predefinedNetwork(Builder* const net) const
{
  if(!sight || !press || rcast || lline || sI.dimUsed != 90 )
    die("Pick correct state\n");

  {
	  const int inputsize[3] = {9,10,1};
	  net->add2DInput(inputsize);
  }
  {
    const int filterSize[3] = {3,3,3};
    const int padding[2] = {1,1};
    const int outSize[3] = {9,10,3};
    const int stride[2] = {1,1};
    net->addConv2DLayer(filterSize, outSize, padding, stride);
  }
  {
    const int filterSize[3] = {3,2,3};
    const int padding[2] = {0,0};
    const int outSize[3] = {3,5,3};
    const int stride[2] = {3,2};
    net->addConv2DLayer(filterSize, outSize, padding, stride);
  }
  {
    const int filterSize[3] = {3,5,32};
    const int padding[2] = {0,0};
    const int outSize[3] = {1,1,32};
    const int stride[2] = {0,0};
    net->addConv2DLayer(filterSize, outSize, padding, stride);
  }
  net->addOutput(1+3+2, "Normal");

  return true;
}
#endif
