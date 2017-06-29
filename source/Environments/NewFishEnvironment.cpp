/*
 *  TwoFishEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "NewFishEnvironment.h"

NewFishEnvironment::NewFishEnvironment(const Uint _nAgents, const string _execpath, Settings & _s) :
Environment(_nAgents, _execpath, _s),
sight( _s.senses    ==0),
rcast( _s.senses    % 2), //if eq {1,  3,  5,  7}
lline((_s.senses/2) % 2), //if eq {  2,3,    6,7}
press((_s.senses/4) % 2), //if eq {      4,5,6,7}
study(_s.rType), goalDY((_s.goalDY>1.)? 1.-_s.goalDY : _s.goalDY)
{
  cheaperThanNetwork = false; //this environment is more expensive to simulate than updating net. todo: think it over?
  assert(_s.senses<8);
}

void NewFishEnvironment::setDims()
{
  sI.inUse.clear();
  {
    // State: Horizontal distance from goal point...
    //one block in between the bounds, one more on each side
    sI.inUse.push_back(sight);
    // ...vertical distance...
    sI.inUse.push_back(sight);
    // ...inclination of1the fish...
    // only positive or negative
    sI.inUse.push_back(sight);
    // ..time % Tperiod (phase of the motion, maybe also some info on what is the incoming vortex?)...
    // Will get ~ 0 or 0.5
		//sI.inUse.push_back(false);
    sI.inUse.push_back(sight);

    // ...last action (HAX!)
    //sI.inUse.push_back(false);
    sI.inUse.push_back(sight);

    // ...second last action (HAX!)
		//sI.inUse.push_back(false); //if l_line i have curvature info
    sI.inUse.push_back(sight); //if l_line i have curvature info
  }
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

  const Uint nSensors = 20;
  for (Uint i=0; i<nSensors; i++) {
      // (VelNAbove  ) x 5 [20]
      sI.inUse.push_back(lline);
  }
  for (Uint i=0; i<nSensors; i++) {
      // (VelTAbove  ) x 5 [25]
      sI.inUse.push_back(lline);
  }
  for (Uint i=0; i<nSensors; i++) {
      // (VelNBelow  ) x 5 [30]
      sI.inUse.push_back(lline);
  }
  for (Uint i=0; i<nSensors; i++) {
      // (VelTBelow  ) x 5 [35]
      sI.inUse.push_back(lline);
  }
  for (Uint i=0; i<nSensors; i++) {
      // (FPAbove  ) x 5 [40]
      sI.inUse.push_back(press);
  }
  for (Uint i=0; i<nSensors; i++) {
      // (FVAbove  ) x 5 [45]
      sI.inUse.push_back(press);
  }
  for (Uint i=0; i<nSensors; i++) {
      // (FPBelow  ) x 5 [50]
      sI.inUse.push_back(press);
  }
  for (Uint i=0; i<nSensors; i++) {
      // (FVBelow ) x 5 [55]
      sI.inUse.push_back(press);
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
    aI.dim = 1;
    aI.values.resize(aI.dim);
    aI.bounded.push_back(1);
    for (Uint i=0; i<aI.dim; i++) {
      aI.values[i].push_back(-.75);
      aI.values[i].push_back(-.50);
      aI.values[i].push_back(-.25);
      aI.values[i].push_back(0.00);
      aI.values[i].push_back(0.25);
      aI.values[i].push_back(0.50);
      aI.values[i].push_back(0.75);
    }
  }
  resetAll=true;
  commonSetup();
}

bool NewFishEnvironment::pickReward(const State& t_sO, const Action& t_a,
                                const State& t_sN, Real& reward, const int info)
{
  if (fabs(t_sN.vals[5] -t_sO.vals[4])>0.001) {
      printf("Mismatch new and old state!!! \n [%s] \n [%s] \n",
              t_sO._print().c_str(),t_sN._print().c_str());
      abort();
  }
  if (fabs(t_sN.vals[4] -t_a.vals[0])>0.001) {
      printf("Mismatch state and action!!! \n [%s] \n [%s] \n",
              t_a._print().c_str(),t_sN._print().c_str());
      abort();
  }


  if ( fabs(t_sN.vals[3] -t_sO.vals[3])<1e-2 && reward>0 ) {
      printf("Same time for two states!!! \n [%s] \n [%s] \n",
              t_sO._print().c_str(),t_sN._print().c_str());
      abort();
  }
  /*
  if (t_sN.vals[13]>1 || t_sN.vals[13]<0) {
      printf("You modified the efficiency\n");
      abort();
  }
  if (t_a.vals[0]<0 || t_a.vals[0]>4) {
      printf("Actions out of bounds\n");
      abort();
  }
  if (fabs(t_a.valsContinuous[0]-aI.values[0][round(t_a.vals[0])])>1e-2) {
      printf("You modified the actions\n");
      abort();
  }*/

  bool new_sample(false);
  if (reward<-9.9) new_sample=true;

  if (study == 0) {
      //const Real scaledEfficiency = t_sN.vals[13];
      const Real scaledEfficiency = (t_sN.vals[13]-.4)/(1.-.4); //between 0 and 1
      reward = scaledEfficiency; //max cumulative reward = sum gamma^t r < 1/(1-gamma)
      if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
  }
  else if (study == 1) {
      const Real scaledBndEfficiency = (t_sN.vals[16]-.3)/(.6-.3); //between 0 and 1
      reward = scaledBndEfficiency;
      if (new_sample) reward = -1./(1.-gamma);
  }
  else if (study == 2) {
      //const Real scaledRew = 1. -2.*fabs(t_sN.vals[1]-goalDY);
      const Real scaledRew = 1. -2.*sqrt(fabs(t_sN.vals[1]));
      reward =  scaledRew;
      new_sample = false;
      if (new_sample) reward = -1./(1.-gamma);
  }
  else if (study == 3) {
  	const Real DX_penal = 8*fabs(t_sN.vals[0]-goalDY);  //goalDY actually goalDX
  	const Real DY_penal = 2*fabs(t_sN.vals[1]);
      const Real scaledRew = 1. - min(DX_penal+DY_penal, 2.);
      reward = scaledRew;            //max cumulative reward = sum gamma^t r < 1/(1-gamma)
      if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
  }
  else if (study == 4) {
      const Real scaledEfficiency = 0.;
      reward = scaledEfficiency;            //max cumulative reward = sum gamma^t r < 1/(1-gamma)
      if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
  }
  else if (study == 5) {
      reward = (t_sN.vals[16]-.3)/(.6-.3) -8*std::pow(t_sN.vals[1],4);
      if (t_sN.vals[0] > 0.75) reward = 0;
      if (new_sample) reward = -1./(1.-gamma);
  }
  else {
      die("Wrong reward\n");
  }

  if (fabs(t_sN.vals[0])>0.9999)
  reward = std::min((Real)-1/(1.-gamma),reward);

  if (fabs(t_sN.vals[1])>0.9999)
  reward = std::min((Real)-1/(1.-gamma),reward);

  if (fabs(t_sN.vals[2])> M_PI - 0.0001)
  reward = std::min((Real)-1/(1.-gamma),reward);
  //    if(std::fabs(t_a.vals[0])>0.7)
  //
  return new_sample;
}
