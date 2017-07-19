/*
 *  TwoFishEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "TwoFishEnvironment.h"

TwoFishEnvironment::TwoFishEnvironment(const Uint _nAgents, const string _execpath, Settings & _s) :
Environment(_nAgents, _execpath, _s),
sight(_s.senses==0 || _s.senses==2),
l_line(_s.senses==1 || _s.senses==2),
study(_s.rType), goalDY(_s.goalDY)
{
	cheaperThanNetwork=false;
    if (goalDY > 1.) goalDY = 1. - goalDY; //poor man's sign
}


void TwoFishEnvironment::setDims()
{
  /*
   vector<Real> state(6), info(8+3*NpLatLine);
   state[0] = DX;
   state[1] = dYrel;
   state[2] = ang;
   state[3] = relT;
   state[4] = new_a[ID];
   state[5] = old_a[ID];
   info[0]  = _D[i]->ToD;
   info[1]  = _D[i]->Pout;
   info[2]  = _D[i]->defPower;
   info[3]  = eta;
   info[4]  = _D[i]->EffPDef;
   info[5]  = _D[i]->Vx;
   info[6]  = _D[i]->Vy;
   info[7]  = _D[i]->angVel;
   */
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
    sI.inUse.push_back(true); //if l_line i have curvature info
  }
  {
    //T / D 6
    sI.inUse.push_back(false);

    // Pout 7
    sI.inUse.push_back(false);

    // defPower 8
    sI.inUse.push_back(false);

    // eta = kin/kin+Pout 9
    sI.inUse.push_back(false);

    // etaPdef 10
    sI.inUse.push_back(false);

    // Vx 11
    sI.inUse.push_back(false);

    // Vy 12
    sI.inUse.push_back(false);

    // angvel 13
    sI.inUse.push_back(false);
  }
  for (Uint i=0; i<10; i++) { // >=14
      // (p_above  ) x 10
      sI.inUse.push_back(false);
  }

  for (Uint i=0; i<10; i++) // >=24
  {
      // ( p_below ) x 10
      sI.inUse.push_back(false);
  }

  for (Uint i=0; i<10; i++) // >=34
  {
      // ( curvature) x 10
      sI.inUse.push_back(false);
  }

  //sI.values.push_back(-2.);
  //sI.values.push_back(-1.);
  //sI.values.push_back(0.0);
  //sI.values.push_back(1.0);
  //sI.values.push_back(2.0);

  aI.dim = 1;
  aI.values.resize(aI.dim);
  for (Uint i=0; i<aI.dim; i++) {
      aI.values[i].push_back(-.5);
      aI.values[i].push_back(-.25);
      aI.values[i].push_back(0.0);
      aI.values[i].push_back(.25);
      aI.values[i].push_back(0.5);
  }
  commonSetup();
}

bool TwoFishEnvironment::pickReward(const State& t_sO, const Action& t_a,
                                    const State& t_sN, Real& reward,
                                    const int info)
{

   if (fabs(t_sN.vals[4] - t_a.vals[0])>0.001)
   _die("ASSUMING 2F: mismatch between state and reported action!!! [%s]",t_sN._print().c_str());

   if (fabs(t_sN.vals[5] - t_sO.vals[4])>0.001)
   _die("ASSUMING 2F: mismatch between new state and old state!!! [%s]", t_sN._print().c_str());

   if ( fabs(t_sN.vals[3] - t_sO.vals[3])<1e-2 )
   _die("ASSUMING 2F: same time for two states!!! [%s]", t_sN._print().c_str());

  Real ToDmax(2.55764), PoutMax(5.85923e-07), dePowerMax(0.273871);
  Real etaMax(1), effMax(1), etaMin(0.253908), effMin(0.428293);
  Real ToDmin(0.462123), PoutMin(-2.23674e-06), dePowerMin(0);
  bool new_sample;
  if (reward<-9.9) new_sample=true;

  if (study == 0)
  {
    Real scaled_effic = 2.*(t_sN.vals[10]-effMin)/(effMax-effMin) -1.;
    reward = scaled_effic*(1.-gamma);
  }
  else if (study == 1)
  {
    Real scaled_effic = 2.*(t_sN.vals[10]-effMin)/(effMax-effMin) -1.;
    reward = scaled_effic*(1.-gamma);
    if (new_sample) reward = -1.;
  }
  else if (study == 2)
  {
    reward = 1. - fabs(t_sN.vals[1] - goalDY)/0.25;
    reward*=(1.-gamma);
    if (new_sample) reward = -1.;
  }
  else if (study == 3)
  {
    Real scaled_effic = 2.*(t_sN.vals[6]-ToDmin)/(ToDmax-ToDmin) -1.;
    reward = scaled_effic*(1.-gamma);
    if (new_sample) reward = -1.;
  }
  else if (study == 4)
  {
    Real scaled_effic = 2.*(t_sN.vals[7]-PoutMin)/(PoutMax-PoutMin) -1.;
    reward = scaled_effic*(1.-gamma);
    if (new_sample) reward = -1.;
  }
  else if (study == 5)
  {
    Real scaled_effic = 2*(t_sN.vals[8]-dePowerMin)/(dePowerMax-dePowerMin)-1;
    reward = scaled_effic*(1.-gamma);
    if (new_sample) reward = -1.;
  }
  else if (study == 6)
  {
    Real scaled_effic = 2.*(t_sN.vals[9]-etaMin)/(etaMax-etaMin) -1.;
    reward = scaled_effic*(1.-gamma);
    if (new_sample) reward = -1.;
  }
  else if (study == 7)
  {
    Real scaled_effic = 2.*(t_sN.vals[10]-effMin)/(effMax-effMin) -1.;
    reward = - scaled_effic*(1.-gamma);
    if (new_sample) reward = -1.;
  }
  else if (study == 8)
  {
    Real scaled_effic = 2.*(t_sN.vals[7]-PoutMin)/(PoutMax-PoutMin) -1.;
    reward = - scaled_effic*(1.-gamma);
    if (new_sample) reward = -1.;
  }
  else if (study == 9)
  {
    Real scaled_effic = 2*(t_sN.vals[8]-dePowerMin)/(dePowerMax-dePowerMin)-1;
    reward = - scaled_effic*(1.-gamma);
    if (new_sample) reward = -1.;
  }
  else if (study == 10)
  {
    Real scaled_effic = 2.*(t_sN.vals[6]-ToDmin)/(ToDmax-ToDmin) -1.;
    reward = - scaled_effic*(1.-gamma);
    if (new_sample) reward = -1.;
  }
  else
  {
      die("Wrong reward\n");
  }

  return new_sample;
}
