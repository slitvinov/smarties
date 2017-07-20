/*
 *  HingedFishEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Modded by SV on July 13, 2017
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "HingedFishEnvironment.h"

HingedFishEnvironment::HingedFishEnvironment(const Uint _nAgents, const string _execpath, Settings & _s) :
Environment(_nAgents, _execpath, _s),
sight(_s.senses==0 || _s.senses==2),
l_line(_s.senses==1 || _s.senses==2),
study(_s.rType), goalDY(_s.goalDY)
{
  cheaperThanNetwork=false;
    if (goalDY > 1.) goalDY = 1. - goalDY; //poor man's sign
}


void HingedFishEnvironment::setDims()
{
  /*
   vector<Real> state(6), info(8+3*NpLatLine);
   state[0] = dXrel;
   state[1] = dYrel;
   state[2] = relAngle;
   state[3] = relT;
   */
  sI.inUse.clear();

  {
    // State: Horizontal distance from goal point...
    sI.inUse.push_back(sight);

    // ...vertical distance...
    sI.inUse.push_back(sight);

    // ...inclination of1the fish...
    sI.inUse.push_back(sight);

    // ..time % Tperiod (phase of the motion
    sI.inUse.push_back(true);

  }

  {
    //Pout 4
    sI.inUse.push_back(false);

    //defPower 5
    sI.inUse.push_back(false);

    // EffPDef 6
    sI.inUse.push_back(false);

    // PoutBnd 7
    sI.inUse.push_back(false);

    // defPowerBnd 8
    sI.inUse.push_back(false);

    // EffPDefBnd 9
    sI.inUse.push_back(false);

    // Pthrust 10
    sI.inUse.push_back(false);

    // Pdrag 11
    sI.inUse.push_back(false);

    // ToD 12
    sI.inUse.push_back(false);
  }


  {
    aI.dim = 1; // Single action allowed
    aI.values.resize(aI.dim);
    //curavture
    aI.bounded.push_back(1);
    aI.values[0].push_back(-90.0); // Hinge angles can range from -90deg to +90deg
    // Negative values will allow the rudder to deflect in the same direction as done previously, if the fish so desires
    aI.values[0].push_back(90.0);
  }
  resetAll=false;
  commonSetup();
}

bool HingedFishEnvironment::pickReward(const State& t_sO, const Action& t_a,
                                    const State& t_sN, Real& reward,
                                    const int info)
{

  const Real effMax(1.0), effMin(0.0);
  bool new_sample;
  if (reward<-9.9) new_sample=true;

  if (study == 0){
    //Real scaled_effic = 2.*(t_sN.vals[10]-effMin)/(effMax-effMin) -1.;
    //reward = scaled_effic*(1.-gamma);
    reward = (t_sN.vals[6] - effMin)/(effMax-effMin);
  }
  else if (study == 1)
  {
    reward = 1. - fabs(t_sN.vals[1] - goalDY)/0.25;
    //reward*=(1.-gamma);
    if (new_sample) reward = -1.; // NEED TO THINK ABOOT THIS ONE!!
  }
  return new_sample;
}
