/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "CartEnvironment.h"

CartEnvironment::CartEnvironment(const Uint _nAgents, const string _execpath,
  Settings & _s) : Environment(_nAgents,_execpath,_s), allSenses(_s.senses<2),
  swingup(_s.senses%2==0), modRew(_s.rType)
{
//   cheaperThanNetwork=false;
}
/*
bool CartEnvironment::predefinedNetwork(Builder* const net) const
{
  //this function can be used if environment requires particular network settings
  //i.e. not fully connected LSTM/FF network
  //i.e. if you want to use convolutions
  return false;
}

void CartEnvironment::setDims() //this environment is for the cart pole test
{
  sI.inUse.clear();
  //for each state variable:
  // State: coordinate...
  sI.inUse.push_back(true); //ignore, leave as is
  // ...velocity...
  sI.inUse.push_back(allSenses); //ignore, leave as is
  // ...and angular velocity
  sI.inUse.push_back(allSenses); //ignore, leave as is
  // ...angle...
  sI.inUse.push_back(!swingup); //ignore, leave as is
  // ...angle...
  sI.inUse.push_back(swingup); //ignore, leave as is
    // ...angle...
  sI.inUse.push_back(swingup); //ignore, leave as is

  if(1)
  {
      sI.mean.push_back(0);    //x
      sI.mean.push_back(0);    //v
      sI.mean.push_back(0);    //omega
      sI.mean.push_back(0);    //theta
      sI.mean.push_back(0);    //theta
      sI.mean.push_back(0);    //theta

      sI.scale.push_back(.5);  //x
      sI.scale.push_back(1);   //v
      sI.scale.push_back(1);   //omega
      sI.scale.push_back(0.2); //theta
      sI.scale.push_back(1);    //theta
      sI.scale.push_back(1);    //theta
  }

  aI.dim = 1; //number of action that agent can perform per turn: usually 1 (eg DQN)
  aI.values.resize(aI.dim);
  for (Uint i=0; i<aI.dim; i++)
  {
      //this framework sends a real number to the application
      //if you want to receive an integer number between 0 and nOptions (eg action option)
      //just write aI.values[i].push_back(0.1); ... aI.values[i].push_back((nOptions-1) + 0.1);
      //i added the 0.1 is just to be extra safe when converting a float to an integer

      //aI.values[i].push_back(-10.); //here the app accepts real numbers
      aI.values[i].push_back(-10.);
      aI.values[i].push_back( -2.);
      aI.values[i].push_back( 0.0);
      aI.values[i].push_back( 2.0);
      aI.values[i].push_back(10.0);
      //aI.values[i].push_back(10.);
      //the number of components must be ==nOptions
  }
  commonSetup(); //required
}
*/
/*
bool CartEnvironment::pickReward(const State & t_sO, const Action & t_a,
                                 const State& t_sN, Real& reward,const int info)
{
  const bool new_sample = info==2;

  //Compute the reward. If you do not do anything, reward will be whatever was set already to reward.
  //this means that reward will be one sent by the app

  if(modRew && !swingup)
  {
    //here i can change the reward: instead of -1 or 0, i can give a positive reward if angle is small
    reward = 1 - fabs(t_sN.vals[3])/0.2 - fabs(t_sN.vals[0])/2.4;    //max cumulative reward = sum gamma^t r < 1/(1-gamma)
    if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
  }
  return new_sample;
}*/

Uint CartEnvironment::getNdumpPoints()
{
       if( swingup &&  allSenses) return 49 * 15 * 15 * 32;
  else if( swingup && !allSenses) return 49 * 32;
  else if(!swingup &&  allSenses) return 49 * 15 * 15 * 32;
  else if(!swingup && !allSenses) return 49 * 32;
  else {
    die("CartEnvironment::getNdumpPoints()\n");
    return 0;
  }
}

vector<Real> CartEnvironment::getDumpState(Uint k)
{
  if(!allSenses) {
  die("GliderEnvironment::getDumpState\n");
  }

        const vector<Real> ub = {  2.4,  .5,  .5, 2*M_PI};
        const vector<Real> lb = { -2.4, -.5, -.5,  0};
        const vector<Uint> nb = {   49,  15,  15, 32};
        vector<Real> state(4,0);
        for (Uint i=0; i<4; i++) {
                const Uint j = k % nb[i];
                state[i] = lb[i] + (ub[i]-lb[i]) * (j/(Real)(nb[i]-1));
                k /= nb[i];
        }
  if(swingup) {
    state.resize(5);
          const Real cosang = std::cos(state[3]);
          const Real sinang = std::sin(state[3]);
          state[3] = cosang; state[4] = sinang;
  }
        return state;
}
