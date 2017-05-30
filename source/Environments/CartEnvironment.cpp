/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "CartEnvironment.h"

CartEnvironment::CartEnvironment(const Uint _nAgents, const string _execpath, Settings & _s) :
Environment(_nAgents, _execpath, _s), allSenses(_s.senses==0)
{
//   cheaperThanNetwork=false;
}

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
	sI.inUse.push_back(true); //ignore, leave as is

	if(1)
	{
			sI.mean.push_back(0);    //x
			sI.mean.push_back(0);    //v
			sI.mean.push_back(0);    //omega
			sI.mean.push_back(0);    //theta

			sI.scale.push_back(.5);  //x
			sI.scale.push_back(1);   //v
			sI.scale.push_back(1);   //omega
			sI.scale.push_back(0.2); //theta
	}

	/*
	* also valid:
	*
	* for (int i=0; i<some_number_of_vars; i++)
	* {
	* 		sI.top.push_back(MAXVAL); sI.bottom.push_back(MINVAL);
	* 		sI.isLabel.push_back(false); sI.inUse.push_back(true); sI.bounds.push_back(1); //ignore, leave as is
	* }
	*/
	aI.dim = 1; //number of action that agent can perform per turn: usually 1 (eg DQN)
	aI.values.resize(aI.dim);
	for (Uint i=0; i<aI.dim; i++)
	{
			//this framework sends a real number to the application
			//if you want to receive an integer number between 0 and nOptions (eg action option)
			//just write aI.values[i].push_back(0.1); ... aI.values[i].push_back((nOptions-1) + 0.1);
			//i added the 0.1 is just to be extra safe when converting a float to an integer

			aI.values[i].push_back(-10.); //here the app accepts real numbers
			aI.values[i].push_back(-5.);
			aI.values[i].push_back(-1.);
			aI.values[i].push_back(0.0);
			aI.values[i].push_back(1.0);
			aI.values[i].push_back(5.0);
			aI.values[i].push_back(10.);
			//the number of components must be ==nOptions
	}
	commonSetup(); //required
}

bool CartEnvironment::pickReward(const State & t_sO, const Action & t_a,
																 const State& t_sN, Real& reward,const int info)
{
  bool new_sample(false);

  //Compute the reward. If you do not do anything, reward will be whatever was set already to reward.
  //this means that reward will be one sent by the app

  if (reward<-0.9) new_sample=true; //in cart pole example, if reward from the app is -1 then I failed

  //here i can change the reward: instead of -1 or 0, i can give a positive reward if angle is small
  reward = 1 - fabs(t_sN.vals[3])/0.2 - fabs(t_sN.vals[0])/2.4;    //max cumulative reward = sum gamma^t r < 1/(1-gamma)
  if (new_sample)
		reward = -1./(1.-gamma); // = - max cumulative reward
  //was is the last state of the sequence?

  //this must be set: was it the last episode? you can get it from reward?
  return new_sample; //cart pole has failed if r = -1, need to clean this shit and rely only on info
}
