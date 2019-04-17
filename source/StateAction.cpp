//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Dmitry Alexeev.
//

#include "StateAction.h"
#include <math.h>

template<typename T>
static inline void sendRecvVectorFunc(
  const std::function<void(void*, size_t)>& sendRecvFunc, std::vector<T>& vec )
{
  Uint vecSize = vec.size();
  sendRecvFunc(&vecSize, 1 * sizeof(Uint) );
  if(vec.size() not_eq vecSize) vec.resize(vecSize);
  else assert( vecSize == (Uint) vec.size() );
  sendRecvFunc( vec.data(), vecSize * sizeof(T) );
}

void MDPdescriptor::synchronizeDescriptor (
  const std::function<void(void*, size_t)>& sendRecvFunc
)
{
  if(bFinalized)
    die("Tried to initialiaze MDPdescriptor multiple times. Would deadlock");
  bFinalized = true; // hopefully.

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // In this function we first recv all quantities of the descriptor
  // then we send them along. Idea is that simulation ranks will do nothing on
  // recv. Master ranks will first receive info from simulation, then pass it
  // along to worker-less masters.
  sendRecvFunc(&dimState, 1 * sizeof(Uint) );
  if(dimState == 0) warn("Stateless RL");

  sendRecvFunc(&dimAction,        1 * sizeof(Uint) );
  if(dimAction == 0)
    die("Application did not set up dimensionality of action vector.");

  sendRecvFunc(&bDiscreteActions, 1 * sizeof(bool) );

  sendRecvFunc(&nAppendedObs,            1 * sizeof(Uint) );
  sendRecvFunc(&isPartiallyObservable,   1 * sizeof(bool) );

  // by default agent can observe all components of state vector
  if(bStateVarObserved.size() == 0)
    bStateVarObserved = std::vector<bool> (dimState, true);
  sendRecvVectorFunc(sendRecvFunc, bStateVarObserved);
  if( bStateVarObserved.size() not_eq (size_t) dimState)
    die("Application error in setup of bStateVarObserved.");

  // by default state vector scaling is assumed to be with mean 0 and std 1
  if(stateMean.size() == 0) stateMean = std::vector<Real> (dimState, 0);
  sendRecvVectorFunc(sendRecvFunc, stateMean);
  if( stateMean.size() not_eq (size_t) dimState)
    die("Application error in setup of stateMean.");

  // by default agent can observer all components of action vector
  if(stateStdDev.size() == 0) stateStdDev = std::vector<Real> (dimState, 1);
  sendRecvVectorFunc(sendRecvFunc, stateStdDev);
  if( stateStdDev.size() not_eq (size_t) dimState)
    die("Application error in setup of stateStdDev.");

  for(Uint i=0; i<dimState; ++i) {
    if( stateStdDev[i] < std::numeric_limits<Real>::epsilon() )
      _die("Invalid value in scaling of state component %u.", i);
    stateScale[i] = 1/stateStdDev[i];
  }

  dimStateObserved = 0;
  for(Uint i=0; i<dimState; ++i) if(bStateVarObserved[i]) dimStateObserved++;
  if(world_rank == 0) {
   printf("SETUP: State vector has %d components, %d of which are observed. "
   "Action vector has %d components and it consists of %s actions.\n", dimState,
   dimStateObserved, dimAction, bDiscreteActions? "discrete" : "continuous");
  }

  // by default agent's action space is unbounded
  if(bActionSpaceBounded.size() == 0)
    bActionSpaceBounded = std::vector<bool> (dimAction, false);
  sendRecvVectorFunc(sendRecvFunc, bActionSpaceBounded);
  if( bActionSpaceBounded.size() not_eq (size_t) dimAction)
    die("Application error in setup of bActionSpaceBounded.");

  // by default agent's action space not scaled (ie up/low vals are -1 and 1)
  if(upperActionValue.size() == 0)
    upperActionValue = std::vector<Real> (dimAction,  1);
  sendRecvVectorFunc(sendRecvFunc, upperActionValue);
  if( upperActionValue.size() not_eq (size_t) dimAction)
    die("Application error in setup of upperActionValue.");

  // by default agent's action space not scaled (ie up/low vals are -1 and 1)
  if(lowerActionValue.size() == 0)
    lowerActionValue = std::vector<Real> (dimAction, -1);
  sendRecvVectorFunc(sendRecvFunc, lowerActionValue);
  if( lowerActionValue.size() not_eq (size_t) dimAction)
    die("Application error in setup of lowerActionValue.");

  if(bDiscreteActions == false)
  {
    if(world_rank==0) {
      printf("Action vector components :");
      for (Uint i=0; i<dimAction; i++) {
        printf(" [ %u : %s to (%.1f:%.1f) ]", i,
        bActionSpaceBounded[i] ? "bound" : "scaled",
        upperActionValue[i], lowerActionValue[i]);
      }
      printf("\n");
    }
    return; // skip setup of discrete-action stuff
  }

  // Now some logic. The discreteActionValues vector should have size dimAction
  // If action space is continuous, these values are not used. If action space
  // is discrete at the end of synchronization we make sure that each component
  // has size greater than one. Otherwise agent has no options to choose from.
  if(discreteActionValues.size() == 0)
    discreteActionValues = std::vector<Uint> (dimAction, 0);
  sendRecvVectorFunc(sendRecvFunc, discreteActionValues);
  if( discreteActionValues.size() not_eq (size_t) dimAction)
    die("Application error in setup of discreteActionValues.");

  if(world_rank==0) printf("Discrete-action vector options :");
  for(size_t i=0; i<dimAction; ++i) {
    if( discreteActionValues[i] < 2 )
      die("Application error in setup of discreteActionValues: "
          "found less than 2 options to choose from.");
    if(world_rank==0)
      printf(" [ %u : %u options ]", i, discreteActionValues[i]);
  }
  if(world_rank==0) printf("\n")

  discreteActionShifts = std::vector<Uint>(dimAction);
  discreteActionShifts[0] = 1;
  for (Uint i=1; i < dimAction; ++i)
    discreteActionShifts[i] = discreteActionShifts[i-1] *
                              discreteActionValues[i-1];

  maxActionLabel = discreteActionShifts[dimAction-1] *
                   discreteActionValues[dimAction-1];
}

  /*
void StateInfo::copy_observed(Rvec& res, const Uint append=0) const {
  //copy state into res, append is used to chain multiple states together
  Uint k = append*sInfo.dimUsed;
  assert(res.size() >= k+sInfo.dimUsed);
  for (Uint i = 0; i < MDP.dimState; i++)
    if (MDP.bStateVarObserved[i]) res[k++] = vals[i];
}

std::vector<memReal> StateInfo::inUseStd() const {
  std::vector<memReal> ret(dimUsed, 0);
  for(Uint i=0, k=0; i<dim && scale.size(); i++) {
    if(inUse[i]) ret[k++] = scale[i];
    if(i+1 == dim) assert(k == dimUsed);
  }
  return ret;
}
std::vector<memReal> StateInfo::inUseMean() const {
  std::vector<memReal> ret(dimUsed, 1);
  for(Uint i=0, k=0; i<dim && mean.size(); i++) {
    if(inUse[i]) ret[k++] = mean[i];
    if(i+1 == dim) assert(k == dimUsed);
  }
  return ret;
}
std::vector<memReal> StateInfo::inUseInvStd() const {
  std::vector<memReal> ret(dimUsed, 1);
  for(Uint i=0, k=0; i<dim && scale.size(); i++) {
    if(inUse[i]) ret[k++] = 1/scale[i];
    if(i+1 == dim) assert(k == dimUsed);
  }
  return ret;
}

///////////////////////////////////////////////////////////////////////////////
//CONTINUOUS ACTION STUFF
Real ActionInfo::getActMaxVal(const Uint i) const
{
  assert(i<dim && dim==values.size());
  assert(values[i].size()>1); //otherwise scaling is impossible
  return * std::max_element(std::begin(values[i]), std::end(values[i]));
}

Real ActionInfo::getActMinVal(const Uint i) const
{
  assert(i<dim && dim==values.size());
  assert(values[i].size()>1); //otherwise scaling is impossible
  return * std::min_element(std::begin(values[i]), std::end(values[i]));
}

Real ActionInfo::getScaled(const Real unscaled, const Uint i) const
{
  //unscaled value and i is to which component of action vector it corresponds
  //if action space is bounded, return the scaled component, else return unscaled
  //scaling is between max and min of values vector (user specified in environment)
  //scaling function is x/(1+abs(x)) (between -1 and 1 for x in -inf, inf)
  const Real min_a = getActMinVal(i), max_a = getActMaxVal(i);
  assert(max_a-min_a > std::numeric_limits<Real>::epsilon());
  if (bounded[i]) {
    const Real soft_sign = _tanh(unscaled);
    //const Real soft_sign = unscaled/(1. + std::fabs(unscaled));
    return       min_a + 0.5*(max_a - min_a)*(soft_sign + 1);
  } else  return min_a + 0.5*(max_a - min_a)*(unscaled  + 1);
}

///////////////////////////////////////////////////////////////////////////////
//DISCRETE ACTION STUFF
*/
