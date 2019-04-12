//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Dmitry Alexeev.
//

#include "StateAction.h"
#include <algorithm>
#include <math.h>




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
