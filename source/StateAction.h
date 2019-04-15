//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Dmitry Alexeev.
//

#pragma once

#include "Settings.h"

struct MDPdescriptor
{
  // This struct contains all information to fully define the state and action
  // space of an agent. Only source of complication is that this must hold for
  // both discrete and continuous action problems

  ///////////////////////////// STATE DESCRIPTION /////////////////////////////
  // Number of state dimensions and number of state dims observable to learner:
  Uint dimState = 0, dimStateObserved = 0;
  // vector specifying whether a state component is observable to the learner:
  std::vector<bool> bStateVarObserved;
  // mean and scale of state variables: will be computed from replay memory:
  Rvec stateMean, stateStdDev, stateScale;

  // TODO: vector describing shape of state. To enable environment having
  // separate preprocessing for sight as opposed to otehr sensors.
  // This is vector of vectors because each input type will have a vector
  // describing its shape. ( eg. [84 84 1] for atari )
  // std::vector<std::vector<int>> stateShape;

  ///////////////////////////// ACTION DESCRIPTION /////////////////////////////
  // dimensionality of action vector
  Uint dimAction;
  // dimensionality of policy vector (typically 2*dimAction for continuous act,
  // which are mean and diag covariance, or dimAction for discrete policy)
  Uint policyVecDim;

  // whether action have a lower && upper bounded (bool)
  // if true scaled action = tanh ( unscaled action )
  std::vector<bool> bActionSpaceBounded; // TODO 2 bools for semibounded
  // these values are used for scaling or, in case of bounded spaces, as bounds:
  Rvec upperActionValue, lowerActionValue;

  bool bDiscreteActions = false;
  // DISCRETE ACTION stuff:
  //each component of action vector has a vector of possible values:
  std::vector<Uint> discreteActionValues;
  Uint maxActionLabel; //number of actions options for discretized act spaces
  // to map between value and dicrete action option we need 'shifts':
  std::vector<Uint> discreteActionShifts;

  void synchronizeDescriptor( const std::function<void(void*, size_t)>& );
};

struct StateInfo
{
  MDPdescriptor& MDP;
  StateInfo(MDPdescriptor& MDP_) : MDP(MDP_) {}
  StateInfo(const StateInfo& SI) : MDP(SI.MDP) {}

  Uint dim() const { return MDP.dimState; }

  template<typename T = Real>
  std::vector<T> state2observed(const Rvec& state) const
  {
    std::vector<T> ret(MDP.dimStateObserved);
    for (Uint i=0, k=0; i<MDP.dimState; ++i)
      if (MDP.bStateVarObserved[i]) ret[k++] = state[i];
    return ret;
  }

  void scale(std::vector<T>& observed) const
  {
    assert(observed.size() == MDP.dimStateObserved);
    for (Uint i=0; i<MDP.dimStateObserved; ++i)
      observed[i] = ( observed[i] - MPP.stateMean[i] ) * MDP.stateScale[i];
  }

  template<typename T = Real>
  std::vector<T> getScaled(const std::vector<S>& observed) const
  {
    assert(observed.size() == MDP.dimStateObserved);
    std::vector<T> ret(MDP.dimStateObserved);
    for (Uint i=0; i<MDP.dimStateObserved; ++i)
      ret = ( observed[i] - MPP.stateMean[i] ) * MDP.stateScale[i];
  }
};

struct ActionInfo
{
  MDPdescriptor& MDP;
  ActionInfo(MDPdescriptor & MDP_) : MDP(MDP_) {}
  ActionInfo(const ActionInfo& AI) : MDP(AI.MDP) {}

  ///////////////////////////// CONTINUOUS ACTIONS /////////////////////////////
  Real getActMaxVal(const Uint i) const { return MDP.upperActionValue[i]; }
  Real getActMinVal(const Uint i) const { return MDP.lowerActionValue[i]; }
  Uint dim() const { return MDP.dimAction; }

  static inline Real _tanh(const Real x) {
    const Real e2x = std::exp( -2 * x );
    return (1-e2x)/(1+e2x);
  }
  static inline Real _invTanh(const Real y) {
    assert(std::fabs(y) < 1);
    return std::log( (y+1)/(1-y) ) / 2;
  }

  Rvec action2scaledAction(const Rvec& unscaled) const
  {
    assert(not bDiscreteActions);
    Rvec ret(MDP.dimAction);
    assert( unscaled.size() == ret.size() );
    for (Uint i=0; i<MDP.dimAction; ++i)
    {
      const Real y = MDP.bActionSpaceBounded[i]? _tanh(unscaled[i]):unscaled[i];
      const Real min_a=MDP.lowerActionValue[i], max_a=MDP.upperActionValue[i];
      assert( max_a - min_a > std::numeric_limits<Real>::epsilon() );
      ret[i] = min_a + (max_a-min_a)/2 * (y + 1);
    }
    return ret;
  }

  Rvec scaledAction2action(const Rvec& scaled) const
  {
    assert(not bDiscreteActions);
    Rvec ret(MDP.dimAction);
    assert( scaled.size() == ret.size() );
    for (Uint i=0; i<MDP.dimAction; ++i)
    {
      const Real min_a=MDP.lowerActionValue[i], max_a=MDP.upperActionValue[i];
      assert( max_a - min_a > std::numeric_limits<Real>::epsilon() );
      const Real y = 2 * (scaled[i] - min_a)/(max_a - min_a) - 1;
      ret[i] = MDP.bActionSpaceBounded[i] ? _invTanh(y) : y;
    }
    return ret;
  }
  /////////////////////////// CONTINUOUS ACTIONS END ///////////////////////////

  ////////////////////////////// DISCRETE ACTIONS //////////////////////////////
  Uint action2label(const Rvec& action) const
  {
    //map from discretized action (entry per component of values vectors) to int
    assert(bDiscreteActions);
    assert(action.size() == MDP.dimAction);
    assert(MDP.discreteActionShifts.size() == MDP.dimAction);
    Uint label = 0;
    for (Uint i=0; i < MDP.dimAction; ++i) {
      const Uint nOptions = MDP.discreteActionValues[i];
      // actions are passed around like doubles, but here map to int
      const Uint valI = std::floor(action[i]);
      assert(valI < MDP.discreteActionValues[i]);
      label += MDP.discreteActionShifts[i] * valI;
    }
    assert(label < MDP.maxActionLabel);
    return label;
  }

  Rvec label2action(Uint label) const
  {
    assert(bDiscreteActions);
    assert(label < MDP.maxActionLabel);
    //map an int to the corresponding entries in the values vec
    Rvec action( MDP.dimAction );
    for (Uint i = MDP.dimAction; i>0; --i) {
      const Uint index_i = label / MDP.discreteActionShifts[i-1];
      assert(index_i < MDP.discreteActionValues[i]);
      action[i-1] = index_i + 0.1; // convert to real to move around
      label = label % MDP.discreteActionShifts[i-1];
    }
    return action;
  }

  void testDiscrete()
  {
    for(Uint i=0; i<MDP.maxActionLabel; i++)
      if(i != action2label(label2action(i)))
        _die("label %u does not match for action [%s]. returned %u",
          i, print(label2action(i)).c_str(), action2label(label2action(i)) );
  }
  //////////////////////////// DISCRETE ACTIONS END ////////////////////////////
};
