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
  bool bInitialiazed = false;

  ///////////////////////////// STATE DESCRIPTION /////////////////////////////
  // Number of state dimensions and number of state dims observable to learner:
  Uint dimState = 0, dimStateObserved = 0;
  // vector specifying whether a state component is observable to the learner:
  std::vector<bool> bStateVarObserved;
  Rvec stateMean, stateScale; // mean and scale of state variables

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
  Rvec upperActionBoun, lowerActionBound;

  bool bDiscreteActions = false;
  // DISCRETE ACTION stuff:
  //each component of action vector has a vector of possible values:
  std::vector<Rvec> discreteActionValues;
  Uint maxActionLabel; //number of actions options for discretized act spaces
  // to map between value and dicrete action option we need 'shifts':
  std::vector<Uint> discreteActionShifts;
};

struct StateInfo
{
  MDPdescriptor& MDP;
  StateInfo(MDPdescriptor& MDP_) : MDP(MDP_) {}


  //functions returning std, mean, 1/std of observale state components
  std::vector<memReal> inUseStd() const;
  std::vector<memReal> inUseMean() const;
  std::vector<memReal> inUseInvStd() const;
};

struct ActionInfo
{
  MDPdescriptor& MDP;
  ActionInfo(MDPdescriptor& MDP_) : MDP(MDP_) {}

  ///////////////////////////// CONTINUOUS ACTIONS /////////////////////////////
  Real getActMaxVal(const Uint i) const { return MDP.upperActionBound[i]; }
  Real getActMinVal(const Uint i) const { return MDP.lowerActionBound[i]; }
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
    Rvec ret(MDP.dimAction);
    assert( unscaled.size() == ret.size() );
    for (Uint i=0; i<MDP.dimAction; i++)
    {
      const Real y = MDP.bActionSpaceBounded[i]? _tanh(unscaled[i]):unscaled[i];
      const Real min_a=MDP.lowerActionBound[i], max_a=MDP.upperActionBound[i];
      assert( max_a - min_a > std::numeric_limits<Real>::epsilon() );
      ret[i] = min_a + (max_a-min_a)/2 * (y + 1);
    }
    return ret;
  }

  Rvec scaledAction2action(const Rvec& scaled) const
  {
    Rvec ret(MDP.dimAction);
    assert( scaled.size() == ret.size() );
    for (Uint i=0; i<MDP.dimAction; i++)
    {
      const Real min_a=MDP.lowerActionBound[i], max_a=MDP.upperActionBound[i];
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
    assert(action.size() == MDP.dimAction);
    assert(MDP.discreteActionShifts.size() == MDP.dimAction);
    Uint label = 0;
    for (Uint i=0; i < MDP.dimAction; i++) {
      const auto& options = MDP.discreteActionValues[i];
      // From real action for i-th component of action vector
      // convert to an entry in MDP.discreteActionValues vector
      Real dist = 1e9; Uint index = 0;
      for (Uint j=0; j < options.size(); j++) {
        const Real _dist = std::fabs(options[j] - action[i]);
        if (_dist < dist) { dist = _dist; index = j; }
      }
      label += MDP.discreteActionShifts[i] * index;
    }
    assert(label < MDP.maxActionLabel);
    return label;
  }

  Rvec label2action(Uint label) const
  {
    assert(label < MDP.maxActionLabel);
    //map an int to the corresponding entries in the values vec
    Rvec action( MDP.dimAction );
    for (Uint i = MDP.dimAction; i>0; i--) {
      const Uint index_i = label / MDP.discreteActionShifts[i-1];
      action[i-1] = values[i-1][index_i];
      label = label % MDP.discreteActionShifts[i-1];
    }
    return action;
  }

  void updateDiscreteActionShifts()
  {
    if(MDP.bDiscreteActions == false) return;

    MDP.discreteActionShifts = std::vector<Uint>(MDP.dimAction);
    MDP.discreteActionShifts[0] = 1;
    for (Uint i=1; i < MDP.dimAction; i++) {
      assert(MDP.discreteActionValues[i-1].size() > 0);
      MDP.discreteActionShifts[i] = MDP.discreteActionShifts[i-1] *
                                    MDP.discreteActionValues[i-1].size();
    }

    MDP.maxActionLabel = MDP.discreteActionShifts[MDP.dimAction-1] *
                         MDP.discreteActionValues[MDP.dimAction-1].size();

    #ifndef NDEBUG
    for(Uint i=0; i<MDP.maxActionLabel; i++)
      if(i != action2label(label2action(i)))
        _die("label %u, action [%s], ret %u",
          i, print(label2action(i)).c_str(), action2label(label2action(i)) );
    #endif
  }

  //////////////////////////// DISCRETE ACTIONS END ////////////////////////////
};

class State
{
 public:
  const StateInfo& sInfo;
  Rvec vals;

  State(const StateInfo& newSInfo) : sInfo(newSInfo) {
    vals.resize(sInfo.dim);
  };

  State& operator= (const State& s) {
    if (sInfo.dim != s.sInfo.dim) die("Dimension of states differ!!!\n");
    for (Uint i=0; i<sInfo.dim; i++) vals[i] = s.vals[i];
    return *this;
  }

  inline std::string _print() const {
    return print(vals);
  }

  inline void copy_observed(Rvec& res, const Uint append=0) const {
    //copy state into res, append is used to chain multiple states together
    Uint k = append*sInfo.dimUsed;
    assert(res.size() >= k+sInfo.dimUsed);
    for (Uint i=0; i<sInfo.dim; i++)
      if (sInfo.inUse[i]) res[k++] = vals[i];
  }

  template<typename T = Real>
  inline std::vector<T> copy_observed() const {
    std::vector<T> ret(sInfo.dimUsed);
    for (Uint i=0, k=0; i<sInfo.dim; i++)
      if (sInfo.inUse[i]) ret[k++] = vals[i];
    return ret;
  }

  inline void copy(Rvec& res) const {
    assert(res.size() == sInfo.dim);
    for (Uint i=0; i<sInfo.dim; i++) res[i] = vals[i];
  }

  template<typename T>
  inline void set(const std::vector<T>& data) {
    assert(data.size() == sInfo.dim);
    for (Uint i=0; i<sInfo.dim; i++) vals[i] = data[i];
  }
};



class Action
{
 public:
  const ActionInfo& actInfo;
  Rvec vals;

  Action(const ActionInfo& newActInfo) :  actInfo(newActInfo)
  {
    vals.resize(actInfo.dim);
  }

  Action& operator= (const Action& a)
  {
    if (actInfo.dim != a.actInfo.dim) die("Dimension of actions differ!!!");
    for (Uint i=0; i<actInfo.dim; i++) vals[i] = a.vals[i];
    return *this;
  }

  inline std::string _print() const
  {
    return print(vals);
  }

  inline void set(const Rvec& data)
  {
    assert(data.size() == actInfo.dim);
    vals = data;
  }

  inline void set(const Uint label)
  {
    vals = actInfo.labelToAction(label);
  }

  inline Uint getActionLabel() const
  {
    return actInfo.actionToLabel(vals);
  }
};
