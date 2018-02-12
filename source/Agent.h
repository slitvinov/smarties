/*
 *  Agent.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "StateAction.h"
#include "Settings.h"
#include "Communicator.h"

class Agent
{
protected:
  StateInfo  sInfo;
  ActionInfo aInfo;

public:
  State *s = nullptr;
  State *sOld = nullptr;
  Action *a = nullptr;
  Real r = 0;
  Real cumulative_rewards = 0;
  const int ID;
  int Status = 1;
  int transitionID = 0;

  Agent(const int _ID = 0) : ID(_ID) { }

  ~Agent()
  {
    _dispose_object(s);
    _dispose_object(sOld);
    _dispose_object(a);
  }

  inline void getState(State& _s) const
  {
    assert(s not_eq nullptr);
    _s = *s;
  }

  inline void setState(State& _s) const
  {
    *s = _s;
  }

  inline void swapStates()
  {
    assert(s not_eq nullptr);
    assert(sOld not_eq nullptr);
    std::swap(s, sOld);
  }

  inline void getAction(Action& _a) const
  {
    assert(a not_eq nullptr);
    _a = *a;
  }

  inline void getOldState(State& _s) const
  {
    assert(sOld not_eq nullptr);
    _s = *sOld;
  }

  inline void act(Action& _a) const
  {
    *a = _a;
  }

  template<typename T>
  inline void act(const T action) const
  {
    a->set(action);
  }

  inline int getStatus() const
  {
    return Status;
  }

  inline Real getReward() const
  {
    return r;
  }

  inline void reset()
  {
    Status = 1; transitionID=0; cumulative_rewards=0; r=0;
  }

  template<typename T>
  void update(const envInfo _i, const vector<T>& _s, const double _r)
  {
    if(_i == FAIL_COMM) {
      cumulative_rewards = transitionID = r = 0;
      return;
    }
    Status = _i;
    swapStates(); //swap sold and snew
    s->set(_s);
    r = _r;
    if(_i == INIT_COMM) {
      cumulative_rewards = 0;
      transitionID = 0;
    }
    else {
      cumulative_rewards += _r;
      transitionID++;
    }
  }
  StateInfo getStateDims() {return sInfo;}
  ActionInfo getActionDims() {return aInfo;}

  void setDims(const StateInfo& stateInfo, const ActionInfo& actionInfo)
  {
    this->aInfo = actionInfo;
    this->sInfo = stateInfo;
  }

  void writeData(const int rank, const Rvec mu) const
  {
    char cpath[256];
    sprintf(cpath, "obs_rank%02d_agent%03d.raw", rank, ID);
    FILE * pFile = fopen (cpath, "ab");
    const Uint writesize = (3 +sInfo.dim +aInfo.dim +mu.size())*sizeof(float);
    float* buf = (float*) malloc(writesize);
    memset(buf, 0, writesize);
    Uint k=0;
    buf[k++] = Status + 0.1;
    buf[k++] = transitionID + 0.1;
    for (Uint i=0; i<sInfo.dim; i++) buf[k++] = (float) s->vals[i];
    for (Uint i=0; i<aInfo.dim; i++) buf[k++] = (float) a->vals[i];
    buf[k++] = r;
    for (Uint i=0; i<mu.size(); i++) buf[k++] = (float) mu[i];
    assert(k*sizeof(float) == writesize);
    fwrite (buf, sizeof(float), writesize/sizeof(float), pFile);
    fflush(pFile); fclose(pFile);  free(buf);
  }
};
