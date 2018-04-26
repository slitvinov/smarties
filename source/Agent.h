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
#define OUTBUFFSIZE 65536
class Agent
{
protected:
  const StateInfo&  sInfo;
  const ActionInfo& aInfo;

public:
  State * const sOld; // previous state
  State * const s   ; // current state
  // Action performed by agent. Updated by Learner::select and sent to Slave
  Action* const a   ;
  Real r = 0;              // current reward
  Real cumulative_rewards = 0;
  const int ID;
  // status of agent's episode. 1: initial; 0: middle; 2: terminal; 3: truncated
  int Status = 1;
  int transitionID = 0;

  // for dumping to state-action-reward-policy binary log (writeBuffer):
  mutable float buf[OUTBUFFSIZE];
  mutable Uint buffCnter = 0;

  Agent(const int _ID, const StateInfo& _sInfo, const ActionInfo& _aInfo) :
    sInfo(_sInfo), aInfo(_aInfo), sOld(new State(_sInfo)),
    s(new State(_sInfo)), a(new Action(_aInfo)), ID(_ID) {
    }

  ~Agent() {
    _dispose_object(s);
    _dispose_object(sOld);
    _dispose_object(a);
  }

  void writeBuffer(const int rank) const
  {
    if(buffCnter == 0) return;
    char cpath[256];
    sprintf(cpath, "obs_rank%02d_agent%03d.raw", rank, ID);
    FILE * pFile = fopen (cpath, "ab");

    fwrite (buf, sizeof(float), buffCnter, pFile);
    fflush(pFile); fclose(pFile);
    buffCnter = 0;
  }

  void writeData(const int rank, const Rvec mu) const
  {
    // possible race conditions, avoided by the fact that each slave
    // (and therefore agent) can only be handled by one thread at the time
    // atomic op is to make sure that counter gets flushed to all threads
    const Uint writesize = 3 +sInfo.dim +aInfo.dim +mu.size();
    if(OUTBUFFSIZE<writesize) die("Edit compile-time OUTBUFFSIZE variable.");
    assert( buffCnter % writesize == 0 );
    if(buffCnter+writesize > OUTBUFFSIZE) writeBuffer(rank);
    Uint ind = buffCnter;
    buf[ind++] = Status + 0.1;
    buf[ind++] = transitionID + 0.1;

    for (Uint i=0; i<sInfo.dim; i++) buf[ind++] = (float) s->vals[i];
    for (Uint i=0; i<aInfo.dim; i++) buf[ind++] = (float) a->vals[i];
    buf[ind++] = r;
    for (Uint i=0; i<mu.size(); i++) buf[ind++] = (float) mu[i];

    #pragma omp atomic
    buffCnter += writesize;
    assert(buffCnter == ind);
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
    std::swap(s->vals, sOld->vals);
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
      cumulative_rewards = 0; transitionID = 0; r = 0;
      return;
    }
    Status = _i;
    swapStates(); //swap sold and snew
    s->set(_s);
    r = _r;
    if(_i == INIT_COMM) {
      cumulative_rewards = 0; transitionID = 0;
    }
    else {
      cumulative_rewards += _r;
      transitionID++;
    }
  }
};
