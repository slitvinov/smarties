/*
 *  Learner.h
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Learner_utils.h"

class Learner_onPolicy: public Learner_utils
{
protected:
  const Uint nAgentsPerSlave, nA;

  //TODO should be grouped in a single struct for cleanliness: {
  const vector< vector<Activation*> *> work;
  vector<int> work_assign, work_done;
  const vector< vector<vector<Real>> *> work_actions;
  const vector< vector<Real> *> work_rewards;
  // } (all vectors have size batchsize)
  std::vector<std::mt19937>& generators;

  vector<vector<Activation*>*> alloc_work(const Uint nagents)
  {
    vector<vector<Activation*>*> ret(nagents, nullptr);
    for(Uint i=0; i<nagents; i++) ret[i] = new vector<Activation*>();
    return ret;
  }

  vector<vector<Real>*> alloc_rewards(const Uint nagents)
  {
    vector<vector<Real>*> ret(nagents, nullptr);
    for(Uint i=0; i<nagents; i++) ret[i] = new vector<Real>();
    return ret;
  }

  vector<vector<vector<Real>>*> alloc_actions(const Uint nagents)
  {
    vector<vector<vector<Real>>*> ret(nagents, nullptr);
    for(Uint i=0; i<nagents; i++) ret[i] = new vector<vector<Real>>();
    return ret;
  }

  inline int checkFirstAvailable() const
  {
    int avail=-1;
    for(Uint i=0; i<batchSize && avail<0; i++)
      if(work_assign[i] == -1) avail = i; //first available workspace
    assert(!work_done[avail]);
    return avail;
  }

  inline int retrieveAssignment(const int agentID) const
  {
    int ret = -1;
    for(Uint i=0; i<batchSize && ret<0; i++)
      if(work_assign[i] == agentID && not work_done[i])
        ret = agentID; //write retrieved
    return ret;
  }

  inline void clip_grad(vector<Real>& grad, const vector<long double>& std) const
  {
    for (Uint i=0; i<grad.size(); i++) {
      #ifdef ACER_GRAD_CUT
        if(grad[i] >  ACER_GRAD_CUT*std[i] && std[i]>2.2e-16)
          grad[i] =  ACER_GRAD_CUT*std[i];
        else
        if(grad[i] < -ACER_GRAD_CUT*std[i] && std[i]>2.2e-16)
          grad[i] = -ACER_GRAD_CUT*std[i];
      #endif
    }
  }

public:
  Learner_onPolicy(MPI_Comm mcom,Environment*const _e, Settings&_s, Uint ng):
  Learner_utils(mcom, _e, _s, ng), nAgentsPerSlave(_e->nAgentsPerRank),
  nA(_e->aI.dim), work(alloc_work(batchSize)), work_assign(batchSize,-1),
  work_done(batchSize,0), work_actions(alloc_actions(batchSize)),
  work_rewards(alloc_rewards(batchSize)), generators(_s.generators) {}

  virtual ~Learner_onPolicy()
  {
    for (const auto & dmp : work) {
      net->deallocateUnrolledActivations(dmp);
      delete dmp;
    }
    for (const auto & dmp : work_actions) delete dmp;
    for (const auto & dmp : work_rewards) delete dmp;
  }
  //main training functions:
  int spawnTrainTasks(const int availTasks) override;
  //void applyGradient() override;
  void prepareData() override;
  bool batchGradientReady() override;

  bool readyForAgent(const int slave, const int agentID) override;
  bool slaveHasUnfinishedSeqs(const int slave) const override;
};
