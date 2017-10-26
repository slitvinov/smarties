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

struct Workspace
{
  int agent = -1;
  int done = 0;
  vector<vector<Real>> actions, policy;
  vector<vector<Real>> observations;
  //vector<Activation*> series;
  vector<Real> rewards, GAE, Vst;
  ~Workspace()
  {
    //Network::deallocateUnrolledActivations(&series);
  }
  void clear()
  {
    agent = -1;
    done = 0;
    Vst.clear();
    GAE.clear();
    policy.clear();
    actions.clear();
    rewards.clear();
    observations.clear();
    //Network::deallocateUnrolledActivations(&series);
    //assert(series.size() == 0);
  }

  inline void push_back(const vector<Real>& inp, const vector<Real>& act,
    const vector<Real>&mu, const Real V)
  {
    assert(actions.size() == observations.size());
    assert(actions.size() == policy.size());
    assert(actions.size() == Vst.size());
    actions.push_back(act);
    observations.push_back(inp);
    policy.push_back(mu);
    Vst.push_back(V);
  }
};

class Learner_onPolicy: public Learner_utils
{
protected:
  const Uint nAgentsPerSlave, nEpochs = 10, nHorizon = 4092;
  Uint cntHorizon = 0, cntTrajectories = 0, cntEpoch = 0, cntBatch = 0;
  vector< Workspace *> work;
  vector< Workspace *> completed;
  std::vector<std::mt19937>& generators;

  //vector<Workspace*> alloc_workspace(const Uint nbatch)
  //{
  //  vector<Workspace*> ret(nbatch, nullptr);
  //  for(Uint i=0; i<nbatch; i++) ret[i] = new Workspace();
  //  return ret;
  //}
  mutable std::mutex buffer_mutex;
  inline void addTasks(Workspace* traj)
  {
    lock_guard<mutex> lock(buffer_mutex);
    cntHorizon += traj->GAE.size();
    completed.push_back(traj);
    cntTrajectories++;
  }

  inline int checkFirstAvailable()
  {
    //this is called if an agent is starting a new sequence
    //block creation if we have reached enough data for a batch
    if(cntHorizon>=nHorizon) return -1;

    int avail=-1;
    for(Uint i=0; i<work.size() && avail<0; i++)
      if(work[i]->agent == -1) avail = i; //first available workspace

    if(avail>=0) assert(!work[avail]->done);

    //still nothing available, allocate new workspace
    if(avail<0 && cntHorizon<nHorizon) {
      avail = work.size();
      lock_guard<mutex> lock(buffer_mutex);
      for(Uint i=0; i<nAgentsPerSlave; i++)
        work.push_back(new Workspace());
    }
    return avail;
  }

  inline int retrieveAssignment(const int agentID) const
  {
    int ret = -1;
    for(Uint i=0; i<work.size() && ret<0; i++)
      if(work[i]->agent == agentID && work[i]->done == 0)
        ret = i; //write retrieved
    return ret;
  }

  inline void clip_grad(vector<Real>& grad, const vector<long double>& std) const
  {
    #ifdef ACER_GRAD_CUT
    for (Uint i=0; i<grad.size(); i++) {
        if(grad[i] >  ACER_GRAD_CUT*std[i] && std[i]>2.2e-16)
          grad[i] =  ACER_GRAD_CUT*std[i];
        else
        if(grad[i] < -ACER_GRAD_CUT*std[i] && std[i]>2.2e-16)
          grad[i] = -ACER_GRAD_CUT*std[i];
    }
    #endif
  }

public:
  Learner_onPolicy(MPI_Comm mcom,Environment*const _e, Settings&_s, Uint ng):
  Learner_utils(mcom, _e, _s, ng), nAgentsPerSlave(_e->nAgentsPerRank),
  generators(_s.generators)
  {
    work.reserve(nHorizon);
    completed.reserve(nHorizon/2);
  }

  virtual ~Learner_onPolicy()
  {
    for (const auto & dmp : work) {
      //net->deallocateUnrolledActivations(&(dmp->series));
      delete dmp;
    }
  }
  //main training functions:
  bool unlockQueue() override;
  int spawnTrainTasks(const int availTasks) override;
  void sampleTransitions(Uint&seq, Uint&trans, const Uint thrID);
  void applyGradient() override;
  void prepareData() override;
  bool batchGradientReady() override;
  bool readyForAgent(const int slave) override;
  bool slaveHasUnfinishedSeqs(const int slave) const override;
};
