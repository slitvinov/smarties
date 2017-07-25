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
  vector<vector<Real>> actions;
  vector<Activation*> series;
  vector<Real> rewards;
  ~Workspace()
  {
    Network::deallocateUnrolledActivations(&series);
  }
  void clear()
  {
    agent = -1;
    done = 0;
    actions.clear();
    rewards.clear();
    Network::deallocateUnrolledActivations(&series);
    assert(series.size() == 0);
  }
};

class Learner_onPolicy: public Learner_utils
{
protected:
  const Uint nAgentsPerSlave, nA;
  const vector< Workspace *> work;

  std::vector<std::mt19937>& generators;

  vector<Workspace*> alloc_workspace(const Uint nbatch)
  {
    vector<Workspace*> ret(nbatch, nullptr);
    for(Uint i=0; i<nbatch; i++) ret[i] = new Workspace();
    return ret;
  }

  inline int checkFirstAvailable() const
  {
    int avail=-1;
    for(Uint i=0; i<work.size() && avail<0; i++)
      if(work[i]->agent == -1) avail = i; //first available workspace
    if(avail>=0) assert(!work[avail]->done);
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
  nA(_e->aI.dim), work(alloc_workspace(batchSize)), generators(_s.generators) {}

  virtual ~Learner_onPolicy()
  {
    for (const auto & dmp : work) {
      net->deallocateUnrolledActivations(&(dmp->series));
      delete dmp;
    }
  }
  //main training functions:
  int spawnTrainTasks(const int availTasks) override;
  void applyGradient() override;
  void prepareData() override;
  bool batchGradientReady() override;
  bool readyForAgent(const int slave, const int agentID) override;
  bool slaveHasUnfinishedSeqs(const int slave) const override;
};

/*
  //TODO should be grouped in a single struct for cleanliness: {
  //const vector< vector<Activation*> *> work;
//vector<int> work_assign, work_done;
//const vector< vector<vector<Real>> *> work_actions;
//const vector< vector<Real> *> work_rewards;
// } (all vectors have size batchsize)

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
*/
