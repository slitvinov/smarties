//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Collector.h"

Collector::Collector(const Settings&S, Learner*const L, MemoryBuffer*const RM) :
 settings(S), env(RM->env), replay(RM), sharing( new DataSharing(S, L, RM) )
{
  assert(_s.nAgents>0);
  inProgress.resize(_s.nAgents);
  for (int i=0; i<_s.nAgents; i++) inProgress[i] = new Sequence();
}

// Once learner receives a new observation, first this function is called
// to add the state and reward to the memory buffer
// this is called first also bcz memory buffer is used by net to pick new action
void Collector::add_state(const Agent&a)
{
  // if no tuples, init state. if tuples, cannot be initial state:
  assert( (inProgress[a.ID]->tuples.size() == 0) == (a.Status == INIT_COMM) );

  #ifndef NDEBUG // check that last new state and new old state are the same
    if(inProgress[a.ID]->tuples.size()) {
      bool same = true;
      const Rvec vecSold = a.sOld.copy_observed();
      const auto memSold = inProgress[a.ID]->tuples.back()->s;
      for (Uint i=0; i<vecSold.size() && same; i++)
        same = same && std::fabs(memSold[i]-vecSold[i]) < 2e-7;
      //debugS("Agent %s and %s",
      //  print(vecSold).c_str(), print(memSold).c_str() );
      if (!same) die("Unexpected termination of sequence");
    }
  #endif

  // environment interface can overwrite reward. why? it can be useful.
  env->pickReward(a);
  inProgress[a.ID]->ended = a.Status==TERM_COMM;
  inProgress[a.ID]->add_state(a.s.copy_observed(), a.r);
}

// Once network picked next action, call this method
void Collector::add_action(const Agent& a, const Rvec pol)
{
  assert(pol.size() == policyVecDim);
  assert(a.Status < TERM_COMM);
  if(a.Status not_eq INIT_COMM) nSeenTransitions_loc ++;

  inProgress[a.ID]->add_action(a.a.vals, pol);
  if(bWriteToFile) a.writeData(learn_rank, pol, nSeenTransitions);
}

// If the state is terminal, instead of calling `add_action`, call this:
void Collector::terminate_seq(Agent&a)
{
  assert(a.Status>=TERM_COMM);
  assert(inProgress[a.ID]->tuples.back()->mu.size() == 0);
  assert(inProgress[a.ID]->tuples.back()->a.size()  == 0);
  // fill empty action and empty policy:
  a.act(Rvec(env->aI.dim, 0));
  inProgress[a.ID]->add_action(a.a.vals, Rvec(policyVecDim, 0));

  if(bWriteToFile)
    a.writeData(learn_rank, Rvec(policyVecDim, 0), nSeenTransitions);
  push_back(a.ID);
}

// Transfer a completed trajectory from the `inProgress` buffer to the data set
void Collector::push_back(const int & agentId)
{
  if(inProgress[agentId]->tuples.size() > 2 ) //at least s0 and sT
  {
    inProgress[agentId]->finalize( readNSeenSeq() );
    sharing->addComplete(inProgress[agentId]);
    nSeenTransitions_loc++;
    nSeenSequences_loc++;
  }
  else
  {
    printf("Trashing %lu obs.\n",inProgress[agentId]->tuples.size());
    _dispose_object(inProgress[agentId]);
  }
  inProgress[agentId] = new Sequence();
}

Collector::~Collector() {
  delete sharing;
  for (auto & trash : inProgress) _dispose_object( trash);
}
