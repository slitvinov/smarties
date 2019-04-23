//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Collector.h"

Collector::Collector(MemoryBuffer*const RM) : replay(RM), sharing(new MemorySharing(RM))
{
  globalStep_reduce.update({nSeenSequences_loc.load(), nSeenTransitions_loc.load()});
  inProgress.resize(distrib.nAgents, nullptr);
  for (int i=0; i<distrib.nAgents; i++) inProgress[i] = new Sequence();
}

// Once learner receives a new observation, first this function is called
// to add the state and reward to the memory buffer
// this is called first also bcz memory buffer is used by net to pick new action
void Collector::add_state(Agent&a)
{
  assert(a.ID < inProgress.size());
  Sequence* const S = inProgress[a.ID];
  const std::vector<nnReal> storedState = a.s.copy_observed<memReal>();
  if(a.trackSequence == false)
  {
    // contain only one state and do not add more. to not store rewards either
    // RNNs then become automatically not supported because no time series!
    // (this is accompained by check in approximator)
    S->states = std::vector<std::vector<nnReal>>{ storedState };
    S->rewards = std::vector<Real>{ (Real)0 };
    S->SquaredError.clear(); S->Q_RET.clear();
    S->offPolicImpW.clear(); S->action_adv.clear();
    S->KullbLeibDiv.clear(); S->state_vals.clear();
    a.agentStatus = INIT; // one state stored, lie to avoid catching asserts
    return;
  }

  // if no tuples, init state. if tuples, cannot be initial state:
  assert( (S->nsteps() == 0) == (a.agentStatus == INIT) );
  #ifndef NDEBUG // check that last new state and new old state are the same
    if( S->nsteps() ) {
      bool same = true;
      const std::vector<nnReal> vecSold = a.sOld.copy_observed<nnReal>();
      const auto memSold = S->states.back();
      for (Uint i=0; i<vecSold.size() && same; i++)
        same = same && std::fabs(memSold[i]-vecSold[i]) < 1e-6;
      //debugS("Agent %s and %s",
      //  print(vecSold).c_str(), print(memSold).c_str() );
      if (!same) die("Unexpected termination of sequence");
    }
  #endif

  // environment interface can overwrite reward. why? it can be useful.
  //env->pickReward(a);
  S->ended = a.agentStatus == TERM;
  S->states.push_back(storedState);
  S->rewards.push_back(a.r);
  if( a.Status not_eq INIT ) S->totR += a.r;
  else assert(std::fabs(a.r)<2.2e-16); //rew for init state must be 0
}

// Once network picked next action, call this method
void Collector::add_action(const Agent& a, const Rvec pol)
{
  assert(pol.size() == aI.dimPol());
  assert(a.agentStatus < TERM);
  if(a.trackSequence == false) {
    // do not store more stuff in sequence but also do not track data counter
    inProgress[a.ID]->actions = std::vector<std::vector<Real>>{ a.a.vals };
    inProgress[a.ID]->policies = std::vector<std::vector<Real>>{ pol };
    return;
  }

  if(a.agentStatus not_eq INIT) nSeenTransitions_loc ++;
  inProgress[a.ID]->actions.push_back(a.a.vals);
  inProgress[a.ID]->policies.push_back(pol);
  if(distrib.logAllSamples)
    a.writeData(learn_rank, pol, nSeenTransitions_loc.load());
}

// If the state is terminal, instead of calling `add_action`, call this:
void Collector::terminate_seq(Agent&a)
{
  assert(a.agentStatus >= TERM);
  if(a.trackSequence == false) return; // do not store seq
  // fill empty action and empty policy:
  a.act(Rvec(aI.dim(), 0));
  inProgress[a.ID]->actions.push_back(  Rvec(aI.dim(),  0) );
  inProgress[a.ID]->policies.push_back( Rvec(aI.dimPol(), 0) );

  if(distrib.logAllSamples)
    a.writeData(learn_rank, Rvec(policyVecDim, 0), nSeenTransitions_loc.load());
  push_back(a.ID);
}

// Transfer a completed trajectory from the `inProgress` buffer to the data set
void Collector::push_back(const int & agentId)
{
  assert(agentId < (int) inProgress.size());
  const Uint seq_len = inProgress[agentId]->states.size();
  if( seq_len > 1 ) //at least s0 and sT
  {
    inProgress[agentId]->finalize( nSeenSequences_loc.load() );
    if(prepareImpWeights)
      inProgress[agentId]->priorityImpW = std::vector<float>(seq_len, 1);

    sharing->addComplete(inProgress[agentId]);
    nSeenTransitions_loc++;
    nSeenSequences_loc++;
  }
  else die("Found episode with no steps");

  inProgress[agentId] = new Sequence();

  globalStep_reduce.update({nSeenSequences_loc.load(), nSeenTransitions_loc.load()});
  const std::vector<long> nDataGlobal = globalStep_reduce.get();
  nSeenSequences = nDataGlobal[0];
  nSeenTransitions = nDataGlobal[1];
}

Collector::~Collector() {
  delete sharing;
  for (auto & S : inProgress) Utilities::dispose_object(S);
}
