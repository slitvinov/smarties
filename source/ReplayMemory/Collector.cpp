//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Collector.h"
#include "Utils/FunctionUtilities.h"
#include "DataCoordinator.h"

namespace smarties
{

Collector::Collector(MemoryBuffer*const RM, DataCoordinator*const C) :
replay(RM), sharing(C)
{
  inProgress.resize(distrib.nAgents, nullptr);
  for (Uint i=0; i<distrib.nAgents; ++i) inProgress[i] = new Sequence();
}

// Once learner receives a new observation, first this function is called
// to add the state and reward to the memory buffer
// this is called first also bcz memory buffer is used by net to pick new action
void Collector::add_state(Agent&a)
{
  assert(a.ID < inProgress.size());
  assert(MDP.localID == a.localID);
  Sequence* const S = inProgress[a.ID];

  // assign or check id of agent generating episode
  if (a.agentStatus == INIT) S->agentID = a.localID;
  else assert(S->agentID == a.localID);

  const Fvec storedState = a.getObservedState<Fval>();
  if(a.trackSequence == false)
  {
    // contain only one state and do not add more. to not store rewards either
    // RNNs then become automatically not supported because no time series!
    // (this is accompained by check in approximator)
    S->states  = std::vector<Fvec>{ storedState };
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
      const std::vector<nnReal> vecSold = a.getObservedOldState<nnReal>();
      const auto memSold = S->states.back();
      static constexpr Fval fEPS = std::numeric_limits<Fval>::epsilon();
      for (Uint i=0; i<vecSold.size() && same; ++i)
        same = same && std::fabs(memSold[i]-vecSold[i]) < 100*fEPS;
      //debugS("Agent %s and %s",
      //  print(vecSold).c_str(), print(memSold).c_str() );
      if (!same) die("Unexpected termination of sequence");
    }
  #endif

  // environment interface can overwrite reward. why? it can be useful.
  //env->pickReward(a);
  S->ended = a.agentStatus == TERM;
  S->states.push_back(storedState);
  S->rewards.push_back(a.reward);
  if( a.agentStatus not_eq INIT ) S->totR += a.reward;
  else assert(std::fabs(a.reward)<2.2e-16); //rew for init state must be 0
}

// Once network picked next action, call this method
void Collector::add_action(const Agent& a, const Rvec pol)
{
  assert(pol.size() == aI.dimPol());
  assert(a.agentStatus < TERM);
  if(a.trackSequence == false) {
    // do not store more stuff in sequence but also do not track data counter
    inProgress[a.ID]->actions = std::vector<std::vector<Real>>{ a.action };
    inProgress[a.ID]->policies = std::vector<std::vector<Real>>{ pol };
    return;
  }

  if(a.agentStatus not_eq INIT) nSeenTransitions_loc ++;
  inProgress[a.ID]->actions.push_back( a.action );
  inProgress[a.ID]->policies.push_back(pol);
  if(distrib.logAllSamples) // TODO was learner rank
    a.writeData(distrib.world_rank, pol, nSeenTransitions_loc.load());
}

// If the state is terminal, instead of calling `add_action`, call this:
void Collector::terminate_seq(Agent&a)
{
  assert(a.agentStatus >= TERM);
  if(a.trackSequence == false) return; // do not store seq
  // fill empty action and empty policy:
  const Rvec dummyAct = Rvec(aI.dim(), 0), dummyPol = Rvec(aI.dimPol(), 0);

  a.act(dummyAct);
  inProgress[a.ID]->actions.push_back( dummyAct );
  inProgress[a.ID]->policies.push_back( dummyPol );

  if(distrib.logAllSamples) // TODO was learner rank
    a.writeData(distrib.world_rank, dummyPol, nSeenTransitions_loc.load());
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
    if( replay->bRequireImportanceSampling() )
      inProgress[agentId]->priorityImpW = std::vector<float>(seq_len, 1);

    sharing->addComplete(inProgress[agentId]);
    nSeenTransitions_loc++;
    nSeenSequences_loc++;
  }
  else die("Found episode with no steps");

  inProgress[agentId] = new Sequence();
}

Collector::~Collector() {
  for (auto & S : inProgress) Utilities::dispose_object(S);
}

}
