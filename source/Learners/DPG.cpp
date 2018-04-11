/*
 *  NFQ.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#include "../StateAction.h"
#include "../Math/Utils.h"
#include "../Math/Gaussian_policy.h"
#include "../Network/Builder.h"
#include "DPG.h"
//#define DKL_filter

DPG::DPG(Environment*const _env, Settings& _set): Learner_offPolicy(_env,_set),
tgtFrac(_set.klDivConstraint)
{
  _set.splitLayers = 0;
  #if 1
    if(input->net not_eq nullptr) {
      delete input->opt; input->opt = nullptr;
      delete input->net; input->net = nullptr;
    }
    Builder input_build(_set);
    bool bInputNet = false;
    input_build.addInput( input->nOutputs() );
    bInputNet = bInputNet || env->predefinedNetwork(input_build);
    bInputNet = bInputNet || predefinedNetwork(input_build, _set);
    if(bInputNet) {
      Network* net = input_build.build(true);
      input->initializeNetwork(net, input_build.opt);
    }
  #endif

  F.push_back(new Approximator("policy", _set, input, data));
  //F[0]->blockInpGrad = true; // this line must happen b4 initialize
  relay = new Aggregator(_set, data, nA, F[0]);
  F.push_back(new Approximator("critic", _set, input, data, relay));
  Builder build_pol = F[0]->buildFromSettings(_set, nA);

  #if 1
    Builder build_val = F[1]->buildFromSettings(_set, 1 );
  #else
    Builder build_val(_set);
    build_val.stackSimple(input->nOutputs(), {0});
    build_val.addInput(relay->nOutputs()); // add actions
    build_val.addLayer(1, "Linear", true); // output
  #endif
  const Real initParam = Gaussian_policy::precision_inverse(greedyEps);
  build_pol.addParamLayer(nA, "Linear", initParam);
  //relay->scaling = Rvec(nA, 1/greedyEps); //

  _set.outWeightsPrefac = 0.001;
  F[0]->initializeNetwork(build_pol, 0);

  _set.learnrate *= 10; // DPG wants critic faster than actor
  _set.nnLambda = 1e-4; // also wants L2 penl coef
  // we want initial Q to be approx equal to 0 everywhere.
  // if LRelu we need to make initialization multiplier smaller:
  _set.targetDelay = 0.001;
  F[1]->initializeNetwork(build_val, 0);
  printf("DPG\n");
}

void DPG::select(const Agent& agent)
{
  const int thrID= omp_get_thread_num();
  Sequence* const traj = data->inProgress[agent.ID];
  data->add_state(agent);
  if( agent.Status < TERM_COMM ) { // not last of a sequence
    //Compute policy and value on most recent element of the sequence. If RNN
    // recurrent connection from last call from same agent will be reused
    Rvec pol = F[0]->forward_agent(traj, agent, thrID);
    Gaussian_policy policy = prepare_policy(pol);
    Rvec MU = policy.getVector();
    Rvec act = policy.finalize(bTrain, &generators[thrID], MU);
    if(OrUhDecay>0)
      act = policy.updateOrUhState(OrUhState[agent.ID], MU, OrUhDecay);
    agent.a->set(act);
    data->add_action(agent, MU);
    //if(nStep)cout << print(MU) << " " << print(act) << endl;
  } else {
    OrUhState[agent.ID] = Rvec(nA, 0);
    data->terminate_seq(agent);
  }
}

void DPG::Train_BPTT(const Uint seq, const Uint thrID) const
{
  die("");
}

void DPG::Train(const Uint seq, const Uint t, const Uint thrID) const
{
  if(thrID==1) profiler->stop_start("FWD");
  Sequence* const traj = data->Set[seq];
  F[0]->prepare_one(traj, t, thrID);
  F[1]->prepare_one(traj, t, thrID);

  const Rvec polVec = F[0]->forward(traj, t, thrID);
  const Gaussian_policy POL = prepare_policy(polVec, traj->tuples[t]);
  const Real KLdiv = POL.kl_divergence(traj->tuples[t]->mu);
  //if(!thrID) cout<<"tpol "<<print(polVec)<<" act: "<<print(POL.sampAct)<<endl;
  #ifdef DKL_filter
    const bool isOff = traj->distFarPolicy(t, KLdiv, 1+KLdiv, CmaxRet-1);
  #else
    const bool isOff = traj->isFarPolicy(t, POL.sampImpWeight, CmaxRet);
    traj->setSquaredError(t, POL.kl_divergence(traj->tuples[t]->mu));
  #endif
  // if CmaxPol==0 this is never triggered:
  if(isOff && beta>10*learnR && canSkip()) return resample(thrID);

  relay->prepare_one(traj, t, thrID, ACT);
  const Rvec q_curr = F[1]->forward(traj, t, thrID); // inp here is {s,a}

  relay->prepare(NET, thrID); // tell relay to pass policy (output of F[0])
  const Rvec v_curr = F[1]->forward<CUR, TGT>(traj, t, thrID); //here is {s,pi}
  const Rvec detPolG = F[1]->relay_backprop({1}, t, thrID);
  //if(!thrID) cout << "G "<<print(detPolG) << endl;

  Real target = data->scaledReward(traj, t+1);
  if (not traj->isTerminal(t+1)) {
    const Rvec pol_next = F[0]->forward<TGT>(traj, t+1, thrID);
    //if(!thrID) cout << "nterm pol "<<print(pol_next) << endl;
    const Rvec v_next = F[1]->forward<TGT>(traj, t+1, thrID);//here is {s,pi}_+1
    target += gamma * v_next[0];
  }

  { //code to compute policy grad:
    Real a_curr = target - v_curr[0];
    if(a_curr > 0 && POL.sampImpWeight >  2) a_curr = 0;
    if(a_curr < 0 && POL.sampImpWeight < .5) a_curr = 0;
    // this is an experimental change to update stdev using policy gradient
    // not fully analyzed therefore should be turned off by default
    // policy gradient is fully overwritten in 2 and 6 lines from here
    Rvec polG = POL.policy_grad(POL.sampAct, POL.sampImpWeight*a_curr);
    for (Uint i=0; i<nA; i++) polG[i] = detPolG[i];
    #ifdef LearnStDev // ugly hack to keep stdev const == init user value
      if(isZero(a_curr))
    #endif
        for (Uint i=0; i<nA; i++) polG[i+nA] = greedyEps - POL.stdev[i];

    const Rvec penG = POL.div_kl_grad(traj->tuples[t]->mu, -1);
    // if beta=1 (which is inevitable for CmaxPol=0) this will be equal to polG
    Rvec mixG = weightSum2Grads(polG, penG, beta);
    Rvec finalG(F[0]->nOutputs(), 0);
    POL.finalize_grad(mixG, finalG);
    //#pragma omp critical //"O:"<<print(polVec)<<
    //if(!thrID) cout<<"G:"<<print(polG)<<" D:"<<print(penG)<<endl;
    F[0]->backward(finalG, t, thrID);
  }

  { //code to compute value grad:
    const Rvec grad_val = {(target-q_curr[0])};
    //traj->SquaredError[t] = grad_val[0]*grad_val[0];
    Vstats[thrID].dumpStats(q_curr[0], grad_val[0]);
    F[1]->backward(clampGrad(grad_val,q_curr[0]), t, thrID);
  }
  if(thrID==1)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
  F[1]->gradient(thrID);
}

void DPG::prepareGradient()
{
  const bool bWasPrepareReady = updateComplete;

  Learner::prepareGradient();

  if(not bWasPrepareReady) return;

  profiler->stop_start("PRNE");
  advanceCounters();
  CmaxRet = 1 + annealRate(CmaxPol, nStep, epsAnneal);
  if(CmaxRet<=1) die("Either run lasted too long or epsAnneal is wrong.");
  data->prune(CmaxPol>0 ? MAXERROR : OLDEST, CmaxRet);
  Real fracOffPol = data->nOffPol / (Real) data->readNData();
  profiler->stop_start("SLP");

  if (learn_size > 1) {
    const bool firstUpdate = nData_request == MPI_REQUEST_NULL;
    if(not firstUpdate) MPI_Wait(&nData_request, MPI_STATUS_IGNORE);

    // prepare an allreduce with the current data:
    ndata_partial_sum[0] = data->nOffPol;
    ndata_partial_sum[1] = data->readNData();
    // use result from prev AllReduce to update rewards (before new reduce).
    // Assumption is that the number of off Pol trajectories does not change
    // much each step. Especially because here we update the off pol W only
    // if an observation is actually sampled. Therefore at most this fraction
    // is wrong by batchSize / nTransitions ( ~ 0 )
    // In exchange we skip an mpi implicit barrier point.
    fracOffPol = ndata_reduce_result[0] / ndata_reduce_result[1];

    MPI_Iallreduce(ndata_partial_sum, ndata_reduce_result, 2, MPI_DOUBLE,
                   MPI_SUM, mastersComm, &nData_request);
    // if no reduction done, partial sums are meaningless
    if(firstUpdate) return;
  }

  // if CmaxPol<=0 no samples will be counted as far pol, beta will be 1
  if(fracOffPol>tgtFrac) beta = (1-learnR)*beta; // iter converges to 0
  else beta = learnR +(1-learnR)*beta; //fixed point iter converge to 1

  if( beta <= 10*learnR && nStep % 1000 == 0)
  warn("beta too low. Decrease learnrate and/or increase klDivConstraint.");
}

void DPG::getMetrics(ostringstream& buff) const {
  buff<<" "<<std::setw(6)<<std::setprecision(3)<<beta;
}
void DPG::getHeaders(ostringstream& buff) const {
  buff <<"| beta ";
}
