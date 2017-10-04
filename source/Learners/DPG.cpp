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
#include "DPG.h"

DPG::DPG(MPI_Comm comm, Environment*const _env, Settings & _s) :
Learner_utils(comm,_env,_s,_s.nnOutputs), nA(_env->aI.dim),
nS(_env->sI.dimUsed*(1+_s.appendedObs)), cntValGrad(nThreads+1,0),
avgValGrad(nThreads+1,vector<long double>(1,0)), stdValGrad(nThreads+1,vector<long double>(1,0))
{
  stdValGrad[0] = vector<long double>(1,100);
  #ifdef NDEBUG
  if(bRecurrent) die("DPG with RNN is Not ready!\n");
  #endif

  buildNetwork(vector<Uint>(1,1), _s, vector<Uint>(1,nA));
  net_value = net; series_1_value = series_1; currAct_value = currAct;
  opt_value = opt; series_2_value = series_2; prevAct_value = prevAct;
  //reset containers so that buildNetwork allocates new memory for actor:
  series_1.resize(0); series_2.resize(0); currAct.resize(0); prevAct.resize(0);

  buildNetwork(vector<Uint>(1,nA), _s);
  policyVecDim = 2*nA;
}

void DPG::select(const int agentId, const Agent& agent)
{
  vector<Real> beta(policyVecDim,0);
  if(agent.Status==2) { data->passData(agentId, agent, beta); return; }

  vector<Real> output = output_value_iteration(agentId, agent);
  const Real annealedVar = bTrain ? .2*annealingFactor()+greedyEps : greedyEps;

  if(positive(annealedVar)) {
    std::normal_distribution<Real> dist(0, annealedVar);
    for(Uint i=0; i<nA; i++) {
      beta[i] = output[i];
      beta[i+nA] = 1/annealedVar/annealedVar;
      output[i] += dist(*gen);
    }
  }

  //scale back to action space size:
  agent.a->set(aInfo.getScaled(output));
  data->passData(agentId, agent, beta);
  dumpNetworkInfo(agentId);
}

void DPG::Train_BPTT(const Uint seq, const Uint thrID) const
{
  const Real rGamma = annealedGamma();
  const Uint ndata = data->Set[seq]->tuples.size();
  const Uint ntgts = data->Set[seq]->ended ? ndata-1 : ndata;
  Grads* tmp_grad = new Grads(net_value->getnWeights(),net_value->getnBiases());
  vector<Activation*>valSeries=net_value->allocateUnrolledActivations(ndata-1);
  vector<Activation*>polSeries=net->allocateUnrolledActivations(ndata);
  Activation* tgtAct = net_value->allocateActivation();
  vector<Real> qcurrs(ndata-1), vnexts(ndata-1);

  for (Uint k=0; k<ntgts; k++)
  { //state in k=[0:N-2], act&rew in k+1, last state (N-1) not used for Q update
    const Tuple*const t_  = data->Set[seq]->tuples[k]; //contains sOld
    vector<Real> s = data->standardize(t_->s);
    vector<Real> pol(nA), val(1), polgrad(nA);
    net->predict(s, pol, polSeries, k); //Compute policy with state as input

    //Advance target network with state and policy
    s.insert(s.end(), pol.begin(), pol.end());
    //Prev step action "a" was performed, not policy. Therefore, next target is
    //computed with recur inputs from value-net computed with cur weights:
    const Activation*const recur = k>0 ? valSeries[k-1] : nullptr;
    net_value->predict(s, val, recur, tgtAct, net_value->tgt_weights, net_value->tgt_biases);
    if(k) vnexts[k-1] = val[0];

    if(k==ndata-1) continue;
    //Advance current network with state and action
    const vector<Real> a = aInfo.getInvScaled(data->Set[seq]->tuples[k]->a);
    for(Uint i=0; i<nA; i++) s[nInputs+i] = a[i];
    net_value->predict(s, val, valSeries, k); //Compute value
    qcurrs[k] = val[0];

    //only one-step backprop because policy-net tries to maximize Q given past transitions, so cannot affect previous Q
    net_value->backProp(vector<Real>(1,1), tgtAct, net_value->tgt_weights_back, net_value->tgt_biases, tmp_grad);

    for(Uint j=0; j<nA; j++) polgrad[j]= tgtAct->errvals[net_value->iInp[nS+j]];
    statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], polgrad);
    clip_gradient(polgrad, stdGrad[0], seq, k);
    net->setOutputDeltas(polgrad, polSeries[k]);
  }

  //im done using the term state for the policy, and i want to bptt:
  delete polSeries.back();
  polSeries.pop_back();
  if (thrID==0) net->backProp(polSeries, net->grad);
  else net->backProp(polSeries, net->Vgrad[thrID]);

  for (Uint k=0; k<ndata-1; k++)
  {
    vector<Real> gradient(1);
    const Tuple*const _t  = data->Set[seq]->tuples[k+1];
    const bool terminal = k+2==ndata && data->Set[seq]->ended;
    const Real target = (terminal) ? _t->r : _t->r + rGamma*vnexts[k];
    gradient[0] = target - qcurrs[k];
    data->Set[seq]->tuples[k]->SquaredError = gradient[0]*gradient[0];
    statsGrad(avgValGrad[thrID+1],stdValGrad[thrID+1],cntValGrad[thrID+1],gradient);
    clip_gradient(gradient, stdValGrad[0], seq, k);
    net_value->setOutputDeltas(gradient, valSeries[k]);
    dumpStats(Vstats[thrID], qcurrs[k], gradient[0]);
  }
  if (thrID==0) net_value->backProp(valSeries, net_value->grad);
  else net_value->backProp(valSeries, net_value->Vgrad[thrID]);

  net_value->deallocateUnrolledActivations(&valSeries);
  net->deallocateUnrolledActivations(&polSeries);
  _dispose_object(tgtAct);
  _dispose_object(tmp_grad);
}

void DPG::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  const Real rGamma = annealedGamma();
  const Uint ndata = data->Set[seq]->tuples.size(), nMaxBPTT = MAX_UNROLL_BFORE;
  const Uint iRecurr = bRecurrent ? max(nMaxBPTT,samp)-nMaxBPTT : samp;
  const Uint nRecurr = bRecurrent ? min(nMaxBPTT,samp)+1 : 1;
  const bool terminal = samp+2==ndata && data->Set[seq]->ended;
  vector<Activation*> actPolcur=net->allocateUnrolledActivations(nRecurr);
  vector<Activation*> actValcur=net_value->allocateUnrolledActivations(nRecurr);
  Grads*const tmp = new Grads(net_value->getnWeights(),net_value->getnBiases());
  Activation* tgtVal = net_value->allocateActivation();
  Activation* tgtPol = net->allocateActivation();
  vector<Real> vnext(1), vcurr(1), grad_pol(nA), qcurr(1), grad_val(1,1);
  //number of state inputs to value net, =/= nS in case multiple obs fed
  const Uint NSIN = net_value->getnInputs()-nA;

  for (Uint k=iRecurr, j=0; k<samp+1; k++, j++) {
    const vector<Real> a = aInfo.getInvScaled(data->Set[seq]->tuples[k]->a);
    vector<Real> s = data->standardize(data->Set[seq]->tuples[k]->s);
    net->seqPredict_inputs(s, actPolcur[j]);
    s.insert(s.end(), a.begin(), a.end());
    net_value->seqPredict_inputs(s, actValcur[j]);
    if(k==samp) {
      assert(j==nRecurr-1);
      net->seqPredict_execute(actPolcur,actPolcur);
      net_value->seqPredict_execute(actValcur,actValcur);
      const vector<Real> pol = net->getOutputs(actPolcur.back());
      qcurr = net_value->getOutputs(actValcur.back());
      for(Uint i=0; i<nA; i++) s[NSIN+i] = pol[i];
      net_value->predict(s, vcurr, actValcur[j-1], tgtVal, net_value->tgt_weights, net_value->tgt_biases);
    }
  }
  net_value->backProp(grad_val, tgtVal, net_value->tgt_weights_back, net_value->tgt_biases, tmp);
  for(Uint j=0;j<nA;j++) grad_pol[j] = tgtVal->errvals[net_value->iInp[NSIN+j]];
  statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], grad_pol);
  clip_gradient(grad_pol, stdGrad[0], seq, samp);
  net->setOutputDeltas(grad_pol, actPolcur.back());

  const Tuple * const _t = data->Set[seq]->tuples[samp+1]; //contains sNew, rew
  if(!terminal) {
    vector<Real> snew = data->standardize(_t->s), polnext(nA);
    net->predict(snew, polnext, actPolcur.back(), tgtPol, net->tgt_weights, net->tgt_biases);
    snew.insert(snew.end(), polnext.begin(), polnext.end());
    net_value->predict(snew, vnext, actValcur.back(), tgtVal, net_value->tgt_weights, net_value->tgt_biases);
  }

  const Real target = (terminal) ? _t->r : _t->r + rGamma * vnext[0];
  grad_val[0] = target - qcurr[0];

  data->Set[seq]->tuples[samp]->SquaredError = grad_val[0]*grad_val[0];
  dumpStats(Vstats[thrID], qcurr[0], grad_val[0]);
  statsGrad(avgValGrad[thrID+1], stdValGrad[thrID+1], cntValGrad[thrID+1], grad_val);
  clip_gradient(grad_val, stdValGrad[0], seq, samp);
  net_value->setOutputDeltas(grad_val, actValcur.back());

  if(thrID==0) net_value->backProp(actValcur, net_value->grad);
  else net_value->backProp(actValcur, net_value->Vgrad[thrID]);
  if(thrID==0) net->backProp(actPolcur, net->grad);
  else net->backProp(actPolcur, net->Vgrad[thrID]);

  net->deallocateUnrolledActivations(&actValcur);
  net->deallocateUnrolledActivations(&actPolcur);
  _dispose_object(tgtVal);
  _dispose_object(tgtPol);
  _dispose_object(tmp);
}

void DPG::updateTargetNetwork()
{
  assert(bTrain);
  if (cntUpdateDelay <= 0) { //DQN-style frozen weight
    cntUpdateDelay = tgtUpdateDelay;
    opt_value->moveFrozenWeights(tgtUpdateAlpha);
    opt->moveFrozenWeights(tgtUpdateAlpha);
  }
  if(cntUpdateDelay>0) cntUpdateDelay--;
}

void DPG::stackAndUpdateNNWeights()
{
  assert(nAddedGradients>0 && bTrain);
  opt->nepoch ++;
  opt_value->nepoch ++;
  Uint nTotGrads = nAddedGradients;
  opt_value->stackGrads(net_value->grad, net_value->Vgrad);
  opt->stackGrads(net->grad, net->Vgrad); //add up gradients across threads

  if (learn_size > 1) {
    MPI_Allreduce(MPI_IN_PLACE, net_value->grad->_W, net_value->getnWeights(),
        MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE, net_value->grad->_B, net_value->getnBiases(),
        MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE, net->grad->_W, net->getnWeights(),
        MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE, net->grad->_B, net->getnBiases(),
        MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE,&nTotGrads,1,MPI_UNSIGNED,MPI_SUM,mastersComm);
  }
  //update is deterministic: can be handled independently by each node
  //communication overhead is probably greater than a parallelised sum
  opt->update(net->grad, nTotGrads);
  opt_value->update(net_value->grad, nTotGrads); //update
}

void DPG::processGrads()
{
  const vector<long double> oldValSum=avgValGrad[0], oldValStd=stdValGrad[0];
  const vector<long double> oldsum = avgGrad[0], oldstd = stdGrad[0];
  statsVector(avgValGrad, stdValGrad, cntValGrad);
  statsVector(avgGrad, stdGrad, cntGrad);

  //std::ostringstream o1; o1 << "Grads avg (std): ";
  //for (Uint i=0;i<avgGrad[0].size();i++)
  //  o1<<avgGrad[0][i]<<" ("<<stdGrad[0][i]<<") ";
  //for (Uint i=0;i<avgValGrad[0].size();i++)
  //  o1<<avgValGrad[0][i]<<" ("<<stdValGrad[0][i]<<") ";
  //cout<<o1.str()<<endl;
  if(!learn_rank) {
    ofstream filestats;
    filestats.open("grads.txt", ios::app);
    filestats<<print(avgGrad[0]).c_str()<<" "<<print(stdGrad[0]).c_str()<<" "
      <<print(avgValGrad[0]).c_str()<<" "<<print(stdValGrad[0]).c_str()<<endl;
    filestats.close();
  }
  for (Uint i=0; i<avgGrad[0].size(); i++) {
    avgGrad[0][i] = .99*oldsum[i] +.01*avgGrad[0][i];
    stdGrad[0][i] = max(0.99*oldstd[i], stdGrad[0][i]);
  }
  for (Uint i=0; i<avgValGrad[0].size(); i++) {
    avgValGrad[0][i] = .99*oldValSum[i] +.01*avgValGrad[0][i];
    stdValGrad[0][i] = max(0.99*oldValStd[i], stdValGrad[0][i]);
  }
}
