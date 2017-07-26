/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#include "../StateAction.h"
#include "GAE.h"

GAE::GAE(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner_onPolicy(comm, _env, settings, settings.nnOutputs)
{
  vector<Real> out_weight_inits = {-1, settings.outWeightsPrefac, -1};
  buildNetwork(net, opt, net_outputs, settings, out_weight_inits);
  printf("GAE: Built network with outputs: %s %s\n",
    print(net_indices).c_str(),print(net_outputs).c_str());
  assert(nOutputs == net->getnOutputs());
  assert(nInputs == net->getnInputs());
}

void GAE::buildNetwork(Network*& _net , Optimizer*& _opt, const vector<Uint> nouts, Settings& settings, vector<Real> weightInitFac, const vector<Uint> addedInputs)
{
  const string netType = settings.nnType, funcType = settings.nnFunc;
  const vector<int> lsize = settings.readNetSettingsSize();
  assert(nouts.size()>0);

  if(!weightInitFac.size()) weightInitFac.resize(nouts.size(),-1);
  if(weightInitFac.size()!=nouts.size()) die("Err in output weights factors size\n");

  Builder build(settings);
  //check if environment wants a particular network structure
  if(not env->predefinedNetwork(&build)) build.addInput(nInputs);

  Uint nsplit = min(static_cast<size_t>(settings.splitLayers),lsize.size());
  for(Uint i=0; i<lsize.size()-nsplit; i++)
    build.addLayer(lsize[i], netType, funcType);

  const Uint firstSplit = lsize.size()-nsplit;
  const vector<int> jointLayer = vector<int>{build.getLastLayerID()};

  if(nsplit) {
    #ifdef INTEGRATEANDFIREMODEL
    die("GAE: nsplit with INTEGRATEANDFIREMODEL\n");
    #endif
    for (Uint i=0; i<nouts.size(); i++)
    {
      build.addLayer(lsize[firstSplit], netType, funcType, jointLayer);
      for (Uint j=firstSplit+1; j<lsize.size(); j++)
        build.addLayer(lsize[j], netType, funcType);
      build.addOutput(static_cast<int>(nouts[i]) , "FFNN", weightInitFac[i]);
    }
  } else {
    #ifndef INTEGRATEANDFIREMODEL
      const int sum =static_cast<int>(accumulate(nouts.begin(),nouts.end(),0));
      const Real fac=*max_element(weightInitFac.begin(),weightInitFac.end());
      build.addOutput(sum, "FFNN", jointLayer, fac);
      assert(fac<=1.);
    #else
      build.addOutput(1, "FFNN", jointLayer, -1.);
      build.addOutput(nA,"IntegrateFire","Sigm",jointLayer,weightInitFac[1]);
      #ifdef INTEGRATEANDFIRESHARED
      build.addParamLayer(1, "Linear", 1);
      #else
      build.addParamLayer(nA, "Linear", 1);
      #endif
    #endif
  }

  _net = build.build();
  if(learn_size>1) {
    MPI_Bcast(_net->weights,_net->getnWeights(),MPI_NNVALUE_TYPE,0,mastersComm);
    MPI_Bcast(_net->biases, _net->getnBiases(), MPI_NNVALUE_TYPE,0,mastersComm);
  }

  _net->updateFrozenWeights();
  _net->sortWeights_fwd_to_bck();
  _opt = new AdamOptimizer(_net, profiler, settings);

  if (!learn_rank)
    _opt->save("initial");
  #ifndef NDEBUG
    MPI_Barrier(mastersComm);
    _opt->restart("initial");
    _opt->save("restarted"+to_string(learn_rank));
  #endif
}

void GAE::select(const int agentId, const Agent& agent)
{
  const int thrid = omp_get_thread_num();
  const int workid = retrieveAssignment(agentId);
  //printf("Thread %d working with agent %d on task %d with status %d\n", thrid, agentId, workid, agent.Status);
  //fflush(0);
  if(workid<0) die("FATAL: GAE Workspace not allocated.\n");
  //printf("(%lu %lu %lu)\n", work[workid]->series.size(), work[workid]->actions.size(), work[workid]->rewards.size());
  //fflush(0);

  if(agent.Status==2)
  {
    work[workid]->rewards.push_back(agent.r);
    work[workid]->done = 1;
    #pragma omp flush
    if(!bTrain) return;
    addToNTasks(1);
    #pragma omp task firstprivate(workid)
    {
      const int thrID = omp_get_thread_num();
      assert(thrID>=0);
      Train_BPTT(workid, static_cast<Uint>(thrID));
      nAddedGradients += work[workid]->series.size()-1;
      addToNTasks(-1);
    }
    return;
  }

  if(thrid==1) profiler->stop_start("FWD");
  if(thrid==0 && profiler_ext != nullptr) profiler_ext->stop_start("WORK");

  work[workid]->series.push_back(net->allocateActivation());
  vector<Real> output(nOutputs), input = agent.s->copy_observed();
  //if required, chain together nAppended obs to compose state
  assert(!nAppended); //not supported
  const Uint step = work[workid]->series.size() - 1;
  net->predict(data->standardize(input), output, work[workid]->series, step);

  const auto pol = prepare_policy(output);
  vector<Real> beta_mean=pol.getMean(), beta_std=pol.getStdev(), beta(2*nA,0);
  vector<Real> act(nA,0);
  for(Uint i=0; i<nA; i++) {
    beta[i] = beta_mean[i];
    beta[nA+i] = beta_std[i];
    #ifdef INTEGRATEANDFIREMODEL
    std::lognormal_distribution<Real> dist_cur(beta_mean[i], beta_std[i]);
    #else
    std::normal_distribution<Real> dist_cur(beta_mean[i], beta_std[i]);
    #endif
    act[i] = bTrain ? dist_cur(*gen) : beta_mean[i];
  }

  if(agent.Status!=1) work[workid]->rewards.push_back(agent.r);
  work[workid]->actions.push_back(act);

  #ifndef INTEGRATEANDFIREMODEL
  agent.a->set(aInfo.getScaled(act));
  #else
  agent.a->set(act);
  #endif

  data->writeData(agentId, a, beta);
  if(thrid==0 && profiler_ext != nullptr) profiler_ext->stop_start("COMM");
  if(thrid==1) profiler->pop_stop();
  //data->passData(agentId, agent, beta);
  //dumpNetworkInfo(agentId);
}

void GAE::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  die("ERROR GAE::Train on sequences rather than samples.\n");
}

void GAE::Train_BPTT(const Uint workid, const Uint thrID) const
{
  //printf("GAE Train_BPTT %d %d %lu %lu %lu\n", thrID, workid, work[workid]->series.size(), work[workid]->actions.size(), work[workid]->rewards.size());
  //fflush(0);
  const Uint ndata = work[workid]->series.size();
  assert(work[workid]->actions.size() == ndata);
  assert(work[workid]->rewards.size() == ndata);

  Real A_GAE = 0, Vnext = 0, V_MC = 0;
  if(thrID==1) profiler->stop_start("CMP");

  for (int k=static_cast<int>(ndata)-1; k>=0; k--)
  {
    vector<Real> out = net->getOutputs(work[workid]->series[k]);
    vector<Real> grad = compute(workid, k, A_GAE, Vnext, V_MC, out, thrID);

    //write gradient onto output layer:
    statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], grad);
    clip_grad(grad, stdGrad[0]);
    net->setOutputDeltas(grad, work[workid]->series[k]);
  }

  if(thrID==1) profiler->stop_start("BCK");
  if (thrID==0) net->backProp(work[workid]->series, net->grad);
  else net->backProp(work[workid]->series, net->Vgrad[thrID]);
  if(thrID==1) profiler->pop_stop();
}
