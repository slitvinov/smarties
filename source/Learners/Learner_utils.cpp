/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 * TODO:
  - define a virtual func to remove aux tasks from output
  - fix dump policy to be more general
 */

#include "Learner_utils.h"
#include "../Math/Utils.h"

void Learner_utils::stackAndUpdateNNWeights()
{
  if(!nAddedGradients) die("Error in stackAndUpdateNNWeights\n");
  assert(bTrain);
  opt->nepoch++;
  Uint nTotGrads = nAddedGradients;
  opt->stackGrads(net->grad, net->Vgrad); //add up gradients across threads
  if (learn_size > 1) { //add up gradients across masters
    MPI_Allreduce(MPI_IN_PLACE, net->grad->_W, net->getnWeights(),
        MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE, net->grad->_B, net->getnBiases(),
        MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE,&nTotGrads,1,MPI_UNSIGNED,MPI_SUM,mastersComm);
  }
  //update is deterministic: can be handled independently by each node
  //communication overhead is probably greater than a parallelised sum
  opt->update(net->grad, nTotGrads);
}

void Learner_utils::updateTargetNetwork()
{
  assert(bTrain);
  if (cntUpdateDelay == 0) { //DQN-style frozen weight
    cntUpdateDelay = tgtUpdateDelay;
    opt->moveFrozenWeights(tgtUpdateAlpha);
  }
  if(cntUpdateDelay>0) cntUpdateDelay--;
}

void Learner_utils::buildNetwork(const vector<Uint> nouts, Settings& settings,
 vector<Uint> addedInputs)
{
  Builder build(settings);
  //check if environment wants a particular network structure
  //if (not env->predefinedNetwork(&build)) build.addInput(nInputs);
  addedInputs.insert( addedInputs.begin(), nInputs);
  build.stackSimple( addedInputs, nouts );

  //vector<int> inputToFullyConn = {build.getLastLayerID()};
  //for (Uint i=0; i<addedInputs.size(); i++) {
  //  build.addInput(addedInputs[i]);
  //  inputToFullyConn.push_back(build.getLastLayerID());
  //}

  //const Uint nsplit = min(static_cast<size_t>(settings.splitLayers),lsize.size());
  //for (Uint i=0; i<lsize.size()-nsplit; i++) {
  //  build.addLayer(lsize[i], netType, funcType, inputToFullyConn);
  //  inputToFullyConn.resize(0); //when size is 0 it links to last one
  //}

  //const Uint firstSplit = lsize.size()-nsplit;
  //const vector<int> lastJointLayer =
  //(inputToFullyConn.size()==0) ? vector<int>{build.getLastLayerID()} : inputToFullyConn;
  // ^ were the various inputs already fed to a FC layer? then link to previous

  net = build.build();

  finalize_network(build);
  //for (const auto & l : _net->layers) l->profiler = profiler;
}

vector<Real> Learner_utils::output_stochastic_policy(const int agentId, const Agent& agent) const
{
  const int thrID = omp_get_thread_num();
  if(currAct.size() < nThreads ) net->prepForFwdProp(&currAct, nThreads);
  if(prevAct.size() < nThreads ) net->prepForFwdProp(&prevAct, nThreads);
  Activation* const currActivation = currAct[thrID];

  vector<Real> output(nOutputs), input = agent.s->copy_observed();
  //if required, chain together nAppended obs to compose state
  if (nAppended>0) {
    const Uint sApp = nAppended*sInfo.dimUsed;
    if(agent.Status==1)
      input.insert(input.end(),sApp, 0);
    else {
      assert(data->Tmp[agentId]->tuples.size()!=0);
      const Tuple * const last = data->Tmp[agentId]->tuples.back();
      input.insert(input.end(),last->s.begin(),last->s.begin()+sApp);
      assert(last->s.size()==input.size());
    }
  }

  if(agent.Status==1) {
    net->predict(data->standardize(input), output, currActivation
      #ifdef __EntropySGD //then we sample from target weights
        , net->tgt_weights, net->tgt_biases
      #endif
    );
  } else { //then if i'm using RNN i need to load recurrent connections (else no effect)
    Activation* const prevActivation = prevAct[thrID];
    prevActivation->loadMemory(net->mem[agentId]);
    net->predict(data->standardize(input), output, prevActivation, currActivation
      //#ifdef __EntropySGD //then we sample from target weights
      //  , net->tgt_weights, net->tgt_biases
      //#endif
    );
  }
  //save network transition
  currActivation->storeMemory(net->mem[agentId]);
  return output;
}

vector<Real> Learner_utils::output_value_iteration(const int agentId, const Agent& agent) const
{
  assert(agent.Status==1 || data->Tmp[agentId]->tuples.size());
  Activation* currActivation = net->allocateActivation();
  vector<Real> output(nOutputs), inputs(nInputs,0);
  agent.s->copy_observed(inputs);
  if (agent.Status==1) {
    vector<Real> scaledSold = data->standardize(inputs);
    net->predict(scaledSold, output, currActivation);
  } else {
    //then if i'm using RNN i need to load recurrent connections
    if (nAppended>0) {
      const Tuple * const last = data->Tmp[agentId]->tuples.back();
      assert(last->s.size()==nInputs);
      for(Uint i=0; i<nAppended*sInfo.dimUsed; i++)
        inputs[sInfo.dimUsed + i] = last->s[i];
    }
    Activation* prev = net->allocateActivation();
    prev->loadMemory(net->mem[agentId]);
    net->predict(data->standardize(inputs), output, prev, currActivation);
    _dispose_object(prev);
  }
  //save network transition
  currActivation->storeMemory(net->mem[agentId]);
  _dispose_object(currActivation);
  return output;
}

void Learner_utils::processStats()
{
  stats.minQ= 1e9;stats.MSE =0;stats.dCnt=0;
  stats.maxQ=-1e9;stats.avgQ=0;stats.relE=0;

  for (Uint i=0; i<Vstats.size(); i++) {
    stats.MSE  += Vstats[i]->MSE;
    stats.avgQ += Vstats[i]->avgQ;
    stats.stdQ += Vstats[i]->stdQ;
    stats.dCnt += Vstats[i]->dCnt;
    stats.minQ = std::min(stats.minQ, Vstats[i]->minQ);
    stats.maxQ = std::max(stats.maxQ, Vstats[i]->maxQ);
    Vstats[i]->minQ= 1e9; Vstats[i]->MSE =0; Vstats[i]->dCnt=0;
    Vstats[i]->maxQ=-1e9; Vstats[i]->avgQ=0; Vstats[i]->stdQ=0;
  }

  if (learn_size > 1) {
  MPI_Allreduce(MPI_IN_PLACE,&stats.MSE, 1,MPI_LONG_DOUBLE,MPI_SUM,mastersComm);
  MPI_Allreduce(MPI_IN_PLACE,&stats.dCnt,1,MPI_LONG_DOUBLE,MPI_SUM,mastersComm);
  MPI_Allreduce(MPI_IN_PLACE,&stats.avgQ,1,MPI_LONG_DOUBLE,MPI_SUM,mastersComm);
  MPI_Allreduce(MPI_IN_PLACE,&stats.stdQ,1,MPI_LONG_DOUBLE,MPI_SUM,mastersComm);
  MPI_Allreduce(MPI_IN_PLACE,&stats.minQ,1,MPI_LONG_DOUBLE,MPI_MIN,mastersComm);
  MPI_Allreduce(MPI_IN_PLACE,&stats.maxQ,1,MPI_LONG_DOUBLE,MPI_MAX,mastersComm);
  }

  stats.epochCount++;
  const long double sum=stats.avgQ, sumsq=stats.stdQ, cnt=stats.dCnt;
  //stats.MSE  /= cnt-1;
  stats.MSE   = std::sqrt(stats.MSE/cnt);
  stats.avgQ /= cnt; //stats.relE/=stats.dCnt;
  stats.stdQ  = std::sqrt((sumsq-sum*sum/cnt)/cnt);
  processGrads();
  if(learn_rank) return;

  long double sumWeights = 0, distTarget = 0, sumWeightsSq = 0;

  #pragma omp parallel for reduction(+:sumWeights,distTarget,sumWeightsSq)
  for (Uint w=0; w<net->getnWeights(); w++) {
    sumWeights += std::fabs(net->weights[w]);
    sumWeightsSq += net->weights[w]*net->weights[w];
    distTarget += std::fabs(net->weights[w]-net->tgt_weights[w]);
  }

  printf("%d (%lu), rmse:%Lg, avg_Q:%Lg, std_Q:%Lg, min_Q:%Lg, max_Q:%Lg, weight:[%Lg %Lg %Lg], N:%Lg\n",
    stats.epochCount, opt->nepoch, stats.MSE, stats.avgQ, stats.stdQ,
    stats.minQ, stats.maxQ, sumWeights, sumWeightsSq, distTarget, stats.dCnt);
    fflush(0);

  ofstream filestats;
  filestats.open("stats.txt", ios::app);
  filestats<<stats.epochCount<<"\t"<<opt->nepoch<<"\t"<<stats.MSE<<"\t"
    <<stats.avgQ<<"\t"<<stats.stdQ<<"\t"<<stats.minQ<<"\t"<<stats.maxQ<<"\t"
    <<sumWeights<<"\t"<<sumWeightsSq<<"\t"<<distTarget<<"\t"<<stats.dCnt<<endl;
  filestats.close();
  filestats.flush();

  if (stats.epochCount % 100==0) save("policy");
}

void Learner_utils::processGrads()
{
  const vector<long double> oldsum = avgGrad[0], oldstd = stdGrad[0];
  statsVector(avgGrad, stdGrad, cntGrad);
  //std::ostringstream o; o << "Grads avg (std): ";
  //for (Uint i=0; i<avgGrad[0].size(); i++)
  //  o<<avgGrad[0][i]<<" ("<<stdGrad[0][i]<<") ";
  //cout<<o.str()<<endl;
  if(!learn_rank) {
    ofstream filestats;
    filestats.open("grads.txt", ios::app);
    filestats<<print(avgGrad[0]).c_str()<<" "<<print(stdGrad[0]).c_str()<<endl;
    filestats.close();
  }
  for (Uint i=0; i<avgGrad[0].size(); i++) {
    avgGrad[0][i] = .999*oldsum[i] +.001*avgGrad[0][i];
    //stdGrad[0][i] = .99*oldstd[i] +.01*stdGrad[0][i];
    stdGrad[0][i] = max(0.999*oldstd[i], stdGrad[0][i]);
  }
}

void Learner_utils::statsVector(vector<vector<long double>>& sum,
  vector<vector<long double>>& sqr, vector<long double>& cnt)
{
  assert(sum.size()>1);
  assert(sum.size() == cnt.size() && sqr.size() == cnt.size());

  for (Uint i=0; i<sum[0].size(); i++)
    sum[0][i] = sqr[0][i] = 0;
  cnt[0] = 0;

  for (Uint i=1; i<sum.size(); i++) {
    cnt[0] += cnt[i]; cnt[i] = 0;
    for (Uint j=0; j<sum[0].size(); j++)
    {
      sum[0][j] += sum[i][j]; sum[i][j] = 0;
      sqr[0][j] += sqr[i][j]; sqr[i][j] = 0;
    }
  }
  cnt[0] = std::max((long double)2.2e-16, cnt[0]);

  if (learn_size > 1) {
    MPI_Allreduce(MPI_IN_PLACE, &cnt[0],     1,
        MPI_LONG_DOUBLE, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE, sum[0].data(), sum[0].size(),
        MPI_LONG_DOUBLE, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE, sqr[0].data(), sqr[0].size(),
        MPI_LONG_DOUBLE, MPI_SUM, mastersComm);
  }

  for (Uint j=0; j<sum[0].size(); j++) {
    sqr[0][j] = std::sqrt((sqr[0][j]-sum[0][j]*sum[0][j]/cnt[0])/cnt[0]);
    sum[0][j] /= cnt[0];
  }
}

void Learner_utils::dumpPolicy()
{
  //a fail in any of these amounts to a big and fat TODO
  if(nAppended) die("TODO missing features\n");
  const Uint nDumpPoints = env->getNdumpPoints();
  const Uint n_outs = 4;
  printf("n_outs:%u, nInputs:%u, nDumpPoints:%u\n",n_outs,nInputs, nDumpPoints);
  FILE * pFile = fopen ("dump.raw", "wb");
  vector<Real> output(nOutputs);
  vector<float> dump(nInputs+n_outs);
  Activation* act = net->allocateActivation();
  for (Uint i=0; i<nDumpPoints; i++)
  {
    vector<Real> state = env->getDumpState(i);
    assert(state.size()==nInputs);
    net->predict(data->standardize(state), output, act);
    Uint k=0;
    for (Uint j=0; j<nInputs; j++) dump[k++] = state[j];
    for (Uint j=0; j<n_outs; j++) dump[k++] = output[j];
    //state.insert(state.end(),output.begin(),output.end()); //unsafe
    fwrite(dump.data(),sizeof(float),dump.size(),pFile);
  }
  _dispose_object(act);
  fclose (pFile);
}

void Learner_utils::dumpNetworkInfo(const int agentId) const
{
  #ifdef _dumpNet_
  if (bTrain) return;
  #else
  return;
  #endif
  net->dump(agentId);
  vector<Real> output(nOutputs);
  const Uint ndata = data->Tmp[agentId]->tuples.size(); //last one already placed
  if (ndata == 0) return;

  vector<Activation*> series_base = net->allocateUnrolledActivations(ndata);
  for (Uint k=0; k<ndata; k++) {
    const Tuple * const _t = data->Tmp[agentId]->tuples[k];
    net->predict(data->standardize(_t->s), output, series_base, k);
  }

  string fname="gradInputs_"+to_string(agentId)+"_"+to_string(ndata)+".dat";
  ofstream out(fname.c_str());
  if (!out.good()) _die("Unable to open save into file %s\n", fname.c_str());
  //first row of file is net output:
  for(Uint j=0; j<nOutputs; j++) out << output[j] << " ";
  out << "\n";
  //sensitivity of value for this action in this state wrt all previous inputs
  Uint start0 = ndata > MAX_UNROLL_BFORE ? ndata-MAX_UNROLL_BFORE-1 : 0;
  for (Uint ii=start0; ii<ndata; ii++) {
    Uint start1 = ii > MAX_UNROLL_BFORE ? ii-MAX_UNROLL_BFORE-1 : 0;
    for (Uint i=0; i<nInputs; i++) {
      vector<Activation*> series =net->allocateUnrolledActivations(ndata);
      for (Uint k=start1; k<ndata; k++) {
        vector<Real> state = data->Tmp[agentId]->tuples[k]->s;
        if (k==ii) state[i] = 0;
        net->predict(data->standardize(state), output, series, k);
      }
      vector<Real> oDiff = net->getOutputs(series.back());
      vector<Real> oBase = net->getOutputs(series_base.back());
      //other rows of file are d(net output)/d(state_i(t):
      for(Uint j=0; j<nOutputs; j++) {
        const Real dOut = oDiff[j]-oBase[j];
        const Real dState = data->Tmp[agentId]->tuples[ii]->s[i];
        out << dOut/dState << " ";
      }
      out << "\n";
      net->deallocateUnrolledActivations(&series);
    }
  }
  out.close();
  net->deallocateUnrolledActivations(&series_base);
}
