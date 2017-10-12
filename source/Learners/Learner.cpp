/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Learner.h"
#include <chrono>

Learner::Learner(MPI_Comm comm, Environment*const _env, Settings & _s) :
mastersComm(comm), env(_env), tgtUpdateDelay((Uint)_s.targetDelay),
nAgents(_s.nAgents), batchSize(_s.batchSize), nAppended(_s.appendedObs),
maxTotSeqNum(_s.maxTotSeqNum),totNumSteps(_s.totNumSteps),nThreads(_s.nThreads),
nSlaves(_s.nSlaves), nSThreads(_s.nThreads), learn_rank(_s.learner_rank),
learn_size(_s.learner_size), nInputs(_s.nnInputs), nOutputs(_s.nnOutputs),
bRecurrent(_s.bRecurrent), bSampleSequences(_s.bSampleSequences),
bTrain(_s.bTrain), tgtUpdateAlpha(_s.targetDelay), greedyEps(_s.greedyEps),
gamma(_s.gamma), epsAnneal(_s.epsAnneal), obsPerStep(_s.obsPerStep),
aInfo(env->aI), sInfo(env->sI), gen(&_s.generators[0])
{
  assert(nThreads>1);
  if(bSampleSequences) printf("Sampling sequences.\n");
  profiler = new Profiler();
  data = new Transitions(mastersComm, env, _s);
}

void Learner::clearFailedSim(const int agentOne, const int agentEnd)
{
  data->clearFailedSim(agentOne, agentEnd);
}

void Learner::pushBackEndedSim(const int agentOne, const int agentEnd)
{
  data->pushBackEndedSim(agentOne, agentEnd);
}

int Learner::spawnTrainTasks(const int availTasks) //this must be called from omp parallel region
{
  if ( !readyForTrain() ) return 0;
  #ifdef FULLTASKING
    if ( !availTasks ) return 0;
    const int nSpawn = availTasks;
  #else
    const int nSpawn = sequences.size();
  #endif

  if(bSampleSequences)
  {
    for (int i=0; i<nSpawn && sequences.size(); i++) {
      const Uint sequence = sequences.back(); sequences.pop_back();
      addToNTasks(1);
      #ifdef FULLTASKING
#pragma omp task firstprivate(sequence) if(readNTasks()<nSThreads)
      #else
#pragma omp task firstprivate(sequence) //if(!availTasks)
      #endif
      {
        const int thrID = omp_get_thread_num();
        if(!thrID) profiler_ext->stop_start("WORK");
        Train_BPTT(sequence, static_cast<Uint>(thrID));
        addToNTasks(-1);
        #pragma omp atomic
        taskCounter++;
      }
    }
  }
  else
  {
    for (int i=0; i<nSpawn && sequences.size(); i++) {
      const Uint sequence = sequences.back(); sequences.pop_back();
      const Uint transition = transitions.back(); transitions.pop_back();
      addToNTasks(1);
      #ifdef FULLTASKING
#pragma omp task firstprivate(sequence,transition) if(readNTasks()<nSThreads)
      #else
#pragma omp task firstprivate(sequence,transition) //if(!availTasks)
      #endif
      {
        const int thrID = omp_get_thread_num();
        if(!thrID) profiler_ext->stop_start("WORK");
        Train(sequence, transition, static_cast<Uint>(thrID));
        addToNTasks(-1);
        #pragma omp atomic
        taskCounter++;
      }
    }
  }

  #ifndef FULLTASKING
    if(!availTasks) return 0;
    #pragma omp taskwait
  #endif
  return 0;
}

void Learner::prepareData() //this cannot be called from omp parallel region
{
  if ( ! readyForTrain() ) return;

  profiler->push_start("PRE");

  if(opt->nepoch%100==0 || data->requestUpdateSamples())
    data->updateSamples(0); //update sampling //syncDataStats

  #ifdef __CHECK_DIFF //check gradients with finite differences
    if (opt->nepoch % 100000 == 0) net->checkGrads();
  #endif

  //CODE TO DO ONLINE UPDATE OF DATA MEAN/STD: unused
  //if (learn_size > 1) {
  //  MPI_Allreduce(MPI_IN_PLACE, &syncDataStats, 1,
  //      MPI_UNSIGNED, MPI_SUM, mastersComm);
  //}
  //if(syncDataStats) data->update_samples_mean(0); //annealFac

  taskCounter = 0;
  sequences.resize(batchSize);
  transitions.resize(batchSize);

  nAddedGradients = bSampleSequences ? sampleSequences(sequences) :
    sampleTransitions(sequences, transitions);

  profiler->stop_start("SLP");
}

void Learner::applyGradient() //this cannot be called from omp parallel region
{
  if(!nAddedGradients) return; //then this was called WITHOUT a batch ready
  assert(taskCounter == batchSize);

  profiler->stop_start("UPW");
  dataUsage += nAddedGradients;
  batchUsage++;

  stackAndUpdateNNWeights();
  updateTargetNetwork();

  if(opt->nepoch%100 ==0) processStats();

  profiler->stop_all();

  if(opt->nepoch%1000==0 && !learn_rank) {
    profiler->printSummary();
    profiler->reset();

    profiler_ext->stop_all();
    profiler_ext->printSummary();
    profiler_ext->reset();
  }
}

Uint Learner::sampleTransitions(vector<Uint>& seq, vector<Uint>& trans)
{
  assert(seq.size() == batchSize && trans.size() == batchSize);
  assert(!bSampleSequences);
  vector<Uint> load(batchSize), sorting(batchSize), s(batchSize), t(batchSize);
  for (Uint i=0; i<batchSize; i++) {
    const int ind = data->sample();
    if(ind<0) die("not enough data");

    int k=0, back=0, indT=data->Set[0]->tuples.size()-1;
    while (ind >= indT) {
      back = indT;
      indT += data->Set[++k]->tuples.size()-1;
    }

    s[i] = k;
    t[i] = ind-back;
    sorting[i] = i;
    //work per transition (applies to algos with off policy corrections):
    load[i] = data->Set[k]->tuples.size()-1 - t[i];
    //load[i] = data->Set[k]->tuples.size()-1; // ~ this would be for RNN
  }

  //sort elements of sorting according to load for each transition:
  const auto compare = [&] (Uint a, Uint b) { return load[a] < load[b]; };
  std::sort(sorting.begin(), sorting.end(), compare);
  assert(load[sorting[0]] <= load[sorting[batchSize-1]]);
  //sort vectors passed to learning algo:
  for (Uint i=0; i<batchSize; i++) {
    trans[i] = t[sorting[i]];
    seq[i] = s[sorting[i]];
  }
  return batchSize; //always add one grad per transition
}

Uint Learner::sampleSequences(vector<Uint>& seq)
{
  assert(seq.size() == batchSize && bSampleSequences);
  Uint _nAddedGradients = 0;
  for (Uint i=0; i<batchSize; i++)
  {
    const int ind = data->sample();
    if(ind<0) die("not enough data");

    seq[i]  = ind;
    //index[i] = ind;
    const Uint seqSize = data->Set[ind]->tuples.size();
    //to normalize mean gradient for update:
    _nAddedGradients += seqSize-1; //last state = terminal, no next reward
  }
  //sort them such that longer ones are started first, reducing overhead!
  const auto compare = [this] (Uint a, Uint b) {
    return data->Set[a]->tuples.size() < data->Set[b]->tuples.size();
  };
  std::sort(seq.begin(), seq.end(), compare);
  assert(data->Set[seq.front()]->tuples.size() <= data->Set[seq.back()]->tuples.size());

  return _nAddedGradients;
}

bool Learner::batchGradientReady()
{
  const Real requestedData = opt->nepoch * obsPerStep /(Real)learn_size;

  //if there is not enough data for training: go back to master:
  #ifdef PACE_SEQUENCES
  const Real dataCounter = data->nSeenSequences   - nData_b4PolUpdates;
  if(!readyForTrain()){nData_b4PolUpdates=data->nSeenSequences;   return false;}
  #else
  const Real dataCounter = data->nSeenTransitions - nData_b4PolUpdates;
  if(!readyForTrain()){nData_b4PolUpdates=data->nSeenTransitions; return false;}
  #endif
  //If I have done too many gradient steps on the avail data, go back to comm
  if( requestedData > dataCounter ) {
    //profiler_ext->stop_start("STOP");
    return false;
  }

  //else if threads finished processing data:
  return taskCounter >= batchSize;
}

bool Learner::unlockQueue()
{
  if ( ! readyForTrain() ) return true;
  const Real requestedData = (opt->nepoch+1) *obsPerStep/(Real)learn_size;
  #ifdef PACE_SEQUENCES
  return data->nSeenSequences  -nData_b4PolUpdates <= requestedData+nSlaves;
  #else
  return data->nSeenTransitions-nData_b4PolUpdates <= requestedData+nSlaves;
  #endif
}

bool Learner::readyForAgent(const int slave)
{
  return unlockQueue(); //same: if too much data stop
}
bool Learner::slaveHasUnfinishedSeqs(const int slave) const
{
  return true; //Learner assumes off-policy algorithm. it can always use more data
}

void Learner::save(string name)
{
  if (!learn_rank) {
    opt->save(name);
    data->save(name);

    const string stuff = name+".status";
    FILE * f = fopen(stuff.c_str(), "w");
    if (f == NULL) die("Save fail\n");
    fprintf(f, "policy iter: %d\n", data->anneal);
    fprintf(f, "optimize epoch: %lu\n", opt->nepoch);
    //fprintf(f, "epoch count: %lu\n", stats.epochCount);
    fprintf(f, "nData_b4PolUpdates: %lu\n", nData_b4PolUpdates);
    fclose(f);
  }
}

void Learner::restart(string name)
{
  int masterRank;
  MPI_Comm_rank(mastersComm, &masterRank);
  printf("Restarting from saved policy...\n");

  data->restartSamples(policyVecDim);
  if (name == "none") return;

  if ( opt->restart(name) )
    printf("Restart successful, moving on...\n");
  else
    printf("Not all policies restarted. \n");
  data->restart(name);

  const string stuff = name+".status";
  FILE * f = fopen(stuff.c_str(), "r");
  if(f != NULL) {
    {
      int val=-1;
      fscanf(f, "policy iter: %d\n", &val);
      if(val>=0) data->anneal = val;
      printf("policy iter: %d\n", data->anneal);
    }{
      long unsigned ret = 0;
      fscanf(f, "optimize epoch: %lu\n", &ret);
      if(ret>0) opt->nepoch = ret;
      printf("optimize epoch: %lu\n", opt->nepoch);
    //}{
    //  long unsigned ret = 0;
    //  fscanf(f, "epoch count: %lu\n", &ret);
    //  stats.epochCount = ret;
    //  printf("epoch count: %lu\n", stats.epochCount);
    }{
      long unsigned ret = 0;
      fscanf(f, "nData_b4PolUpdates: %lu\n", &ret);
      nData_b4PolUpdates = ret;
      printf("nData_b4PolUpdates: %lu\n", nData_b4PolUpdates);
    }
    fclose(f);
  }
  else printf("No status\n");
  save("restarted_policy");
}
