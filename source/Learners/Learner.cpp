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

Learner::Learner(MPI_Comm comm, Environment*const _env, Settings & settings) :
mastersComm(comm), env(_env), nAgents(settings.nAgents),
batchSize(settings.batchSize), tgtUpdateDelay((Uint)settings.targetDelay),
nThreads(settings.nThreads), nInputs(settings.nnInputs),
nOutputs(settings.nnOutputs), nAppended(settings.appendedObs),
bRecurrent(settings.bRecurrent), bTrain(settings.bTrain),
tgtUpdateAlpha(settings.targetDelay), gamma(settings.gamma),
greedyEps(settings.greedyEps), epsAnneal(settings.epsAnneal),
taskCounter(batchSize), aInfo(env->aI), sInfo(env->sI),
gen(&settings.generators[0])
{
    assert(nThreads>0);
    for (Uint i=0; i<nThreads; i++) Vstats.push_back(new trainData());
    profiler = new Profiler();
    data = new Transitions(mastersComm, env, settings);
}

void Learner::clearFailedSim(const int agentOne, const int agentEnd)
{
  data->clearFailedSim(agentOne, agentEnd);
}

void Learner::pushBackEndedSim(const int agentOne, const int agentEnd)
{
  data->pushBackEndedSim(agentOne, agentEnd);
}

void Learner::TrainBatch()
{
    const Uint ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
    vector<Uint> seq(batchSize), samp(batchSize);
    if (ndata<batchSize) return; //do we have enough data?
    if (!bTrain) return; //are we training?
    Uint nAddedGradients=0;

    if(data->syncBoolOr(data->inds.size()<batchSize))
    { //uniform sampling
        data->updateSamples();
        processStats(Vstats, 0 ); //dump info about convergence
        #ifdef __CHECK_DIFF //check gradients with finite differences, just for debug
        if (stats.epochCount % 1000 == 0) net->checkGrads();
        #endif
    }

    if(bRecurrent) {
        nAddedGradients = sampleSequences(seq);
        for (Uint i=0; i<batchSize; i++)
          Train_BPTT(seq[i]);
    } else {
        nAddedGradients = sampleTransitions(seq, samp);
        for (Uint i=0; i<batchSize; i++)
          Train(seq[i], samp[i]);
    }

    dataUsage += 1./nAddedGradients;
    updateNNWeights(nAddedGradients);
    updateTargetNetwork();
}

void Learner::TrainTasking(Master* const master)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    vector<Uint> seq(batchSize), samp(batchSize);//, index(batchSize);
    Uint nAddedGradients = 0, countElapsed = 0;
    Real sumElapsed = 0;
    Uint ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
  	if (ndata <= 10*batchSize || !bTrain) {
      if(nAgents<1) die("Nothing to do, nowhere to go.\n");
      master->run();
    }

    while (true) {
		    ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
        taskCounter = 0;
        nAddedGradients = 0;

        if(data->syncBoolOr(data->inds.size()<batchSize))
        { //reset sampling
            data->updateSamples();
            processStats(Vstats, sumElapsed/countElapsed); //dump info about convergence
            sumElapsed = 0; countElapsed=0;
            //print_memory_usage();
            #ifdef __CHECK_DIFF //check gradients with finite differences, just for debug
            if (stats.epochCount % 1000 == 0) net->checkGrads();
            #endif
        }
        start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel num_threads(nThreads)
        #pragma omp master
      	{
      		if(bRecurrent) {//we are using an LSTM: do BPTT
            nAddedGradients = sampleSequences(seq);
      			#pragma omp flush

      			for (Uint i=0; i<batchSize; i++) {
              const Uint sequence = seq[i];
      				#pragma omp task firstprivate(sequence)
      				{
      					const int thrID = omp_get_thread_num();
                assert(thrID>=0);
                //#ifndef NDEBUG
                //printf("Thread %d to %d\n",thrID,sequence);
                //fflush(0);
                //#endif
      					Train_BPTT(sequence, static_cast<Uint>(thrID));

      					#pragma omp atomic
      					taskCounter++;
      				}
      			}
      		} else {
      			nAddedGradients = sampleTransitions(seq, samp);
      			#pragma omp flush

      			for (Uint i=0; i<batchSize; i++) {
              const Uint sequence = seq[i];
              const Uint transition = samp[i];
      				#pragma omp task firstprivate(sequence,transition)
      				{
      					const int thrID = omp_get_thread_num();
                assert(thrID>=0);
      					Train(sequence, transition, static_cast<Uint>(thrID));

      					#pragma omp atomic
      					taskCounter++;
      				}
      			}
      		}

          //TODO: can add task to update sampling probabilities for prioritized exp replay

          if(nAgents>0)
          master->run(); //master goes to communicate with slaves
        }

        end = std::chrono::high_resolution_clock::now();
        const Real len = std::chrono::duration<Real>(end-start).count();
        sumElapsed += len/nAddedGradients;
        countElapsed++;

         dataUsage += 1./nAddedGradients;
        //this needs to be compatible with multiple servers
      	stackAndUpdateNNWeights(nAddedGradients);
        // this can be handled node wise
      	updateTargetNetwork();
    }
}

Uint Learner::sampleTransitions(vector<Uint>& sequences, vector<Uint>& transitions)
{
  assert(sequences.size() == batchSize && transitions.size() == batchSize);
  assert(!bRecurrent);
  vector<Uint> load(batchSize), sorting(batchSize), s(batchSize), t(batchSize);
  for (Uint i=0; i<batchSize; i++)
  {
    const Uint ind = data->sample();

    Uint k=0, back=0, indT=data->Set[0]->tuples.size()-1;
    while (ind >= indT) {
      back = indT;
      indT += data->Set[++k]->tuples.size()-1;
    }

    s[i] = k;
    t[i] = ind-back;
    sorting[i] = i;
    //load[i] = data->Set[k]->tuples.size()-1 - t[i];
    load[i] = data->Set[k]->tuples.size()-1;
  }

  //sort elements of sorting according to load for each transition:
  const auto compare = [&] (Uint a, Uint b) { return load[a] > load[b]; };
  std::sort(sorting.begin(), sorting.end(), compare);
  assert(load[sorting[0]] > load[sorting[batchSize-1]]);
  //sort vectors passed to learning algo:
  for (Uint i=0; i<batchSize; i++) {
    transitions[i] = t[sorting[i]];
    sequences[i] = s[sorting[i]];
  }
  return batchSize; //always add one grad per transition
}

Uint Learner::sampleSequences(vector<Uint>& sequences)
{
  assert(sequences.size() == batchSize && bRecurrent);
  Uint nAddedGradients = 0;
  for (Uint i=0; i<batchSize; i++)
  {
    const Uint ind = data->sample();
    sequences[i]  = ind;
    //index[i] = ind;
    const Uint seqSize = data->Set[ind]->tuples.size();
    //to normalize mean gradient for update:
    nAddedGradients += seqSize-1; //last state = terminal, no next reward
  }
  //sort them such that longer ones are started first, reducing overhead!
  const auto compare = [this] (Uint a, Uint b) {
    return data->Set[a]->tuples.size() > data->Set[b]->tuples.size();
  };
  std::sort(sequences.begin(), sequences.end(), compare);

  return nAddedGradients;
}

void Learner::stackAndUpdateNNWeights(const Uint nAddedGradients)
{
    assert(bTrain);
    opt->nepoch++;
    //add up gradients across threads
    opt->stackGrads(net->grad, net->Vgrad);

    //add up gradients across nodes (masters)
    int nMasters;
    MPI_Comm_size(mastersComm, &nMasters);
    if (nMasters > 1) {
      MPI_Allreduce(MPI_IN_PLACE, net->grad->_W, net->getnWeights(),
                    MPI_VALUE_TYPE, MPI_SUM, mastersComm);
      MPI_Allreduce(MPI_IN_PLACE, net->grad->_B, net->getnBiases(),
                    MPI_VALUE_TYPE, MPI_SUM, mastersComm);
    }
    //update is deterministic: can be handled independently by each node
    //communication overhead is probably greater than a parallelised sum
    assert(nMasters>0);
    opt->update(net->grad, nAddedGradients*nMasters);
}

void Learner::updateNNWeights(const Uint nAddedGradients)
{
    assert(bTrain && nAddedGradients>0);
    //add up gradients across nodes (masters)
    int nMasters;
    MPI_Comm_size(mastersComm, &nMasters);
    if (nMasters > 1) {
      MPI_Allreduce(MPI_IN_PLACE, net->grad->_W, net->getnWeights(),
                    MPI_VALUE_TYPE, MPI_SUM, mastersComm);
      MPI_Allreduce(MPI_IN_PLACE, net->grad->_B, net->getnBiases(),
                    MPI_VALUE_TYPE, MPI_SUM, mastersComm);
    }

    opt->nepoch++;
    opt->update(net->grad, nAddedGradients);
}

void Learner::updateTargetNetwork()
{
    assert(bTrain);
    if (cntUpdateDelay == 0) { //DQN-style frozen weight
        cntUpdateDelay = tgtUpdateDelay;

        //2 options: either move tgt_wght = (1-a)*tgt_wght + a*wght
        //if (tgtUpdateDelay==0) net->moveFrozenWeights(tgtUpdateAlpha);
        //else net->updateFrozenWeights(); //or copy tgt_wghts = wghts
        opt->moveFrozenWeights(tgtUpdateAlpha);
    }
    if(cntUpdateDelay>0) cntUpdateDelay--;
}

bool Learner::checkBatch(unsigned long mastersNiter)
{
    const unsigned long dataNiter = bRecurrent ? data->nSeenSequences : mastersNiter;
    const Uint ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
    if (ndata<batchSize*10 || !bTrain) {
      mastersNiter_b4PolUpdates = dataNiter;
      return false;
    }  //do we have enough data? TODO k*ndata?

    //if we are using a cheap to simulate env, we want to prioritize networks
    //if optimizer has done less updates than master has done communications
    // ratio is 1 : 1 in DQN paper
    //then let master thread go to help other threads finish the batch
    //otherwise only go to communicate if batch is over
    const long unsigned learnerNiter = opt->nepoch + mastersNiter_b4PolUpdates;
    if (env->cheaperThanNetwork && dataNiter > learnerNiter)
      return true;

    //If the transition buffer is already backed up, train and pause communicating
    if(data->Buffered.size() >= data->maxTotSeqNum/20)
      return true;

    //Very lax constraint on over-using stale data too much
    if(dataUsage > mastersNiter) return false;

    return taskCounter >= batchSize;
}

void Learner::save(string name)
{
    int masterRank;
    MPI_Comm_rank(mastersComm, &masterRank);
    //    net->save(name);
    if (!masterRank) {
      opt->save(name);
      data->save(name);

      const string stuff = name+".status";
      FILE * f = fopen(stuff.c_str(), "w");
      if (f == NULL) die("Save fail\n");
      fprintf(f, "policy iter: %d\n", data->anneal);
      fprintf(f, "optimize epoch: %lu\n", opt->nepoch);
      fprintf(f, "epoch count: %d\n", stats.epochCount);
      fclose(f);
    }
}

void Learner::restart(string name)
{
  int masterRank;
  MPI_Comm_rank(mastersComm, &masterRank);
  printf("Restarting from saved policy...\n");

  data->restartSamples();
	if (name == "none") return;

  if ( opt->restart(name) )
      printf("Restart successful, moving on...\n")
  else
      printf("Not all policies restarted. \n")
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
      }{
        int val=-1;
        fscanf(f, "epoch count: %d\n", &val);
        if(val>=0) stats.epochCount = val;
        printf("epoch count: %d\n", stats.epochCount);
      }
      fclose(f);
  }
  else printf("No status\n");
  save("restarted_policy");
}

void Learner::dumpStats(trainData* const _stats, const Real& Q,
                        const Real& err, const vector<Real>& Qs) const
{
    const Real max_Q = *max_element(Qs.begin(), Qs.end());
    const Real min_Q = *min_element(Qs.begin(), Qs.end());
    _stats->MSE += err*err;
    _stats->relE += fabs(err)/(max_Q-min_Q);
    _stats->avgQ += Q;
    _stats->minQ = std::min(_stats->minQ,Q);
    _stats->maxQ = std::max(_stats->maxQ,Q);
    _stats->dumpCount++;
}

void Learner::processStats(vector<trainData*> _stats, const Real avgTime)
{
    stats.minQ= 1e5; stats.maxQ=-1e5; stats.MSE=0;
    stats.avgQ=0; stats.relE=0; stats.dumpCount=0;

    for (Uint i=0; i<_stats.size(); i++) {
        stats.MSE += _stats[i]->MSE;
        stats.relE += _stats[i]->relE;
        stats.avgQ += _stats[i]->avgQ;
        stats.dumpCount += _stats[i]->dumpCount;
        stats.minQ = std::min(stats.minQ,_stats[i]->minQ);
        stats.maxQ = std::max(stats.maxQ,_stats[i]->maxQ);
        _stats[i]->minQ= 1e5; _stats[i]->maxQ=-1e5; _stats[i]->MSE=0;
        _stats[i]->avgQ=0; _stats[i]->relE=0; _stats[i]->dumpCount=0;
    }

    if (stats.dumpCount<2) return;
    stats.epochCount++;


    Real sumWeights = 0, distTarget = 0, sumWeightsSq = 0;
    for (Uint w=0; w<net->getnWeights(); w++){
    	sumWeights += std::fabs(net->weights[w]);
      sumWeightsSq += net->weights[w]*net->weights[w];
      distTarget += std::pow(net->weights[w]-net->tgt_weights[w],2);
    }
    //sumWeights *= opt->lambda;

    //stats.MSE=std::sqrt(stats.MSE/stats.dumpCount);
    stats.MSE/=(stats.dumpCount-1);
    stats.avgQ/=stats.dumpCount;
    stats.relE/=stats.dumpCount;

    ofstream filestats;
    filestats.open("stats.txt", ios::app);
    //printf("epoch %d, avg_mse %f, avg_rel_err %f, avg_Q %f, "
    //        "min_Q %f, max_Q %f, errWeights [%f %f %f], N %d, steps %lu, dT %f\n",
    //      stats.epochCount, stats.MSE, stats.relE, stats.avgQ, stats.minQ,
    //      stats.maxQ, sumWeights, sumWeightsSq, distTarget, stats.dumpCount,
	 // opt->nepoch, avgTime);
    printf("%d (%lu), mse:%f, avg_Q:%f, min_Q:%f, max_Q:%f, errWeights [%f %f %f], dT %f\n",
          stats.epochCount, opt->nepoch, stats.MSE, stats.avgQ, stats.minQ,
          stats.maxQ, sumWeights, sumWeightsSq, distTarget, avgTime);
    filestats<<stats.epochCount<<"\t"<<stats.MSE<<"\t" <<stats.relE<<"\t"
             <<stats.avgQ<<"\t"<<stats.maxQ<<"\t"<<stats.minQ<<"\t"
             <<sumWeights<<"\t"<<sumWeightsSq<<"\t"<<distTarget<<"\t"
             <<stats.dumpCount<<"\t"<<opt->nepoch<<"\t"<<avgTime<<endl;
    filestats.close();

    fflush(0);
    if (stats.epochCount % 100==0) save("policy");
}

void Learner::dumpPolicy(const vector<Real> lower, const vector<Real>& upper,
                        const vector<Uint>& nbins)
{}

void Learner::buildNetwork(Network*& _net , Optimizer*& _opt,
  const vector<Uint> nouts, Settings & settings, const vector<Uint> addedInputs)
{
  const string netType = settings.nnType;
  const string funcType = settings.nnFunc;
  const vector<int> lsize = settings.readNetSettingsSize();
  assert(nouts.size()>0);

  Builder build(settings);
  //check if environment wants a particular network structure
  if (not env->predefinedNetwork(&build))
    build.addInput(nInputs);

  for (Uint i=0; i<addedInputs.size(); i++)
    build.addInput(addedInputs[i]);

  {
    //const int nsplit = std::min(static_cast<int>(lsize.size()),2);
    //const int nsplit = lsize.size()>3 ? 2 : 1;
    const Uint nsplit = 1;
    //const int nsplit = lsize.size();
    for (Uint i=0; i<lsize.size()-nsplit; i++)
      build.addLayer(lsize[i], netType, funcType);

    const Uint firstSplit = lsize.size()-nsplit;
    const vector<int> lastJointLayer(1, build.getLastLayerID());

    for (Uint i=0; i<nouts.size(); i++)
    {
      build.addLayer(lsize[firstSplit], netType, funcType, lastJointLayer);

      for (Uint j=firstSplit+1; j<lsize.size(); j++)
        build.addLayer(lsize[j], netType, funcType);

      build.addOutput(static_cast<int>(nouts[i]) , "Normal");
    }
  }
  _net = build.build();

  #ifndef __EntropySGD
    _opt = new AdamOptimizer(net, profiler, settings);
  #else
    _opt = new EntropySGD(net, profiler, settings);
  #endif
}
