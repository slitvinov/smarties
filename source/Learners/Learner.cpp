/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

 #include <sys/time.h>
 #include <sys/resource.h>
 #include <unistd.h>
 #include <cstdlib>
#include <string.h>
#include <stdio.h>      /* printf, fgets */
#include <stdlib.h>     /* atol */
/*
 static void print_memory_usage()
 {
     char pidstatus[256];
     char *line;
     char *vmsize;
     char *vmpeak;
     char *vmrss;
     char *vmhwm;

     size_t len;

     FILE *f;
     static int times = 0;

     if (times % 100 != 0)
     {
         //return;
     }
     times++;

     sprintf(pidstatus, "/proc/%d/status", getpid());
     vmsize = NULL;
     vmpeak = NULL;
     vmrss = NULL;
     vmhwm = NULL;
     //line = malloc(128);
     line = static_cast<char*>(malloc(128));
     len = 128;

     f = fopen(pidstatus, "r");
     if (!f) return;

     // Read memory size data from /proc/pid/status
     while (!vmsize || !vmpeak || !vmrss || !vmhwm)
     {
         if (getline(&line, &len, f) == -1)
         {
             // Some of the information isn't there, die
             return;
         }

         // Find VmPeak
         if (!strncmp(line, "VmPeak:", 7))
         {
             vmpeak = strdup(&line[7]);
         }

         // Find VmSize
         else if (!strncmp(line, "VmSize:", 7))
         {
             vmsize = strdup(&line[7]);
         }


         // Find VmRSS
         else if (!strncmp(line, "VmRSS:", 6))
         {
             vmrss = strdup(&line[7]);
         }

         // Find VmHWM
         else if (!strncmp(line, "VmHWM:", 6))
         {
             vmhwm = strdup(&line[7]);
         }
     }
     free(line);

     fclose(f);


     // Get rid of " kB\n"
     len = strlen(vmsize);
     vmsize[len - 4] = 0;
     len = strlen(vmpeak);
     vmpeak[len - 4] = 0;
     len = strlen(vmrss);
     vmrss[len - 4] = 0;
     len = strlen(vmhwm);
     vmhwm[len - 4] = 0;

     // Output results to stderr

     #if 0
      VmPeak: Peak virtual memory usage
      VmSize: Current virtual memory usage
      VmLck:  Current mlocked memory
      VmHWM:  Peak resident set size
      VmRSS:  Resident set size
      VmData: Size of "data" segment
      VmStk:  Size of stack
      VmExe:  Size of "text" segment
      VmLib:  Shared library usage
      VmPTE:  Pagetable entries size
      VmSwap: Swap space used
      #endif

     long _vmsize, _vmpeak, _vmrss, _vmhwm;

     _vmsize = atol(vmsize);
     _vmpeak = atol(vmpeak);
     _vmrss = atol(vmrss);
     _vmhwm = atol(vmhwm);

     //      fprintf(stderr, "(PID=%d) VmSize:%s\tVmPeak:%s\tVmRSS:%s\tVmHWM:%s\n", getpid(), vmsize, vmpeak, vmrss, vmhwm);
     printf("(PID=%d) VmSize:%8.1fMB, VmPeak:%8.1fMB, VmRSS:%8.1fMB, VmHWM:%8.1fMB\n",
            getpid(), _vmsize/1024., _vmpeak/1024., _vmrss/1024., _vmhwm/1024.);

     free(vmpeak);
     free(vmsize);
     free(vmrss);
     free(vmhwm);

     // Success
     return;
 }
*/

#include "Learner.h"
#include <chrono>
Learner::Learner(MPI_Comm comm, Environment*const _env, Settings & settings) :
mastersComm(comm), env(_env), nAgents(settings.nAgents), batchSize(settings.dqnBatch),
tgtUpdateDelay((int)settings.dqnUpdateC), nThreads(settings.nThreads),
nInputs(settings.nnInputs), nOutputs(settings.nnOutputs), nAppended(settings.dqnAppendS),
bRecurrent(settings.nnType==1), bTrain(settings.bTrain==1),
tgtUpdateAlpha(settings.dqnUpdateC), gamma(settings.gamma), greedyEps(settings.greedyEps),
epsAnneal(settings.epsAnneal), cntUpdateDelay(-1), taskCounter(batchSize),
aInfo(env->aI), sInfo(env->sI), gen(&settings.generators[0]), mastersNiter_b4PolUpdates(0)
{
    for (int i=0; i<max(nThreads,1); i++) Vstats.push_back(new trainData());
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
    const int ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
    vector<int> seq(batchSize), samp(batchSize);
    if (ndata<batchSize) return; //do we have enough data?
    if (!bTrain) return; //are we training?
    int nAddedGradients=0;

    if(data->syncBoolOr(data->inds.size()<batchSize))
    { //uniform sampling
        data->updateSamples();
        processStats(Vstats, 0 ); //dump info about convergence
        #if 1==0 // ndef NDEBUG //check gradients with finite differences, just for debug
        if (stats.epochCount == 0) { //% 100
            vector<vector<Real>> inputs;
            const int ind = data->Set.size()-1;
            for (int k=0; k<data->Set[ind]->tuples.size(); k++)
                inputs.push_back(data->Set[ind]->tuples[k]->s);
            net->checkGrads(inputs, data->Set[ind]->tuples.size()-1);
        }
        #endif
    }

    if(bRecurrent) {
        nAddedGradients = sampleSequences(seq);
        for (int i=0; i<batchSize; i++)
          Train_BPTT(seq[i]);
    } else {
        nAddedGradients = sampleTransitions(seq, samp);
        for (int i=0; i<batchSize; i++)
          Train(seq[i], samp[i]);
    }

    updateNNWeights(nAddedGradients);
    updateTargetNetwork();
}

void Learner::TrainTasking(Master* const master)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    vector<int> seq(batchSize), samp(batchSize);//, index(batchSize);
    int nAddedGradients = 0, countElapsed = 0;
    Real sumElapsed = 0;
    int ndata = (bRecurrent) ? data->nSequences : data->nTransitions;

  	if (ndata <= batchSize || !bTrain) {
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
            #if 1==0 // ndef NDEBUG //check gradients with finite differences, just for debug
            if (stats.epochCount == 0) { //% 100
                vector<vector<Real>> inputs;
                const int ind = data->Set.size()-1;
                for (int k=0; k<data->Set[ind]->tuples.size(); k++)
                    inputs.push_back(data->Set[ind]->tuples[k]->s);
                net->checkGrads(inputs, data->Set[ind]->tuples.size()-1);
            }
            #endif
        }
        start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel num_threads(nThreads)
        #pragma omp master
      	{
      		if(bRecurrent) {//we are using an LSTM: do BPTT
            nAddedGradients = sampleSequences(seq);
      			#pragma omp flush

      			for (int i=0; i<batchSize; i++) {
              const int sequence = seq[i];
      				#pragma omp task firstprivate(sequence)
      				{
      					const int thrID = omp_get_thread_num();
      					Train_BPTT(sequence, thrID);

      					#pragma omp atomic
      					taskCounter++;
      				}
      			}
      		} else {
      			nAddedGradients = sampleTransitions(seq, samp);
      			#pragma omp flush

      			for (int i=0; i<batchSize; i++) {
              const int sequence = seq[i];
              const int transition = samp[i];
      				#pragma omp task firstprivate(sequence,transition)
      				{
      					const int thrID = omp_get_thread_num();
      					Train(sequence, transition, thrID);

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

        //this needs to be compatible with multiple servers
      	stackAndUpdateNNWeights(nAddedGradients);
        // this can be handled node wise
      	updateTargetNetwork();
    }
}

int Learner::sampleTransitions(vector<int>& sequences, vector<int>& transitions)
{
  assert(sequences.size() == batchSize && transitions.size() == batchSize);
  assert(!bRecurrent);
  for (int i=0; i<batchSize; i++)
  {
    const int ind = data->inds.back();
    data->inds.pop_back();

    int k=0, back=0, indT=data->Set[0]->tuples.size()-1;
    while (ind >= indT) {
      back = indT;
      indT += data->Set[++k]->tuples.size()-1;
    }

    sequences[i] = k;
    transitions[i] = ind-back;
    //index[i] = ind;
  }

  return batchSize; //always add one grad per transition
}

int Learner::sampleSequences(vector<int>& sequences)
{
  assert(sequences.size() == batchSize && bRecurrent);
  int nAddedGradients = 0;
  for (int i=0; i<batchSize; i++) {
    const int ind = data->inds.back();
    data->inds.pop_back();
    sequences[i]  = ind;
    //index[i] = ind;
    const int seqSize = data->Set[ind]->tuples.size();
    //to normalize mean gradient for update:
    nAddedGradients += seqSize-1; //last state = terminal, no next reward
  }
  //sort them such that longer ones are started first, reducing overhead!
  const auto compare = [this] (int a, int b) {
    return data->Set[a]->tuples.size() > data->Set[b]->tuples.size();
  };
  std::sort(sequences.begin(), sequences.end(), compare);

  return nAddedGradients;
}

void Learner::stackAndUpdateNNWeights(const int nAddedGradients)
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

    net->updateWhiten(nAddedGradients*nMasters);
    //update is deterministic: can be handled independently by each node
    //communication overhead is probably greater than a parallelised sum
    assert(nMasters>0);
    opt->update(net->grad, nAddedGradients*nMasters);
}

void Learner::updateNNWeights(const int nAddedGradients)
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
    if (cntUpdateDelay <= 0) { //DQN-style frozen weight
        cntUpdateDelay = tgtUpdateDelay;

        //2 options: either move tgt_wght = (1-a)*tgt_wght + a*wght
        if (tgtUpdateDelay==0) net->moveFrozenWeights(tgtUpdateAlpha);
        else net->updateFrozenWeights(); //or copy tgt_wghts = wghts
    }
    cntUpdateDelay--;
}

bool Learner::checkBatch(unsigned long mastersNiter)
{
    const int ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
    if (ndata<batchSize*5) {
      mastersNiter_b4PolUpdates = mastersNiter;
      return false;
    }  //do we have enough data? TODO k*ndata?

    //if we are using a cheap to simulate env, we want to prioritize networks
    //if optimizer has done less updates than master has done communications
    // ratio is 1 : 1 in DQN paper
    //then let master thread go to help other threads finish the batch
    //otherwise only go to communicate if batch is over
    if (env->cheaperThanNetwork && bTrain &&
	mastersNiter > opt->nepoch + mastersNiter_b4PolUpdates)
      return true;
    else
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
  _info("Restarting from saved policy...\n");

  data->restartSamples();
  if ( opt->restart(name) )
      _info("Restart successful, moving on...\n")
  else
      _info("Not all policies restarted. \n")
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

    for (int i=0; i<_stats.size(); i++) {
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
    for (int w=0; w<net->nWeights; w++){
    	sumWeights += std::fabs(net->weights[w]);
      sumWeightsSq += net->weights[w]*net->weights[w];
      distTarget += std::pow(net->weights[w]-net->tgt_weights[w],2);
    }
    //sumWeights *= opt->lambda;

    stats.MSE/=(stats.dumpCount-1);
    stats.avgQ/=stats.dumpCount;
    stats.relE/=stats.dumpCount;
    net->printRunning();
    net->resetRunning();

    ofstream filestats;
    filestats.open("stats.txt", ios::app);
    printf("epoch %d, avg_mse %f, avg_rel_err %f, avg_Q %f, "
            "min_Q %f, max_Q %f, errWeights [%f %f %f], N %d, steps %d, dT %f\n",
          stats.epochCount, stats.MSE, stats.relE, stats.avgQ, stats.minQ,
          stats.maxQ, sumWeights, sumWeightsSq, distTarget, stats.dumpCount,
	  opt->nepoch, avgTime);
    filestats<<stats.epochCount<<"\t"<<stats.MSE<<"\t" <<stats.relE<<"\t"
             <<stats.avgQ<<"\t"<<stats.maxQ<<"\t"<<stats.minQ<<"\t"
             <<sumWeights<<"\t"<<sumWeightsSq<<"\t"<<distTarget<<"\t"
             <<stats.dumpCount<<"\t"<<opt->nepoch<<"\t"<<avgTime<<endl;
    filestats.close();

    fflush(0);
    if (stats.epochCount % 100==0) save("policy");
}

/*
void Learner::dumpStats(const Real& Q, const Real& err, const vector<Real>& Qs)
{
    //ostringstream o;
    //o << "[";
    //for (int i=0; i<Qs.size(); i++) o << Qs[i] << " ";
    //o << "]";
    //printf("Process %f - %f : %s\n", tgt, Q, string(o.str()).c_str());

    const Real max_Q = *max_element(Qs.begin(), Qs.end());
    const Real min_Q = *min_element(Qs.begin(), Qs.end());
    stats.MSE  += err*err;
    stats.relE += fabs(err)/(max_Q-min_Q);
    stats.avgQ += Q;
    stats.minQ = std::min(stats.minQ,Q);
    stats.maxQ = std::max(stats.maxQ,Q);
    stats.dumpCount++;

    if (data->nTransitions==stats.dumpCount && data->nTransitions>1) {
        stats.MSE /=(stats.dumpCount-1);
        stats.avgQ/=stats.dumpCount;
        stats.relE/=stats.dumpCount;

        ofstream filestats;
        filestats.open("stats.txt", ios::app);
        printf("epoch %d, avg_mse %f, avg_rel_err %f, avg_Q %f, min_Q %f, max_Q %f, N %d\n",
               stats.epochCount,      stats.MSE,      stats.relE,      stats.avgQ,      stats.minQ,      stats.maxQ, stats.dumpCount);
        filestats<<
               stats.epochCount<<" "<<stats.MSE<<" "<<stats.relE<<" "<<stats.avgQ<<" "<<stats.maxQ<<" "<<stats.minQ<<endl;
        filestats.close();

        stats.dumpCount = 0;
        stats.epochCount++;
        data->anneal++;
        if (stats.epochCount % 100==0) save("policy");

        stats.minQ=1e5; stats.maxQ=-1e5; stats.MSE=0; stats.avgQ=0; stats.relE=0;
    }
}
*/
