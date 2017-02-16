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
Learner::Learner(MPI_Comm comm, Environment*const env, Settings & settings) :
mastersComm(comm), nAgents(settings.nAgents), batchSize(settings.dqnBatch),
tgtUpdateDelay((int)settings.dqnUpdateC), nThreads(settings.nThreads),
nInputs(settings.nnInputs), nOutputs(settings.nnOutputs),
bRecurrent(settings.nnType==1), bTrain(settings.bTrain==1),
tgtUpdateAlpha(settings.dqnUpdateC), gamma(settings.gamma),
greedyEps(settings.greedyEps), cntUpdateDelay(-1), taskCounter(batchSize),
aInfo(env->aI), sInfo(env->sI), gen(settings.gen)
{
    for (int i=0; i<max(nThreads,1); i++) Vstats.push_back(new trainData());
    profiler = new Profiler();
    data = new Transitions(mastersComm, env, settings);
}

void Learner::clearFailedSim(const int agentOne, const int agentEnd)
{
  data->clearFailedSim(agentOne, agentEnd);
}

void Learner::TrainBatch()
{
    const int ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
    if (ndata<batchSize) return; //do we have enough data?
    int nAddedGradients(0);

    if (data->inds.size()<batchSize)
    { //uniform sampling
        data->updateSamples();
        processStats(Vstats, 0 ); //dump info about convergence
    }

    if(bRecurrent) {

        for (int i=0; i<batchSize; i++)
        {
            const int ind = data->inds.back();
            data->inds.pop_back();
            const int seqSize = data->Set[ind]->tuples.size();
            nAddedGradients += seqSize-1;

            Train_BPTT(ind);
        }

    } else {

        nAddedGradients = batchSize;
        for (int i=0; i<batchSize; i++)
        {
            const int ind = data->inds.back();
            data->inds.pop_back();

            int k(0), back(0), indT(data->Set[0]->tuples.size()-1);
            while (ind >= indT) {
                back = indT;
                indT += data->Set[++k]->tuples.size()-1;
            }

            Train(k, ind-back);
        }

    }

    updateNNWeights(nAddedGradients);
    updateTargetNetwork();
}

void Learner::TrainTasking(Master* const master)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    vector<int> seq(batchSize), samp(batchSize), index(batchSize);
    int nAddedGradients(0);
    Real sumElapsed(0.);
    int countElapsed(0);
    int ndata = (bRecurrent) ? data->nSequences : data->nTransitions;

  	if (ndata <= batchSize) {
      if(nAgents<1) die("Nothing to do, nowhere to go.\n");
      master->hustle();
    }

    while (true) {
		    ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
        taskCounter=0;
        nAddedGradients = 0;

        if (data->inds.size()<batchSize) { //reset sampling
            data->updateSamples();
            processStats(Vstats, sumElapsed/countElapsed); //dump info about convergence
            sumElapsed = 0; countElapsed=0;
            //print_memory_usage();
            #if 1==0// ndef NDEBUG //check gradients with finite differences, just for debug  0==1//
            if (stats.epochCount % 100 == 0) {
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
				for (int i=0; i<batchSize; i++) {
					const int ind = data->inds.back();
					data->inds.pop_back();
					seq[i]  = ind;
					index[i] = ind;
					const int seqSize = data->Set[ind]->tuples.size();
					nAddedGradients += seqSize-1; //to normalize mean gradient for update
				}
				#pragma omp flush

				for (int i=0; i<batchSize; i++) {
					#pragma omp task firstprivate(i)
					{
						const int thrID = omp_get_thread_num();
						Train_BPTT(seq[i], thrID);

						#pragma omp atomic
						taskCounter++;

//                  printf("Thread %d performed task %d\n", thrID, taskCounter); fflush(0);
					}
				}
			} else {
				for (int i=0; i<batchSize; i++)  {
					const int ind = data->inds.back();
					data->inds.pop_back();
					int k(0), back(0), indT(data->Set[0]->tuples.size()-1);
					while (ind >= indT) {
						back = indT;
						indT += data->Set[++k]->tuples.size()-1;
					}
					seq[i]  = k;
					samp[i] = ind-back;
					index[i] = ind;
				}
				nAddedGradients = batchSize;
				#pragma omp flush

				for (int i=0; i<batchSize; i++) {
					#pragma omp task firstprivate(i)
					{
						const int thrID = omp_get_thread_num();
						Train(seq[i], samp[i], thrID);

						#pragma omp atomic
						taskCounter++;

//                  printf("Thread %d performed task %d\n", thrID, taskCounter); fflush(0);
					}
				}
			}

      //TODO: can add task to update sampling probabilities for prioritized exp replay

      if(nAgents>0)
      master->hustle(); //master goes to communicate with slaves
    }

    end = std::chrono::high_resolution_clock::now();
    const auto len = std::chrono::duration<Real>(end-start).count();
    sumElapsed += len/nAddedGradients;
    countElapsed++;

    //this needs to be compatible with multiple servers
		stackAndUpdateNNWeights(nAddedGradients);
    // this can be handled node wise
		updateTargetNetwork();
    }
}

void Learner::stackAndUpdateNNWeights(const int nAddedGradients)
{
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
    if (cntUpdateDelay <= 0) { //DQN-style frozen weight
        cntUpdateDelay = tgtUpdateDelay;

        //2 options: either move tgt_wght = (1-a)*tgt_wght + a*wght
        if (tgtUpdateDelay==0) net->moveFrozenWeights(tgtUpdateAlpha);
        else net->updateFrozenWeights(); //or copy tgt_wghts = wghts
    }
    cntUpdateDelay--;
}

bool Learner::checkBatch() const
{
    const int ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
    if (ndata<batchSize) return false; //do we have enough data?
    return taskCounter >= batchSize;
}

void Learner::save(string name)
{
    int masterRank;
    MPI_Comm_rank(mastersComm, &masterRank);
//    net->save(name);
    if (!masterRank) opt->save(name);
    data->save(name);
    const string stuff = name + ".status";
    FILE * f = fopen(stuff.c_str(), "w");
    if (f == NULL) die("Save fail\n");
    fprintf(f, "policy iter: %d\n", data->anneal);
    fprintf(f, "epoch count: %d\n", stats.epochCount);
    fclose(f);
}

void Learner::restart(string name)
{
    _info("Restarting from saved policy...\n");
    data->restartSamples();
    if ( opt->restart(name) )
        _info("Restart successful, moving on...\n")
    else
        _info("Not all policies restarted. \n")
    data->restart(name);
    save("restarted_policy.net");
    FILE * f = fopen("policy.status", "r");
    if(f != NULL) {
        int val;
        fscanf(f, "policy iter: %d\n", &val);
        if(val>=0) data->anneal = val;
        printf("policy iter: %d\n", data->anneal);
        val=-1;
        fscanf(f, "epoch count: %d\n", &val);
        if(val>=0) stats.epochCount = val;
        printf("epoch count: %d\n", stats.epochCount);
        fclose(f);
    }
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


    Real sumWeights(0.);
    for (int w=0; w<net->nWeights; w++){
    	sumWeights += std::fabs(net->weights[w]);
    }
    sumWeights *= opt->lambda;

    stats.MSE/=(stats.dumpCount-1);
    stats.avgQ/=stats.dumpCount;
    stats.relE/=stats.dumpCount;
    net->printRunning();
    net->resetRunning();

    ofstream filestats;
    filestats.open("stats.txt", ios::app);
    printf("epoch %d, avg_mse %f, avg_rel_err %f, avg_Q %f, "
           "min_Q %f, max_Q %f, errWeights %f, N %d, dT %f\n",
      	   stats.epochCount, stats.MSE, stats.relE, stats.avgQ,
           stats.minQ, stats.maxQ, sumWeights, stats.dumpCount, avgTime);
    filestats<<stats.epochCount<<"\t"<<stats.MSE<<"\t" <<stats.relE<<"\t"
             <<stats.avgQ<<"\t"<<stats.maxQ<<"\t"<<stats.minQ<<"\t"
             <<sumWeights<<"\t"<<stats.dumpCount<<"\t"<<avgTime<<endl;
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
