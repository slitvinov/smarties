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
Learner::Learner(Environment* env, Settings & settings) : nAgents(settings.nAgents),
batchSize(settings.dqnBatch), tgtUpdateDelay((int)settings.dqnUpdateC), nThreads(settings.nThreads),
nInputs(settings.nnInputs), nOutputs(settings.nnOutputs), bRecurrent(settings.nnType==1),
bTrain(settings.bTrain==1), tgtUpdateAlpha(settings.dqnUpdateC), gamma(settings.gamma),
greedyEps(settings.greedyEps), cntUpdateDelay(-1), aInfo(env->aI), sInfo(env->sI), gen(settings.gen), taskCounter(batchSize)
{
    for (int i=0; i<max(nThreads,1); i++) Vstats.push_back(new trainData());
    profiler = new Profiler();
    data = new Transitions(env, settings);
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
        
        for (int i(0); i<batchSize; i++)
        {
            const int ind = data->inds.back();
            data->inds.pop_back();
            const int seqSize = data->Set[ind]->tuples.size();
            nAddedGradients += seqSize-1;
            
            Train_BPTT(ind);
        }
        
    } else {
        
        nAddedGradients = batchSize;
        for (int i(0); i<batchSize; i++)
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
    int nAddedGradients(0), maxBufSize(0);
    Real sumElapsed(0.); 
    int countElapsed(0);
    int ndata = (bRecurrent) ? data->nSequences : data->nTransitions;

    while (true) {
		ndata = (bRecurrent) ? data->nSequences : data->nTransitions;

		if (ndata <= batchSize) master->hustle();
        
        taskCounter=0;
        nAddedGradients = 0;

        if (data->inds.size()<batchSize) { //reset sampling
            data->updateSamples();
            ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
            processStats(Vstats, sumElapsed/countElapsed); //dump info about convergence
            opt->nepoch=stats.epochCount; //used to anneal learning rate
             #ifdef _whitenTarget_
              net->applyBatchStatistics();
             #endif
            sumElapsed = 0; countElapsed=0;

            #if 1==0// ndef NDEBUG //check gradients with finite differences, just for debug  0==1//
            if (stats.epochCount++ % 100 == 0) {
                vector<vector<Real>> inputs;
                const int ind = data->Set.size()-1;
                for (int k=0; k<data->Set[ind]->tuples.size(); k++)
                    inputs.push_back(data->Set[ind]->tuples[k]->s);
                net->checkGrads(inputs, data->Set[ind]->tuples.size()-1);
            }
            #endif
        }
		

#pragma omp parallel num_threads(nThreads)
#pragma omp master
		{
			start = std::chrono::high_resolution_clock::now();

			if(bRecurrent) {//we are using an LSTM: do BPTT
				for (int i(0); i<batchSize; i++) {
					const int ind = data->inds.back();
					data->inds.pop_back();
					seq[i]  = ind;
					index[i] = ind;
					const int seqSize = data->Set[ind]->tuples.size();
					nAddedGradients += seqSize-1; //to normalize mean gradient for update
				}
				#pragma omp flush

				for (int i(0); i<batchSize; i++) {
					#pragma omp task firstprivate(i)
					{
						const int thrID = omp_get_thread_num();
						Train_BPTT(seq[i], thrID);

						#pragma omp atomic
						taskCounter++;
					}
				}
			} else {
				for (int i(0); i<batchSize; i++)  {
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

				for (int i(0); i<batchSize; i++) {
					#pragma omp task firstprivate(i)
					{
						const int thrID = omp_get_thread_num();
						Train(seq[i], samp[i], thrID);

						#pragma omp atomic
						taskCounter++;
					}
				}
			}

            //TODO: can add task to update sampling probabilities for prioritized exp replay
            #ifndef MEGADEBUG
            master->hustle(); //master goes to communicate with slaves
            #endif
            end = std::chrono::high_resolution_clock::now();
            sumElapsed += std::chrono::duration<Real>(end-start).count()/nAddedGradients;
            countElapsed++;
        }

        //here be omp fors:
        if (ndata>batchSize) {
        	stackAndUpdateNNWeights(nAddedGradients);

        	updateTargetNetwork();
        }
    }
}

void Learner::stackAndUpdateNNWeights(const int nAddedGradients)
{
    opt->stackGrads(net->grad, net->Vgrad); //add up gradients across threads (TODO: do not sum 0 component of Vgrad as now we have a pragma omp master above)
    opt->update(net->grad,nAddedGradients); //update
}

void Learner::updateNNWeights(const int nAddedGradients)
{
    opt->nepoch=stats.epochCount;  //used to anneal learning rate
    opt->update(net->grad,nAddedGradients);
}

void Learner::updateTargetNetwork()
{

    if (cntUpdateDelay <= 0) { //DQN-style frozen weight
        #pragma omp master
        cntUpdateDelay = tgtUpdateDelay;
        
        //2 options: either move tgt_wght = (1-a)*tgt_wght + a*wght
        if (tgtUpdateDelay==0) net->moveFrozenWeights(tgtUpdateAlpha);
        else net->updateFrozenWeights(); //or copy tgt_wghts = wghts
    }
    #pragma omp master
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
    //net->save(name + ".net");
    const string stuff = name + ".status";
    FILE * f = fopen(stuff.c_str(), "w");
    if (f == NULL) die("Save fail\n");
    fprintf(f, "policy iter: %d\n", data->anneal);
    fprintf(f, "epoch count: %d\n", stats.epochCount);
    fclose(f);
}

void Learner::restart(string name)
{
      return;
    _info("Restarting from saved policy...\n");
    data->restartSamples();
    if ( net->restart(name + ".net") ) {_info("Restart successful, moving on...\n");}
    else { _info("Not all policies restarted, therefore assumed zero. Moving on...\n");}
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

void Learner::dumpStats(trainData* const _stats, const Real& Q, const Real& err, const vector<Real>& Qs)
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
    printf("epoch %d, avg_mse %f, avg_rel_err %f, avg_Q %f, min_Q %f, max_Q %f, errWeights %f, N %d, dT %f\n",
	   stats.epochCount, stats.MSE, stats.relE, stats.avgQ, stats.minQ, stats.maxQ, sumWeights, stats.dumpCount, avgTime);
    filestats<<
    stats.epochCount<<" "<<stats.MSE<<" "<<stats.relE<<" "<<stats.avgQ<<" "<<stats.maxQ<<" "<<sumWeights<<" "<<stats.minQ<<endl;
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
