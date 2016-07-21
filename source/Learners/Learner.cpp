/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Learner.h"

Learner::Learner(Environment* env, Settings & settings) : nAgents(settings.nAgents),
batchSize(settings.dqnBatch), tgtUpdateDelay((int)settings.dqnUpdateC), nThreads(settings.nThreads),
nInputs(settings.nnInputs), nOutputs(settings.nnOutputs), bRecurrent(settings.nnType==1),
bTrain(settings.bTrain==1), tgtUpdateAlpha(settings.dqnUpdateC), gamma(settings.gamma),
greedyEps(settings.greedyEps), cntUpdateDelay(-1), aInfo(env->aI), sInfo(env->sI), gen(settings.gen)
{
    vector<int> lsize;
    lsize.push_back(nInputs);
    lsize.push_back(settings.nnLayer1);
    if (settings.nnLayer2>1) {
        lsize.push_back(settings.nnLayer2);
        if (settings.nnLayer3>1) {
            lsize.push_back(settings.nnLayer3);
            if (settings.nnLayer4>1) {
                lsize.push_back(settings.nnLayer4);
                if (settings.nnLayer5>1) {
                    lsize.push_back(settings.nnLayer5);
                }
            }
        }
    }
    lsize.push_back(nOutputs);
    
    if (nThreads>1) for (int i=0; i<nThreads; i++) Vstats.push_back(new trainData());

    profiler = new Profiler();
    net = new Network(lsize, bRecurrent, settings);
    opt = new AdamOptimizer(net, profiler, settings);
    data = new Transitions(env, settings);
    flags.resize(batchSize, true);
}

bool Learner::checkBatch() const
{
    const int ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
    if (ndata<batchSize) return false; //do we have enough data?
    
    //if no learning has been done yet: flags initialized as true
    //if learning begun, master sets flags to false and each thread sets flag to true as batch is processed
    //bool done(true);
    //for (int i=0; i<batchSize; i++) done = done && flags[i];
    
    return taskCounter>=batchSize;
}

void Learner::TrainBatch()
{
    const int ndata = (bRecurrent) ? data->nSequences : data->nTransitions;
    if (ndata<batchSize) return; //do we have enough data?
    int nAddedGradients(0);
    
    if (cntUpdateDelay <= 0) { //DQN-style frozen weight
        cntUpdateDelay = tgtUpdateDelay;
        if (tgtUpdateDelay==0) net->moveFrozenWeights(tgtUpdateAlpha);
        else net->updateFrozenWeights();
    }
    cntUpdateDelay--;
    
    if (data->inds.size()<batchSize) { //uniform sampling
        data->inds.resize(ndata);
        std::iota(data->inds.begin(), data->inds.end(), 0);
        random_shuffle(data->inds.begin(), data->inds.end(),*(data->gen));
    }
    
    /*
     vector<vector<Real>> inputs;
     const int ind = 1011;
     for (int k=0; k<T->Set[ind]->tuples.size(); k++)
        inputs.push_back(T->Set[ind]->tuples[k]->s);
     net->checkGrads(inputs, T->Set[ind]->tuples.size()-1, 1);
     */
    
    if(bRecurrent) {
        
        for (int i(0); i<batchSize; i++)
        {
            const int ind = data->inds.back();
            data->inds.pop_back();
            const int seqSize = data->Set[ind]->tuples.size();
            net->allocateSeries(seqSize);
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
    
    opt->nepoch=stats.epochCount;
    opt->update(net->grad,nAddedGradients);
}


void Learner::TrainTasking(Master* const master)
{
    vector<int> seq(batchSize), samp(batchSize);
    int nAddedGradients(0), maxBufSize(0);
    
    #pragma omp parallel num_threads(nThreads)
    while (true)
    {
        const int ndata =(bRecurrent) ? data->nSequences : data->nTransitions;
        
        #pragma omp master
        {
            if (ndata>batchSize)
            {
                taskCounter=0;
                nAddedGradients = 0;
                
                if (data->inds.size()<batchSize) //reset sampling
                {
                    data->inds.resize(ndata);
                    std::iota(data->inds.begin(), data->inds.end(), 0);
                    random_shuffle(data->inds.begin(), data->inds.end(), *(data->gen));
                    processStats(Vstats); //dump info about convergence
                    opt->nepoch=stats.epochCount; //used to anneal learning rate
                }
                
                if(bRecurrent)
                {
                    for (int i(0); i<batchSize; i++)
                    {
                        const int ind = data->inds.back();
                        data->inds.pop_back();
                        seq[i]  = ind;
                        const int seqSize = data->Set[ind]->tuples.size();
                        nAddedGradients += seqSize-1; //to normalize mean gradient for update
                        maxBufSize = max(maxBufSize,seqSize); //allocate network activations
                    }
                    
                    //LSTM NFQ requires size()+1 activations of the net:
                    net->allocateSeries(2+nThreads*(maxBufSize+1)); //0 and 1 reserved
                    
                    #pragma omp flush
                    
                    for (int i(0); i<batchSize; i++) {
                        const int knd = seq[i];
                        #pragma omp task firstprivate(i) firstprivate(knd) shared(maxBufSize)
                        {
                            const int thrID = omp_get_thread_num();
                            const int first = 2+(maxBufSize+1)*thrID;
                            Train_BPTT(knd, first, thrID);
                            
                            #pragma omp atomic
                            taskCounter++;
                        }
                    }
                }
                else
                {
                    for (int i(0); i<batchSize; i++)
                    {
                        const int ind = data->inds.back();
                        data->inds.pop_back();
                        int k(0), back(0), indT(data->Set[0]->tuples.size()-1);
                        while (ind >= indT) {
                            back = indT;
                            indT += data->Set[++k]->tuples.size()-1;
                        }
                        seq[i]  = k;
                        samp[i] = ind-back;
                    }
                    
                    nAddedGradients = batchSize;
                    net->allocateSeries(2+nThreads*3);
                    
                    #pragma omp flush
                    
                    for (int i(0); i<batchSize; i++) {
                        #pragma omp task firstprivate(i)
                        {
                            const int thrID = omp_get_thread_num();
                            const int first = 2+3*thrID;
                            Train(seq[i], samp[i], first, thrID);
                            
                            #pragma omp atomic
                            taskCounter++;
                        }
                    }
                }
            }
            //TODO: can add task to update sampling probabilities for prioritized exp replay
            #ifndef MEGADEBUG
            master->hustle(); //master goes to communicate with slaves
            #endif
        }
        #pragma omp barrier
        
        //here be omp fors:
        if (ndata>batchSize) {
            opt->stackGrads(net->grad, net->Vgrad); //add up gradients across threads (TODO: do not sum 0 component of Vgrad as now we have a pragma omp master above)
            opt->update(net->grad,nAddedGradients); //update
        }
        
        if (cntUpdateDelay <= 0) {
            #pragma omp master
            cntUpdateDelay = tgtUpdateDelay;
            
            if (tgtUpdateDelay==0) net->moveFrozenWeights(tgtUpdateAlpha);
            else net->updateFrozenWeights();
        }
        #pragma omp master
        cntUpdateDelay--;
    }
}

void Learner::save(string name)
{
    net->save(name + ".net");
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

void Learner::dumpStats(const Real& Q, const Real& err, const vector<Real>& Qs)
{
    /*
    ostringstream o;
    o << "[";
    for (int i=0; i<Qs.size(); i++) o << Qs[i] << " ";
    o << "]";
    printf("Process %f - %f : %s\n", tgt, Q, string(o.str()).c_str());
    */
    const Real max_Q = *max_element(Qs.begin(), Qs.end());
    const Real min_Q = *min_element(Qs.begin(), Qs.end());
    stats.MSE  += err*err;
    stats.relE += fabs(err)/(max_Q-min_Q);
    stats.avgQ += Q;
    stats.minQ = std::min(stats.minQ,Q);
    stats.maxQ = std::max(stats.maxQ,Q);
    stats.dumpCount++;
    
    if (data->nTransitions==stats.dumpCount && data->nTransitions>1) {
        const Real mean_err = stats.MSE /(stats.dumpCount-1);
        const Real mean_Q   = stats.avgQ/stats.dumpCount;
        const Real mean_rel = stats.relE/stats.dumpCount;
        
        ofstream filestats;
        filestats.open("stats.txt", ios::app);
        printf("epoch %d, avg_mse %f, avg_rel_err %f, avg_Q %f, min_Q %f, max_Q %f, N %d\n",
               stats.epochCount, mean_err, mean_rel, mean_Q, stats.minQ, stats.maxQ, stats.dumpCount);
        filestats<<stats.epochCount<<" "<<mean_err<<" "<<mean_rel<<" "<<mean_Q<<" "<<stats.maxQ<<" "<<stats.minQ<<endl;
        filestats.close();
        
        stats.dumpCount = 0;
        stats.epochCount++;
        data->anneal++;
        if (stats.epochCount % 100==0) save("policy");
        
        stats.minQ=1e5; stats.maxQ=-1e5; stats.MSE=0; stats.avgQ=0; stats.relE=0;
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

void Learner::processStats(vector<trainData*> _stats)
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
    
    const Real mean_err = stats.MSE/(stats.dumpCount-1);
    const Real mean_Q   = stats.avgQ/stats.dumpCount;
    const Real mean_rel = stats.relE/stats.dumpCount;
    
    ofstream filestats;
    filestats.open("stats.txt", ios::app);
    printf("epoch %d, avg_mse %f, avg_rel_err %f, avg_Q %f, min_Q %f, max_Q %f, N %d\n",
           stats.epochCount, mean_err, mean_rel, mean_Q, stats.minQ, stats.maxQ, stats.dumpCount);
    filestats<<stats.epochCount<<" "<<mean_err<<" "<<mean_rel<<" "<<mean_Q<<" "<<stats.maxQ<<" "<<stats.minQ<<endl;
    filestats.close();
    fflush(0);
    if (stats.epochCount % 100==0) save("policy");
}