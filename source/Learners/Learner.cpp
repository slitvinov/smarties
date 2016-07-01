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
    
    profiler = new Profiler();
    net = new Network(lsize, bRecurrent, settings);
    opt = new AdamOptimizer(net, profiler, settings);
    T = new Transitions(env, settings);
    flags.resize(batchSize, false);
}

bool Learner::checkBatch()
{
    const int ndata = (bRecurrent) ? T->nSequences : T->nTransitions;
    if (ndata<batchSize) return false; //do we have enough data?
    
    bool done(true);
    for (int i=0; i<batchSize; i++)
        done = done && flags[i];
    
    if (done) //reset flags
        for (int i(0); i<batchSize; i++)
            flags[i] = false;
    
    return done;
}

void Learner::TrainBatch()
{
    const int ndata = (bRecurrent) ? T->nSequences : T->nTransitions;
    if (ndata<batchSize) return; //do we have enough data?
    if (cntUpdateDelay <= 0) {
        cntUpdateDelay = tgtUpdateDelay;
        if (tgtUpdateDelay==0) net->moveFrozenWeights(tgtUpdateAlpha);
        else net->updateFrozenWeights();
    }
    cntUpdateDelay--;
    if (T->inds.size()<batchSize) {
        T->inds.resize(ndata);
        std::iota(T->inds.begin(), T->inds.end(), 0);
        random_shuffle(T->inds.begin(), T->inds.end(),*(T->gen));
    }
    /*{
        vector<vector<Real>> inputs;
        const int ind = 1011;
        for (int k=0; k<T->Set[ind]->tuples.size(); k++)
        inputs.push_back(T->Set[ind]->tuples[k]->s);
        
        opt->checkGrads(inputs, T->Set[ind]->tuples.size(), 1);
    }*/
    if(bRecurrent) {
        vector<int> seq(batchSize);
        for (int i(0); i<batchSize; i++) {
            const int ind = T->inds.back();
            T->inds.pop_back();
            seq[i]  = ind;
        }
        Train(seq);
    } else {
        vector<int> seq(batchSize), samp(batchSize);
        for (int i(0); i<batchSize; i++) {
            const int ind = T->inds.back();
            T->inds.pop_back();
            int k(0), back(0), indT(T->Set[0]->tuples.size()-1);
            while (ind>=indT) {
                back=indT;
                indT+=T->Set[++k]->tuples.size()-1;
            }
            seq[i]  = k;
            samp[i] = ind-back;
        }
        Train(seq, samp);
    }
}

void Learner::TrainTasking(Master* const master)
{
    vector<int> seq(batchSize), samp(batchSize), first(batchSize), last(batchSize);
    #pragma omp parallel num_threads(nThreads)
    while (true) {
        const int ndata =(bRecurrent)?T->nSequences:T->nTransitions;
        #pragma omp single
        {
            if (ndata>batchSize) {
                if (T->inds.size()<batchSize) {
                    T->inds.resize(ndata);
                    std::iota(T->inds.begin(), T->inds.end(), 0);
                    random_shuffle(T->inds.begin(),T->inds.end(),*(T->gen));
                    stats.epochCount++;
                    if (stats.epochCount % 100==0) {
                        save("policy");
                        const string stuff = "policy.status";
                        FILE * f = fopen(stuff.c_str(), "w");
                        if (f == NULL) die("Save fail\n");
                        fprintf(f, "policy iter: %d\n", T->anneal);
                        fprintf(f, "epoch count: %d\n", stats.epochCount);
                        fclose(f);
                    }
                }
                
                if(bRecurrent) {
                    for (int i(0); i<batchSize; i++) {
                        first[i] = (i==0) ? 2 : last[i-1]+1; //0 and 1 reserved
                        const int ind = T->inds.back();
                        T->inds.pop_back();
                        seq[i]  = ind;
                        //LSTM NFQ requires size()+1 activations of the net:
                        last[i] = first[i] + T->Set[ind]->tuples.size();
                    }
                    net->allocateSeries(last.back());
                    for (int i(0); i<batchSize; i++) {
                        #pragma omp task firstprivate(i)
                        {
                            Train(seq[i], first[i]);
                            flags[i] = true;
                        }
                    }
                } else {
                    for (int i(0); i<batchSize; i++) {
                        first[i] = (i==0) ? 2 : last[i-1]+1; //0 and 1 reserved
                        const int ind = T->inds.back();
                        T->inds.pop_back();
                        int k(0), back(0), indT(T->Set[0]->tuples.size()-1);
                        while (ind>=indT) {
                            back=indT;
                            indT+=T->Set[++k]->tuples.size()-1;
                        }
                        seq[i]  = k;
                        samp[i] = ind-back;
                        //FFNN NFQ requires 3 activations of the net:
                        last[i] = first[i] + 2;
                    }
                    net->allocateSeries(last.back());
                    for (int i(0); i<batchSize; i++) {
                        #pragma omp task firstprivate(i)
                        {
                            Train(seq[i], samp[i], first[i]);
                            flags[i] = true;
                        }
                    }
                }
            }
            
            #ifndef MEGADEBUG
            master->hustle();
            #endif
        }
        if (ndata>batchSize) {
            opt->stackGrads(net->grad,net->Vgrad);
            opt->update(net->grad, last.back()-2);
        }
        
        if (cntUpdateDelay <= 0) {
            cntUpdateDelay = tgtUpdateDelay;
            if (tgtUpdateDelay==0) net->moveFrozenWeights(tgtUpdateAlpha);
            else net->updateFrozenWeights();
        }
        cntUpdateDelay--;
    }
}

void Learner::save(string name)
{
    net->save(name + ".net");
}

void Learner::restart(string name)
{
    _info("Restarting from saved policy...\n");
    
    T->restartSamples();
    if ( net->restart(name + ".net") ) {_info("Restart successful, moving on...\n");}
    else { _info("Not all policies restarted, therefore assumed zero. Moving on...\n");}
    save("restarted_policy.net");
    FILE * f = fopen("policy.status", "r");
    if(f != NULL) {
        int val;
        fscanf(f, "policy iter: %d\n", &val);
        if(val>=0) T->anneal = val;
        printf("policy iter: %d\n", T->anneal);
        
        val=-1;
        fscanf(f, "epoch count: %d\n", &val);
        if(val>=0) stats.epochCount = val;
        printf("epoch count: %d\n", stats.epochCount);
        
        fclose(f);
    }
}

void Learner::dumpStats(const Real tgt, const Real err, const Real Q)
{
    stats.MSE += err*err;
    stats.relE += fabs(err);///(max_Q-min_Q);
    stats.avgQ += tgt;
    stats.minQ = std::min(stats.minQ,tgt);
    stats.maxQ = std::max(stats.maxQ,tgt);
    stats.dumpCount++;
    
    if (T->nTransitions==stats.dumpCount && T->nTransitions>1) {
        stats.dumpCount = 0;
        stats.epochCount++;
        T->anneal++;
        
        const Real mean_err = stats.MSE /(T->nTransitions-1);
        const Real mean_Q   = stats.avgQ/T->nTransitions;
        const Real mean_rel = stats.relE/T->nTransitions;
        printf("Avg MSE %f, avg Q %f, min Q %f, max Q %f, relE %f, N %d\n",
               mean_err, mean_Q, stats.minQ, stats.maxQ, mean_rel, T->nTransitions);
        ofstream filestats;
        filestats.open("stats.txt", ios::app);
        filestats<<stats.epochCount<<" "<<mean_err<<" "<<mean_Q<<" "<<stats.maxQ<<" "<<stats.minQ<<" "<<mean_rel<<endl;
        filestats.close();
        
        if (stats.epochCount % 100==0) {
            /*
             string restart_file;
             char buf[500];
             sprintf(buf, "restart.net_%09d", T->anneal);
             restart_file = string(buf);
             Q->save(restart_file.c_str());
             */
            save("policy");
            const string stuff = "policy.status";
            FILE * f = fopen(stuff.c_str(), "w");
            if (f == NULL) die("Save fail\n");
            fprintf(f, "policy iter: %d\n", T->anneal);
            fprintf(f, "epoch count: %d\n", stats.epochCount);
            fclose(f);
        }
        
        //cout << profiler->printStat() << endl;
        stats.minQ=1e5; stats.maxQ=-1e5; stats.MSE=0; stats.avgQ=0; stats.relE=0;
    }
}