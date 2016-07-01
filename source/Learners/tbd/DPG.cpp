/*
 *  NFQ.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "NFQ.h"
#include <map>

NFQ::NFQ(Environment* env, Settings & settings) :
Learner(env,settings), bTRAINING(settings.bTrain==1), batchSize(settings.dqnBatch), tgtUpdateDelay(settings.dqnUpdateC), cntUpdateDelay(-1), nActions(1)
{
    for (int i(0); i<actInfo.dim; i++) nActions*=actInfo.bounds[i];
    nStateDims = sInfo.dimUsed;
    
    //changed, now super bad...
    vector<int> lsize;
    lsize.push_back(nStateDims);
    //changed, now super bad...
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
    lsize.push_back(nActions);
    
    profiler = new Profiler();
    net = new Network(lsize, bRecurrent, settings);
    opt = new AdamOptimizer(net, profiler, settings);
    
    scaledInp.resize(nStateDims);
    prediction.resize(nActions);
}

void NFQ::select(const int agentId, State& s, Action& a, State& sOld, Action& aOld, vector<Real> info, Real r)
{   // No learning here!
    //       aOld, r
    // sOld ---------> s
    //
    // Find V(s) = max Q(s, a')
    //              a'
    Real newEps(greedyEps);
    if (bTRAINING) newEps = (.1 +greedyEps*exp(-T->Set.size()/1e3))*agentId/Real(agentId+1);
    
    s.scaleUsed(scaledInp);
    net->expandMemory(net->mem[iAgent], net->series[0]); //used by RNN to update recurrent signals
    net->predict(scaledInp, prediction, net->series[0], net->series[1]);
#ifdef _dumpNet_
    net->dump(iAgent);
#endif
    net->expandMemory(net->mem[iAgent], net->series[1]);
    
    Real val(-1e6)
    int i;
    for (i=0; i<prediction.size(); ++i)
        if (prediction[i]>Val) Val = prediction[i];
    a.unpack(i);
    
    uniform_real_distribution<Real> dis(0.,1.);
    if(dis(*gen) < newEps)  {
        a.getRand();
        printf("Random action\n");
    }
}

void NFQ::Train()
{
    const int ndata = (Q->bRecurrent) ? T->nSequences : T->nTransitions;
    if (ndata<100) return; //do we have enough data?
    
    if (cntUpdateDelay <= 0) {
        cntUpdateDelay = tgtUpdateDelay; //
        Q->updateFrozenWeights();
        
        if (T->avgQ.size()>0) {
            const Real mean_err = accumulate(T->Errs.begin(), T->Errs.end(), 0.)/tgtUpdateDelay;
            const Real mean_Q   = accumulate(T->avgQ.begin(), T->avgQ.end(), 0.)/tgtUpdateDelay;
            const Real mean_rel = accumulate(T->relE.begin(), T->relE.end(), 0.)/tgtUpdateDelay;
            const Real max_Q = *max_element(T->maxQ.begin(), T->maxQ.end());
            const Real min_Q = *min_element(T->minQ.begin(), T->minQ.end());
            printf("Avg MSE %f, avg Q %f, min Q %f, max Q %f, relE %f, N %d\n",
                   mean_err, mean_Q, min_Q, max_Q, mean_rel, ndata);
            ofstream filestats;
            filestats.open("stats.txt", ios::app);
            filestats<<T->anneal<<" "<<mean_err<<" "<<mean_Q<<" "<<max_Q<<" "<<min_Q<<" "<<mean_rel<<endl;
            filestats.close();
            
            if (T->anneal%100==0) {
                /*
                string restart_file;
                char buf[500];
                sprintf(buf, "restart.net_%09d", T->anneal);
                restart_file = string(buf);
                Q->save(restart_file.c_str());
                 */
                Q->save("policy");
                const string stuff = "policy.status";
                FILE * f = fopen(stuff.c_str(), "w");
                if (f != NULL) fprintf(f, "policy iter: %d\n", T->anneal);
                fclose(f);
            }
        } else { //first round
            T->Errs.resize(tgtUpdateDelay);
            T->avgQ.resize(tgtUpdateDelay);
            T->minQ.resize(tgtUpdateDelay);
            T->maxQ.resize(tgtUpdateDelay);
            T->relE.resize(tgtUpdateDelay);
        }
        
    #ifdef _Priority_
        T->updateP();
    #else
        T->anneal++;
    #endif
    }
    cntUpdateDelay--;
    
#ifndef _Priority_
    if (T->inds.size()<batchSize) {
        T->inds.resize(ndata);
        std::iota(T->inds.begin(), T->inds.end(), 0);
        random_shuffle(T->inds.begin(), T->inds.end(),*(T->gen));
    }
#endif
    
    if(Q->bRecurrent) {
        vector<int> seq(batchSize);
        for (int i(0); i<batchSize; i++) {
            #ifdef _Priority_
            const int ind = T->sample();
            Q->stats.weight=T->Ws[ind];
            #else
            const int ind = T->inds.back();
            T->inds.pop_back();
            #endif
            seq[i]  = ind;
        }
        Q->Train(seq);
    } else {
        //vector<vector<Real>*> sOlds(batchSize);
        //vector<Tuple*> samples(batchSize);
        //vector<bool> term(batchSize);
        vector<int> seq(batchSize), samp(batchSize);
        for (int i(0); i<batchSize; i++) {
            const int ind = T->inds.back();
            T->inds.pop_back();
            int k(0), back(0);
            while (ind>=T->indTransitions[k]) {
                back=T->indTransitions[k];
                k++;
            }
            seq[i]  = k;
            samp[i] = ind-back;
            //samples[i] = Set[seq].tuples[ind-back+1];
            //sOlds[i] = &(Set[seq].tuples[ind-back]->s);
            //term[i]= ind==indTransitions[seq]-1 && Set[seq].ended;
        }
        Q->Train(seq, samp);
    }
    
    T->Errs[cntUpdateDelay] = Q->stats.MSE;
    T->avgQ[cntUpdateDelay] = Q->stats.avgQ;
    T->minQ[cntUpdateDelay] = Q->stats.minQ;
    T->maxQ[cntUpdateDelay] = Q->stats.maxQ;
    T->relE[cntUpdateDelay] = Q->stats.relE;
}

void NFQ::Train(const vector<int>& seq)
{
    if(not net->allocatedFrozenWeights) die("Gitouttahier!\n");
    //cleanup memory used by the net, allocate gradient and Q outputs
    Grads * g = new Grads(net->nWeights,net->nBiases);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    vector<Real> Qs(nActions), Qhats(nActions), Qtildes(nActions);
    stats.minQ=1e5; stats.maxQ=-1e5; stats.MSE=0; stats.avgQ=0;
    
    for (int jnd(0); jnd<seq.size(); jnd++) {
        const int ind = seq[jnd];
        const int ndata = T->Set[ind]->tuples.size();
        net->allocateSeries(ndata+1);
        profiler->start("D");
        net->assignDropoutMask();
        profiler->stop("D");
        profiler->start("F");
        
        net->predict(T->Set[ind]->tuples[0]->s, Qhats,
                     net->series[0], net->series[1]);
        
        for (int k=1; k<ndata; k++) {//TODO clean this shit up
            Qs = Qhats;
            const Real * const _t = T->Set[ind]->tuples[k];
            
            if (k+1==ndata && T->Set[ind]->ended) {
                for (int i=0; i<nActions; i++)
                    *(net->series[k]->errvals +net->iOutputs+i) = 0;
                
                const Real target = _t->r;
                const Real err =  (target - Qs[_t->a]);
                *(net->series[k]->errvals +net->iOutputs+_t->a) = stats.weight*err;
                const Real max_Q = *max_element(Qs.begin(), Qs.end());
                const Real min_Q = *min_element(Qs.begin(), Qs.end());
                stats.MSE  += err*err;
                stats.relE += fabs(err)/(max_Q-min_Q);
                stats.avgQ += target;
                stats.minQ  = std::min(stats.minQ,target);
                stats.maxQ  = std::max(stats.maxQ,target);
            } else {
                #pragma omp parallel sections
                {
                    #pragma omp section
                    net->predict(_t->s, Qhats,   net->series[k], net->series[k+1]);
                    #pragma omp section
                    net->predict(_t->s, Qtildes, net->series[k], net->series[ndata+1], net->frozen_weights, net->frozen_biases);
                }
                
                int Nbest; Real Vhat(-1e10);
                for (int i=0; i<nActions; i++) {
                    *(net->series[k]->errvals +net->iOutputs +i) = 0;
                    if (Qhats[i]>Vhat)  { Nbest=i; Vhat=Qhats[i]; }
                }
                
                Real target = _t->r + gamma*Qtildes[Nbest];

                Real err =  (target - Qs[_t->a]);
                *(net->series[k]->errvals +net->iOutputs+_t->a) = stats.weight*err;
                const Real max_Q = *max_element(Qs.begin(), Qs.end());
                const Real min_Q = *min_element(Qs.begin(), Qs.end());
                stats.MSE += err*err;
                stats.relE += fabs(err)/(max_Q-min_Q);
                stats.avgQ += target;
                stats.minQ = std::min(stats.minQ,target);
                stats.maxQ = std::max(stats.maxQ,target);
            }
        }
        {
            profiler->stop("F");
            //net->clearErrors(net->series[ndata+1]);
            profiler->start("B");
            net->computeDeltasEnd(net->series, ndata);
            for (int k=ndata-1; k>=1; k--)
            {
                #ifdef _BPTT_
                net->computeDeltasSeries(net->series, k);
                #else
                net->computeDeltasEnd(net->series, k);
                #endif
            }
            profiler->stop("B");
            profiler->start("G");
            
            //#pragma omp parallel
            for (int k=1; k<=ndata; k++)
            {
                net->computeGradsLightSeries(net->series, k, g);
                opt->stackGrads(net->grad,g);
            }
            net->removeDropoutMask(); //before the update!!!
            profiler->stop("G");
        }
    }
    
    profiler->start("O");
    opt->update(net->grad);
    profiler->stop("O");
    
    delete g;
    stats.MSE=std::sqrt(stats.MSE/(ndata-2));
    stats.avgQ/=ndata;
    stats.relE/=ndata;
}

void NFQ::Train(const vector<int>& seq, const vector<int>& samp)
{
    //samples[i] = Set[seq].tuples[ind-back+1];
    //sOlds[i] = &(Set[seq].tuples[ind-back]->s);
    //term[i]= ind==indTransitions[seq]-1 && Set[seq].ended;
    if(not net->allocatedFrozenWeights) die("Allocate them!\n");
    const int ndata = seq.size();
    //cleanup memory used by the net, allocate gradient and Q outputs
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    Grads * g = new Grads(net->nWeights,net->nBiases);
    vector<Real> Qs(nActions), Qhats(nActions), Qtildes(nActions);
    stats.minQ=1e5; stats.maxQ=-1e5; stats.MSE=0; stats.avgQ=0;
    
    for (int k=0; k<ndata; k++) { //TODO clean this shit up
        profiler->start("F");
        const int knd(seq[k]), ind(samp[k]);
        net->predict(T->Set[knd]->tuples[ind]->s, Qs,
                     net->series[0], net->series[1]);
        const Real * const _t = T->Set[knd]->tuples[ind+1];
        #pragma omp parallel sections
        {
        #pragma omp section
            net->predict(_t->s, Qhats,   net->series[1], net->series[2]);
        #pragma omp section
            net->predict(_t->s, Qtildes, net->series[1], net->series[3],
                         net->frozen_weights, net->frozen_biases);
        }
        
        int Nbest; Real Vhat(-1e10);
        for (int i=0; i<nActions; i++) {
            *(net->series[1]->errvals +net->iOutputs+i) = 0;
            if (Qhats[i]>Vhat)  { Nbest=i; Vhat=Qhats[i]; }
        }
        const bool term = ind+2==T->Set[knd]->tuples.size() && T->Set[knd]->ended;
        const Real target = (term) ? _t->r : _t->r + gamma*Qtildes[Nbest];

        Real err =  (target - Qs[a[k]]);
        *(net->series[1]->errvals +net->iOutputs+t->a) = stats.weight*err;
        const Real max_Q = *max_element(Qs.begin(), Qs.end());
        const Real min_Q = *min_element(Qs.begin(), Qs.end());
        stats.MSE += err*err;
        stats.relE += fabs(err)/(max_Q-min_Q);
        stats.avgQ += target;
        stats.minQ = std::min(stats.minQ,target);
        stats.maxQ = std::max(stats.maxQ,target);
        profiler->stop("F");
        profiler->start("B");
        net->computeDeltasEnd(net->series, 1);
        profiler->stop("B");
        profiler->start("G");
        net->computeGradsLightSeries(net->series, 1, g);
        opt->stackGrads(net->grad,g);
        profiler->stop("G");
    }
    profiler->start("O");
    opt->update(net->grad);
    profiler->stop("O");
    
    delete g;
    stats.MSE=std::sqrt(stats.MSE/(ndata-1));
    stats.avgQ/=ndata;
    stats.relE/=ndata;
}

void NFQ::save(string name)
{
    net->save(name + ".net");
}

bool NFQ::restart(string name)
{
    _info("Restarting from saved policy...\n");
    
    T->restartSamples();
    if ( net->restart(name + ".net") ) {_info("Restart successful, moving on...\n");}
    else { _info("Not all policies restarted, therefore assumed zero. Moving on...\n");}
    savePolicy(fname);
    
    FILE * f = fopen("policy.status", "r");
    if(f != NULL) {
        int val;
        fscanf(f, "policy iter: %d\n", &val);
        if(val>=0) T->anneal = val;
        printf("policy iter: %d\n", T->anneal);
        fclose(f);
    }
}



/*
void NFQ::get(const State& sOld, vector<Real> & Qold, const State& s, vector<Real> & Q, int iAgent)
{
    vector<Real> scaledInpOld(nStateDims);
    
    s.scaleUsed(scaledInp);
    sOld.scaleUsed(scaledInpOld);
    Qold.resize(nActions);
    Q.resize(nActions);
    
    net->allocateSeries(2);
    net->expandMemory(net->mem[iAgent], net->series[0]);
    
    net->predict(scaledInpOld, Qold, net->series[0], net->series[1]);
    net->predict(scaledInp,    Q,    net->series[1], net->series[2]);
    
    net->expandMemory(net->mem[iAgent], net->series[1]);
}

Real NFQ::get(const State& s, const Action& a, int iAgent)
{
    s.scaleUsed(scaledInp);
    
    net->expandMemory(net->mem[iAgent], net->series[0]);
    net->predict(scaledInp, prediction, net->series[0], net->series[1]);
    net->expandMemory(net->mem[iAgent], net->series[1]);
    
    return prediction[a.vals[0]];
}


Real NFQ::getMax(const State& s, Action& a, int iAgent)
{
    s.scaleUsed(scaledInp);
    Real Val = -1e10;
    
    net->expandMemory(net->mem[iAgent], net->series[0]); //used by RNN to update recurrent signals
    net->predict(scaledInp, prediction, net->series[0], net->series[1]);
#ifdef _dumpNet_
    net->dump(iAgent);
#endif
    net->expandMemory(net->mem[iAgent], net->series[1]);
    
    for (int i=0; i<prediction.size(); ++i)
        if (prediction[i]>Val)
        {
            a.vals[0] = i;
            Val = prediction[i];
        }
    
    return Val;
}

void NFQ::correct(const State& s, const Action& a, Real err, int iAgent)
{
    for (int i=0; i<prediction.size(); ++i) prediction[i] = 0.;
    prediction[a.vals[0]] = err;
    
    net->expandMemory(net->mem[iAgent], net->series[1]);
    net->computeGrads(prediction, net->series[0], net->series[1], net->grad);
    opt->update(net->grad);
    net->expandMemory(net->mem[iAgent], net->series[1]);
}
*/