/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "NAF.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <cmath>



NAF::NAF(Environment* env, Settings & settings) :
Learner(env,settings), nA(aInfo.dim), nL((aInfo.dim*aInfo.dim+aInfo.dim)/2)
{
}

void NAF::select(const int agentId, State& s, Action& a, State& sOld, Action& aOld, const int info, Real r)
{
    Real newEps(greedyEps);
    vector<Real> output(nOutputs), inputs(nInputs);
    const int handicap = min(static_cast<int>(T->Set.size()), stats.epochCount*10);
    if (bTrain) newEps = (.1 +greedyEps*exp(-handicap/500.));//*agentId/Real(agentId+1);
    if (info!=1) T->passData(agentId, info, sOld, aOld, s, r); //else sOld,aOld,r are junk
    
    s.scaleUsed(inputs);
    net->expandMemory(net->mem[agentId], net->series[0]); //RNN to update recurrent signals
    net->predict(inputs, output, net->series[0], net->series[1]);
#ifdef _dumpNet_
    net->dump(agentId);
#endif
    net->expandMemory(net->mem[agentId], net->series[1]);
    a.descale(getPolicy(output));
    //printf("%d %d %f\n",a.vals[0],i,a.valsContinuous[0]);
    
    uniform_real_distribution<Real> dis(0.,1.);
    if(dis(*gen) < newEps) a.getRand();
}

void NAF::Train(const int thrID, const int seq, const int first)
{
    if(not net->allocatedFrozenWeights) die("Gitouttahier!\n");
    vector<Real> output(nOutputs),gradient(nOutputs);

    const int ndata = T->Set[seq]->tuples.size();
    
    for (int k=0; k<ndata-1; k++) {
        const Tuple * const _t = T->Set[seq]->tuples[k+1];
        if(k==0) net->predict(T->Set[seq]->tuples[0]->s, output, net->series[first]);
        else net->predict(T->Set[seq]->tuples[k]->s, output, net->series[first+k-1], net->series[first+k]);
        //, net->Vweights[thrID], net->Vbiases[thrID]
        
        vector<Real> Q(computeQandGrad(gradient, _t->aC, output));
        //printf("%d %f %f %f\n", thrID, Q[0], Q[1], Q[2]);
        
        if (k+2==ndata && T->Set[seq]->ended) {
            const Real tgt = _t->r;
            //printf("last r %f\n",_t->r);
            const Real err = tgt - Q[0];
            dumpStats(Vstats[thrID], tgt, err, Q);
            for (int i(0); i<nOutputs; i++)
                *(net->series[first+k]->errvals +net->iOutputs+i) = err*gradient[i];
        } else {
            net->predict(_t->s, output, net->series[first+k], net->series[first+ndata-1],
                                        net->frozen_weights,  net->frozen_biases);
            // net->VFweights[thrID], net->VFbiases[thrID]
            const Real tgt = _t->r + gamma*output[0];
            //printf("r %f\n",_t->r);
            const Real err = tgt - Q[0];
            dumpStats(Vstats[thrID], tgt, err, Q);
            for (int i(0); i<nOutputs; i++)
                *(net->series[first+k]->errvals +net->iOutputs+i) = err*gradient[i];
        }
    }
    {
        net->computeDeltasSeries(net->series, first, first+ndata-2);
        //, net->Vweights[thrID], net->Vbiases[thrID]
        net->computeAddGradsSeries(net->series, first, first+ndata-2, net->Vgrad[thrID]);
    }
}

void NAF::Train(const vector<int>& seq)
{
    if(not net->allocatedFrozenWeights) die("Gitouttahier!\n");
    vector<Real> output(nOutputs),gradient(nOutputs);
    int countUpdate(0);
    
    for (int jnd(0); jnd<seq.size(); jnd++) {
        const int ind = seq[jnd];
        const int ndata = T->Set[ind]->tuples.size();
        net->allocateSeries(ndata-1);
        //net->assignDropoutMask();
        
        for (int k=0; k<ndata-1; k++) {
            if(k==0) net->predict(T->Set[ind]->tuples[0]->s, output, net->series[0]);
            else     net->predict(T->Set[ind]->tuples[k]->s, output, net->series[k-1], net->series[k]);
            const Tuple * const _t = T->Set[ind]->tuples[k+1];
            vector<Real> Q(computeQandGrad(gradient, _t->aC, output));
            
            if (k+2==ndata && T->Set[ind]->ended) {
                const Real tgt = _t->r;
                const Real err = tgt - Q[0];
                dumpStats(tgt, err, Q);
                for (int i(0); i<nOutputs; i++)
                    *(net->series[k]->errvals +net->iOutputs+i) = err*gradient[i];
            } else {
                net->predict(_t->s, output, net->series[k], net->series[ndata-1],
                             net->frozen_weights, net->frozen_biases);
                const Real tgt = _t->r + gamma*output[0];
                const Real err = tgt - Q[0];
                dumpStats(tgt, err, Q);
                for (int i(0); i<nOutputs; i++)
                    *(net->series[k]->errvals +net->iOutputs+i) = err*gradient[i];
            }
        }
        {
            net->computeDeltasSeries(net->series, 0, ndata-2);
            net->computeAddGradsSeries(net->series, 0, ndata-2, net->grad);
            countUpdate+=ndata-2;
            /*
            for (int k=0; k<ndata-1; k++) {
                net->computeGradsSeries(net->series, k, net->_grad);
                opt->stackGrads(net->grad,net->_grad);
                countUpdate++;
            }*/
            //net->removeDropoutMask(); //before the update!
        }
    }
    opt->nepoch=stats.epochCount;
    opt->update(net->grad,countUpdate);
}

void NAF::Train(const int thrID, const int seq, const int samp, const int first)
{
    if(not net->allocatedFrozenWeights) die("Allocate them!\n");
    vector<Real> output(nOutputs),gradient(nOutputs);
    
    const Tuple * const _t = T->Set[seq]->tuples[samp+1];
    net->predict(T->Set[seq]->tuples[samp]->s, output, net->series[first], net->Vweights[thrID], net->Vbiases[thrID]);
    
    const bool term = samp+2==T->Set[seq]->tuples.size() && T->Set[seq]->ended;
    vector<Real> Q(computeQandGrad(gradient, _t->aC, output));
    
    if (not term) {
        net->predict(_t->s, output, net->series[first],  net->series[first+1], net->VFweights[thrID], net->VFbiases[thrID]);
    }
    const Real target = (term) ? _t->r : _t->r + gamma*output[0];
    const Real err =  (target - Q[0]);
    dumpStats(Vstats[thrID], target, err, Q);
    for (int i(0); i<nOutputs; i++) {
        *(net->series[first]->errvals +net->iOutputs+i) = err*gradient[i];
    }
    
    net->computeDeltas(net->series[first], net->Vweights[thrID], net->Vbiases[thrID]);
    net->computeAddGrads(net->series[first], net->Vgrad[thrID]);
}

void NAF::Train(const vector<int>& seq, const vector<int>& samp)
{
    if(not net->allocatedFrozenWeights) die("Allocate them!\n");
    vector<Real> output(nOutputs),gradient(nOutputs);
    const int ndata = seq.size();
    int countUpdate(0);
    
    for (int k=0; k<ndata; k++) { //TODO clean this shit up
        const int knd(seq[k]), ind(samp[k]);
        const Tuple * const _t = T->Set[knd]->tuples[ind+1];
        net->predict(T->Set[knd]->tuples[ind]->s, output, net->series[0]);
        
        const bool term = ind+2==T->Set[knd]->tuples.size() && T->Set[knd]->ended;
        vector<Real> Q(computeQandGrad(gradient, _t->aC, output));
        
        if (not term) {
            net->predict(_t->s, output, net->series[0], net->series[1],
                                net->frozen_weights, net->frozen_biases);
        }
        const Real target = (term) ? _t->r : _t->r + gamma*output[0];
        const Real err =  (target - Q[0]);
        dumpStats(target, err, Q);
        for (int i(0); i<nOutputs; i++) {
            *(net->series[0]->errvals +net->iOutputs+i) = err*gradient[i];
        }
        
        net->computeDeltas(net->series[0]);
        net->computeAddGrads(net->series[0], net->grad);
        //net->computeGrads(net->series[0], net->_grad);
        //opt->stackGrads(net->grad,net->_grad);
        countUpdate++;
    }
    opt->nepoch=stats.epochCount;
    opt->update(net->grad, countUpdate);
}

vector<Real> NAF::getPolicy(const vector<Real>& out) const
{
    vector<Real> act(nA);
    for (int j(0); j<nA; j++) act[j] = out[1+nL+j];
    return act;
}

vector<Real> NAF::computeQandGrad(vector<Real>& grad, const vector<Real>& act, const vector<Real>& out) const
{
    vector<Real> Q(3, out[0]);
    vector<Real> _L(nA*nA,0), _A(nA*nA,0), _dLdl(nA*nA), _dPdl(nA*nA), _u(nA), _uL(nA), _uU(nA);
    grad[0] = 1.;
    
    int kL(1);
    for (int j(0); j<nA; j++) {
        _u[j]  = act[j] - out[1+nL+j];
        _uL[j] = -1. - out[1+nL+j];
        _uU[j] =  1. - out[1+nL+j];
        for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            if (i<=j) _L[ind] = out[kL++];
        }
    }
    assert(kL==1+nL);
    
    for (int j(0); j<nA; j++) for (int i(0); i<nA; i++) {
        const int ind = nA*j + i;
        for (int k(0); k<nA; k++) {
            const int k1 = nA*j + k;
            const int k2 = nA*i + k;
            _A[ind] += _L[k1] * _L[k2];
        }
        Q[0] -= .5*_A[ind]*_u[i]*_u[j]; //rescaled!! -1 1
        Q[1] -= .5*_A[ind]*_uL[i]*_uL[j];
        Q[2] -= .5*_A[ind]*_uU[i]*_uU[j];
    }
    Q[2] = std::min(Q[1],Q[2]);
    Q[1] = out[0];
    //printf("Process %d %f %f %f\n", omp_get_thread_num(), Q[0], Q[1], Q[2]);
    
    for (int il(0); il<nL; il++) {
        int kD(0);
        for (int j(0); j<nA; j++) for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            _dLdl[ind] = 0;
            if(i<=j) { if(kD++==il) _dLdl[ind]=1; }
        }
        assert(kD==nL);
        
        for (int j(0); j<nA; j++) for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            _dPdl[ind] = 0;
            for (int k(0); k<nA; k++) {
                const int k1 = nA*j + k;
                const int k2 = nA*i + k;
                _dPdl[ind] += _dLdl[k1]*_L[k2]+_L[k1]*_dLdl[k2];
            }
        }
        
        grad[1+il] = 0.;
        for (int j(0); j<nA; j++) for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            grad[1+il] += -0.5*_dPdl[ind]*_u[i]*_u[j];
        }
    }
    
    for (int ia(0); ia<nA; ia++) {
        grad[1+nL+ia] = 0.;
        for (int i(0); i<nA; i++) {
            const int ind = nA*ia + i;
            grad[1+nL+ia] += _A[ind]*_u[i];
        }
    }
    
    return Q;
}
/*
Real NAF::computeQandGrad(vector<Real>& grad, const vector<Real>& act, const vector<Real>& out) const
{
    Real Q(out[0]);
    vector<Real> _L(nA*nA,0), _A(nA*nA,0), _dLdl(nA*nA), _dPdl(nA*nA), _u(nA);
    grad[0] = 1.;
    
    int kL(1);
    for (int j(0); j<nA; j++) {
        _u[j] = act[j] - out[1+nL+j];
        for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            if (i<=j) _L[ind] = out[kL++];
        }
    }
    assert(kL==1+nL);
    
    for (int j(0); j<nA; j++) for (int i(0); i<nA; i++) {
        const int ind = nA*j + i;
        for (int k(0); k<nA; k++) {
            const int k1 = nA*j + k;
            const int k2 = nA*i + k;
            _A[ind] += _L[k1] * _L[k2];
        }
        Q -= .5*_A[ind]*_u[i]*_u[j];
    }
    
    for (int il(0); il<nL; il++) {
        int kD(0);
        for (int j(0); j<nA; j++) for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            _dLdl[ind] = 0;
            if(i<=j) { if(kD++==il) _dLdl[ind]=1; }
        }
        assert(kD==nL);
        
        for (int j(0); j<nA; j++) for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            _dPdl[ind] = 0;
            for (int k(0); k<nA; k++) {
                const int k1 = nA*j + k;
                const int k2 = nA*i + k;
                _dPdl[ind] += _dLdl[k1]*_L[k2]+_L[k1]*_dLdl[k2];
            }
        }
        
        grad[1+il] = 0.;
        for (int j(0); j<nA; j++) for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            grad[1+il] += -0.5*_dPdl[ind]*_u[i]*_u[j];
        }
    }
    
    for (int ia(0); ia<nA; ia++) {
        grad[1+nL+ia] = 0.;
        for (int i(0); i<nA; i++) {
            const int ind = nA*ia + i;
            grad[1+nL+ia] += _A[ind]*_u[i];
        }
    }

    return Q;
}*/