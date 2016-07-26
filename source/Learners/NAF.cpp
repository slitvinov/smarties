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

void NAF::select(const int agentId,State& s,Action& a,State& sOld,Action& aOld,const int info,Real r)
{
    vector<Real> output(nOutputs), inputs(nInputs);
    s.scaleUsed(inputs);
    
    if (info==1) //is it the first action performed by agent?
    {   //then old state, old action and reward are meaningless, just give the action
        net->predict(inputs, output, net->series[1]);
    }
    else
    {   //then if i'm using RNN i need to load recurrent connections
        net->expandMemory(net->mem[agentId], net->series[0]);
        net->predict(inputs, output, net->series[0], net->series[1]);
        //also, store sOld, aOld -> sNew, r
        data->passData(agentId, info, sOld, aOld, s, r);
    }
    #ifdef _dumpNet_
    net->dump(agentId);
    #endif
    
    //save network transition
    net->expandMemory(net->mem[agentId], net->series[1]);
    
    //load computed policy into a
    vector<Real> act(nA);
    for (int j(0); j<nA; j++) act[j] = output[1+nL+j];
    a.descale(act);
    
    //printf("%f %f\n",act[0],a.valsContinuous[0]);
    //random action?
    Real newEps(greedyEps);
    if (bTrain) { //if training: anneal random chance if i'm just starting to learn
        const Real crutch_1 = 1e6;//stats.relE < 0.5 ? 100. : 1./stats.relE;
        const Real crutch_2 = static_cast<int>(data->Set.size())/500.;
        const Real crutch_3 = stats.epochCount/10.;
        const Real handicap = min(min(crutch_1,crutch_2),crutch_3);
        newEps = (.1 +greedyEps*exp(-handicap));//*agentId/Real(agentId+1);
        //printf("Random action %f %f %f %f\n",crutch_1,crutch_2,crutch_3,newEps);
    }
    
    uniform_real_distribution<Real> dis(0.,1.);
    if(dis(*gen) < newEps) {
        //printf("Random action %d %d %f\n",data->Set.size(),stats.epochCount,newEps);
        std::normal_distribution<Real> dist(0.,0.5);
        vector<Real> scaledRandomAction(nA);
        for (int i=0; i<nA; i++) scaledRandomAction[i] = std::tanh(dist(*gen));
        
        a.descale(scaledRandomAction);
        //printf("Random action %d  %f %f \n",a.vals[0], a.valsContinuous[0], scaledRandomAction[0]);fflush(0);
    }
    
    //if (info!=1) printf("Agent %d: %s > %s with %s rewarded with %f acting %s\n", agentId, sOld.print().c_str(), s.print().c_str(), aOld.print().c_str(), r ,a.print().c_str());
}

void NAF::Train_BPTT(const int seq, const int first, const int thrID)
{
    if(not net->allocatedFrozenWeights) die("Gitouttahier!\n");
    vector<Real> target(nOutputs), output(nOutputs), gradient(nOutputs);
    const int ndata = data->Set[seq]->tuples.size();
    
    for (int k=0; k<ndata-1; k++) {//state in k=[0:N-2], act&rew in k+1, last state (N-1) not used for Q update
        //this tuple contains a, sNew, reward:
        const Tuple * const _t = data->Set[seq]->tuples[k+1];
        //this tuple contains sOld:
        const Tuple * const _tOld = data->Set[seq]->tuples[k];
        
        //if first in a sequence, no input from recurrent links
        if(k==0)
            net->predict(_tOld->s, output, net->series[first]);
        else
            net->predict(_tOld->s, output, net->series[first+k-1], net->series[first+k]);
        
        const bool terminal = k+2==ndata && data->Set[seq]->ended;
        
        if (not terminal) {
            net->predict(_t->s, target, net->series[first+k], net->series[first+ndata-1],
                                        net->tgt_weights,  net->tgt_biases);
        }
        
        Real err = (terminal) ? _t->r : _t->r + gamma*target[0];
        const vector<Real> Q(computeQandGrad(gradient, _t->aC, output, err));
        net->setOutputErrors(gradient, net->series[first+k]);
        //for (int i(0); i<nOutputs; i++) { //put grad into network
        //    *(net->series[first+k]->errvals +net->iOutputs+i) = gradient[i];
        //}
        
        dumpStats(Vstats[thrID], Q[0], err, Q);
    }
    
    net->computeDeltasSeries(net->series, first, first+ndata-2);
    
    if (first==0)
        net->computeAddGradsSeries(net->series, 0, ndata-2, net->grad);
    else
        net->computeAddGradsSeries(net->series, first, first+ndata-2, net->Vgrad[thrID]);
}

void NAF::Train(const int seq, const int samp, const int first, const int thrID)
{
    if(not net->allocatedFrozenWeights) die("Allocate them!\n");
    vector<Real> target(nOutputs), output(nOutputs), gradient(nOutputs);
    const int ndata = data->Set[seq]->tuples.size();
    
    //this tuple contains a, sNew, reward:
    const Tuple * const _t = data->Set[seq]->tuples[samp+1];
    //sOld contained in previous tuple:
    net->predict(data->Set[seq]->tuples[samp]->s, output, net->series[first]);
    
    const bool terminal = samp+2==ndata && data->Set[seq]->ended;
    if (not terminal) {
        net->predict(_t->s, target, net->series[first+1],net->tgt_weights,net->tgt_biases);
    }
    
    Real err = (terminal) ? _t->r : _t->r + gamma*target[0];
    const vector<Real> Q(computeQandGrad(gradient, _t->aC, output, err));
    net->setOutputErrors(gradient, net->series[first]);
    //for (int i(0); i<nOutputs; i++) {
    //    *(net->series[first]->errvals +net->iOutputs+i) = gradient[i];
    //}
    
    dumpStats(Vstats[thrID], Q[0], err, Q);
    net->computeDeltas(net->series[first]);
    
    if (first==0) net->computeAddGrads(net->series[first], net->grad);
    else          net->computeAddGrads(net->series[first], net->Vgrad[thrID]);    
}

#if 1==1 //original formulation of advantage = 0.5 (a - pi)' * A * (a - pi), does not work: why?

vector<Real> NAF::computeQandGrad(vector<Real>& grad,const vector<Real>& act,vector<Real>& out,Real& error) const
{
    vector<Real> Q(3), _L(nA*nA,0), _A(nA*nA,0), _dLdl(nA*nA), _dPdl(nA*nA), _u(nA), _uL(nA), _uU(nA);
    
    int kL(1);
    for (int j(0); j<nA; j++) {
        //compute u = act-pi
        _u[j]  = act[j] - out[1+nL+j];
        //to compute the relative error: Q for the max and min actions
        _uL[j] = -1.    - out[1+nL+j];
        _uU[j] =  1.    - out[1+nL+j];
        
        #ifdef _scaleR_
        if (out[1+j]>0.99999) out[1+j] = 0.99999;
        if (out[1+j]<-.99999) out[1+j] = -.99999;
        #endif
        //put in place elements of lower diag matrix L
        for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            #ifdef _scaleR_
            //if scaleR, net output is logic e [-1, 1] this transforms l outputs back to linear
            if (i<=j) {
                const Real l = .5*std::log((1+out[kL])/(1-out[kL]));
                _L[ind] = l;
                kL++;
            }
            #else
            if (i<=j) _L[ind] = out[kL++];
            #endif
        }
    }
    assert(kL==1+nL);
    
    //A = L * L'
    for (int j(0); j<nA; j++)
    for (int i(0); i<nA; i++) {
        const int ind = nA*j + i;
        for (int k(0); k<nA; k++) {
            const int k1 = nA*j + k;
            const int k2 = nA*i + k;
            _A[ind] += _L[k1] * _L[k2];
        }
        Q[0] += _A[ind]*_u[i]*_u[j]; //Advantage = u' * A * u
        Q[1] += _A[ind]*_uL[i]*_uL[j];
        Q[2] += _A[ind]*_uU[i]*_uU[j];
    }

    //Q(s,a) = V(s) - .5 * Advantage(s,a)
    Q[2] = out[0]- .5*std::max(Q[1],Q[2]);
    Q[0] = out[0]- .5*Q[0];
    Q[1] = out[0];
    error -= Q[0];
    
    grad[0] = error;
    for (int il(0); il<nL; il++) {
        int kD(0);
        for (int j(0); j<nA; j++)
        for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            _dLdl[ind] = 0;
            if(i<=j) { if(kD++==il) _dLdl[ind]=1; }
        }
        assert(kD==nL);
        
        for (int j(0); j<nA; j++)
        for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            _dPdl[ind] = 0;
            for (int k(0); k<nA; k++) {
                const int k1 = nA*j + k;
                const int k2 = nA*i + k;
                _dPdl[ind] += _dLdl[k1]*_L[k2] + _L[k1]*_dLdl[k2];
            }
        }
        
        grad[1+il] = 0.;
        for (int j(0); j<nA; j++)
        for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            grad[1+il] += -0.5*_dPdl[ind]*_u[i]*_u[j];
        }
        
        #ifdef _scaleR_
        //if scaleR, net output is logic e [-1, 1], this transforms l gradient back to linear
        grad[1+il] *= error/(1 - out[1+il]*out[1+il]);
        #else
        grad[1+il] *= error;
        #endif
    }
    
    for (int ia(0); ia<nA; ia++) {
        grad[1+nL+ia] = 0.;
        for (int i(0); i<nA; i++) {
            const int ind = nA*ia + i;
            grad[1+nL+ia] += _A[ind]*_u[i];
        }
        grad[1+nL+ia] *= error;
    }
    //1 action dim dump:
    ///printf("act %f, err %f, out %f %f %f, u %f, Q %f, grad %f %f %f\n", act[0], error, out[0], out[1], out[2], _u[0], Q[0], grad[0], grad[1], grad[2]);
    //2 actions dim dump
    ///printf("act %f %f, err %f, out %f %f %f %f %f %f, u %f %f, Q %f, grad %f %f %f %f %f %f\n", act[0], act[1], error,
    ///out[0], out[1], out[2], out[3], out[4], out[5], _u[0], _u[1], Q[0], grad[0], grad[1], grad[2], grad[3], grad[4], grad[5]);
    return Q;
}

#else  //my formulation of advantage = 2 pow((a - pi)' * A * (a - pi), 0.25), works

vector<Real> NAF::computeQandGrad(vector<Real>& grad,const vector<Real>& act,vector<Real>& out,Real& error) const
{
    vector<Real> Q(3,0), _L(nA*nA,0), _A(nA*nA,0), _dLdl(nA*nA), _dPdl(nA*nA), _u(nA), _uL(nA), _uU(nA);
    
    int kL(1);
    for (int j(0); j<nA; j++) { //compute u = act-pi and matrix L
        _u[j]  = act[j] - out[1+nL+j];
        _uL[j] = -1.    - out[1+nL+j];
        _uU[j] =  1.    - out[1+nL+j];

        #ifdef _scaleR_
        if (out[1+j]>0.99999) out[1+j] = 0.99999;
        if (out[1+j]<-.99999) out[1+j] = -.99999;
        #endif
            
        for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            #ifdef _scaleR_
            //if scaleR, net output is logic e [-1, 1] this transforms l outputs back to linear
            if (i<=j) {
                const Real l = .5*std::log((1+out[kL])/(1-out[kL]));
                _L[ind] = l;
                kL++;
            }
            #else
            if (i<=j) _L[ind] = out[kL++];
            #endif
        }
    }
    assert(kL==1+nL);
    
    for (int j(0); j<nA; j++)
    for (int i(0); i<nA; i++) { //A = L * L'
        const int ind = nA*j + i;
        for (int k(0); k<nA; k++) {
            const int k1 = nA*j + k;
            const int k2 = nA*i + k;
            _A[ind] += _L[k1] * _L[k2];
        }
        Q[0] += _A[ind]*_u[i]*_u[j]; //Advantage = u' * A * u
        Q[1] += _A[ind]*_uL[i]*_uL[j];
        Q[2] += _A[ind]*_uU[i]*_uU[j];
    }
    
    const Real fac = .5*pow(Q[0],-.75);
    Q[2] = out[0]-2.*std::pow(std::max(Q[1],Q[2]),0.25);
    Q[0] = out[0]-2.*std::pow(Q[0],0.25);  //Q = V - 2*Adv^.25
    Q[1] = out[0];
    error -= Q[0];
    
    grad[0] = error;
    for (int il(0); il<nL; il++) {
        int kD(0);
        for (int j(0); j<nA; j++)
        for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            _dLdl[ind] = 0;
            if(i<=j) { if(kD++==il) _dLdl[ind]=1; }
        }
        assert(kD==nL);
        
        for (int j(0); j<nA; j++)
        for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            _dPdl[ind] = 0;
            for (int k(0); k<nA; k++) {
                const int k1 = nA*j + k;
                const int k2 = nA*i + k;
                _dPdl[ind] += _dLdl[k1]*_L[k2]+_L[k1]*_dLdl[k2];
            }
        }
        
        grad[1+il] = 0.;
        for (int j(0); j<nA; j++)
        for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            grad[1+il] += -_dPdl[ind]*_u[i]*_u[j];
        }
        
        #ifdef _scaleR_ 
        //if scaleR, net output is logic e [-1, 1], this transforms l gradient back to linear
        grad[1+il] *= fac*error/(1 - out[1+il]*out[1+il]);
        #else
        grad[1+il] *= fac*error;
        #endif
    }
    
    for (int ia(0); ia<nA; ia++) {
        grad[1+nL+ia] = 0.;
        for (int i(0); i<nA; i++) {
            const int ind = nA*ia + i;
            grad[1+nL+ia] += 2.*_A[ind]*_u[i];
        }
        grad[1+nL+ia] *= fac*error;
    }
    //1 action dim dump:
    //printf("act %f, err %f, out %f %f %f, u %f, Q %f, grad %f %f %f\n", act[0], error, out[0], out[1], out[2], _u[0], Q[0], grad[0], grad[1], grad[2]);
    //2 actions dim dump
    ///printf("act %f %f, err %f, out %f %f %f %f %f %f, u %f %f, Q %f, grad %f %f %f %f %f %f\n", act[0], act[1], error, out[0],  out[1], out[2], out[3],  out[4], out[5], _u[0], _u[1], Q[0], grad[0], grad[1], grad[2], grad[3], grad[4], grad[5]);
    return Q;
}
#endif

/*
#ifdef _scaleR_
vector<Real> NAF::computeQandGrad(vector<Real>& grad, const vector<Real>& act, const vector<Real>& out, Real& error) const
{
    vector<Real> Q(3, out[0]), _u(nA), _uL(nA), _uU(nA);
    
    for (int j(0); j<nA; j++) {
        _u[j]  = act[j] - out[1+nA+j];
        _uL[j] = -1.    - out[1+nA+j];
        _uU[j] =  1.    - out[1+nA+j];
        const Real A = -std::log(.5-.5*out[1+j]);
        //Q[0] -= A*fabs(_u[j] );
        //Q[1] -= A*fabs(_uL[j]);
        //Q[2] -= A*fabs(_uU[j]);
        Q[0] -= sqrt(A*fabs(_u[j] ));
        Q[1] -= sqrt(A*fabs(_uL[j]));
        Q[2] -= sqrt(A*fabs(_uU[j]));
    }
    
    error -= Q[0];
    Q[2] = std::min(Q[1],Q[2]);
    Q[1] = out[0];
    
    grad[0] = error;
    for (int j(0); j<nA; j++) {
        const Real A = -std::log(.5-.5*out[1+j]);
        grad[1+j]    = -error*fabs(_u[j])/(1-out[1+j]);
        grad[1+nA+j] = (_u[j]>0.) ? error*A : -error*A;
        const Real fac = 0.5 / sqrt( A * fabs(_u[j] ));
        grad[1+j]    = -error*fac*fabs(_u[j])/(1-out[1+j]);
        grad[1+nA+j] = (_u[j]>0.) ? error*A*fac : -error*A*fac;
    }
    
    //printf("act %f, err %f, out %f %f %f, u %f, Q %f, grad %f %f %f\n", act[0], error, out[0],  out[1], out[2], _u[0], Q[0], grad[0], grad[1], grad[2]);
    return Q;
}
#else
 
 //THE FOLLOWING HAS ADV = -A*ABS(ACT-PI) WHERE A IS A POSITIVE COEFFICIENT
vector<Real> NAF::computeQandGrad(vector<Real>& grad, const vector<Real>& act, const vector<Real>& out, Real& error) const
{
    vector<Real> Q(3, out[0]), _u(nA), _uL(nA), _uU(nA);
    
    for (int j(0); j<nA; j++) {
        _u[j]  = act[j] - out[1+nA+j];
        _uL[j] = -1.    - out[1+nA+j];
        _uU[j] =  1.    - out[1+nA+j];
        const Real A = (out[1+j]<0) ? 0 : out[1+j];
        Q[0] -= sqrt(A*fabs(_u[j] ));
        Q[1] -= sqrt(A*fabs(_uL[j]));
        Q[2] -= sqrt(A*fabs(_uU[j]));
    }
    
    error -= Q[0];
    Q[2] = std::min(Q[1],Q[2]);
    Q[1] = out[0];
    
    grad[0] = error;
    for (int j(0); j<nA; j++) {
        if (out[1+j]<=0) {
            grad[1+j] = 10;
            grad[1+nA+j] = 0;
        } else {
            const Real fac = .5/sqrt(out[1+j]*fabs(_u[j]));
            grad[1+j]    = -error*fac*fabs(_u[j]);
            grad[1+nA+j] = (_u[j]>0.) ? error*fac*out[1+j] : -error*fac*out[1+j];
        }
    }
    
    //printf("act %f, err %f, out %f %f %f, u %f, Q %f, grad %f %f %f\n", act[0], error, out[0],  out[1], out[2], _u[0], Q[0], grad[0], grad[1], grad[2]);
    return Q;
}
#endif

 //THE FOLLOWING HAS ADV = -A*ABS(ACT-PI) WHERE A IS A POSITIVE COEFFICIENT
 // A = A1 IF ACT-PI>0, ELSE A=A2
vector<Real> NAF::computeQandGrad(vector<Real>& grad, const vector<Real>& act, const vector<Real>& out, Real& error) const
{
    vector<Real> Q(3, out[0]), _u(nA), _uL(nA), _uU(nA);
    //printf("out %f %f %f %f \n", out[0], out[1], out[2], out[3]);
    
    for (int j(0); j<nA; j++) {
        _u[j]  = act[j]- out[1+2*nA+j];
        _uL[j] = -1.   - out[1+2*nA+j];
        _uU[j] =  1.   - out[1+2*nA+j];
        
        const Real A1 = (out[1   +j]<0) ? 0 : out[1+j];
        const Real A2 = (out[1+nA+j]<0) ? 0 : out[1+nA+j];
        Q[0] += (_u[j]>0. ) ? -A1*_u[j]  : A2*_u[j];
        Q[1] += (_uL[j]>0.) ? -A1*_uL[j] : A2*_uL[j];
        Q[2] += (_uU[j]>0.) ? -A1*_uU[j] : A2*_uU[j];
    }
    
    error -= Q[0];
    Q[2] = std::min(Q[1],Q[2]);
    Q[1] = out[0];
    
    grad[0] = error;
    for (int j(0); j<nA; j++) {
        const Real A1 = (out[1   +j]<0) ? 0 : out[1+j];
        const Real A2 = (out[1+nA+j]<0) ? 0 : out[1+nA+j];
        grad[1+2*nA+j] = (_u[j] > 0.) ? error*A1 : -error*A2;
        grad[1+j]    = (out[1+j]<0)   ? 10. : ((_u[j]>0.)? -error*_u[j] : 0);
        grad[1+nA+j] = (out[1+nA+j]<0)? 10. : ((_u[j]<0.)?  error*_u[j] : 0);
    }
    //printf("act %f, err %f, out %f %f %f %f , u %f, Q %f, grad %f %f %f %f \n", act[0], error, out[0], out[1], out[2], out[3], _u[0], Q[0], grad[0], grad[1], grad[2], grad[3]);
    return Q;
}
 
Real NAF::computeQandGrad(vector<Real>& grad, const vector<Real>& act, const vector<Real>& out, const Real error) const
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
