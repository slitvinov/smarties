/*
 *  Layers.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include "Layers.h"
#include "../ErrorHandling.h"
#include <cassert>

using namespace ErrorHandling;

/* propagate(Mem/Lab * M, Lab * N, Real* weights, Real* biases)
 *
 */

void NormalLayer::propagate(Mem * M, Lab * N, Real* weights, Real* biases)
{
    #pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        updateInputs( n, N, nullptr, weights); //nullptr just because if it's ever accessed
        updateOutputs(n, N, nullptr, weights, biases); // then I DESERVE a segfault
    }
}

void NormalLayer::propagate(Lab * M, Lab * N, Real* weights, Real* biases)
{
    #pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        updateInputs( n, N, nullptr, weights);
        updateOutputs(n, N, nullptr, weights, biases);
    }
}

void LSTMLayer::propagate(Mem * M, Lab * N, Real* weights, Real* biases)
{
    #pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        updateInputs( n, N, M->outvals, weights);
        updateOutputs(n, N, M->ostates, weights, biases);
    }
}

void LSTMLayer::propagate(Lab * M, Lab * N, Real* weights, Real* biases)
{
    #pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        updateInputs( n, N, M->outvals, weights);
        updateOutputs(n, N, M->ostates, weights, biases);
    }
}

/* backPropagate(Mem * M, Lab * N, Dsdw * dsdw, Grads * grad, Real* weights, Real* biases)
 *
 */

void NormalLayer::backPropagate(Mem * M, Lab * N, Dsdw * dsdw, Grads * grad, Real* weights, Real* biases)
{
    #pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        if(!last)
            *(N->errvals +n1stNeuron +n) = 0.0; //here i truncate future errors
        for (const auto & l : *curr_output_links) 
            addErrors(n, l, N, N->errvals, weights);
        
        *(N->errvals +n1stNeuron +n) *= func->evalDiff(*(N->in_vals +n1stNeuron +n));
        
        updateGrads(n, N, nullptr, nullptr, dsdw, grad);
    }
}

void LSTMLayer::backPropagate(Mem * M, Lab * N, Dsdw * dsdw, Grads * grad, Real* weights, Real* biases)
{
    #pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        if(!last)
            *(N->errvals +n1stNeuron +n) = 0.0; //here i truncate future errors
        for (const auto & l : *curr_output_links)
            addErrors(n, l, N, N->errvals, weights);
        
        *(N->eMCell +n1stCell+n) = ifun->evalDiff(*(N->in_vals +n1stNeuron +n)) * *(N->oIGates +n1stCell +n);
        *(N->eIGates+n1stCell+n) = sigm->evalDiff(*(N->iIGates +n1stCell   +n)) * *(N->oMCell  +n1stCell +n);
        *(N->eFGates+n1stCell+n) = sigm->evalDiff(*(N->iFGates +n1stCell   +n)) * *(M->ostates +n1stCell +n);
        *(N->eOGates+n1stCell+n) = sigm->evalDiff(*(N->iOGates +n1stCell   +n)) * ofun->eval(*(N->ostates +n1stCell +n)) * *(N->errvals +n1stNeuron +n);
        *(N->errvals +n1stNeuron +n) *= *(N->oOGates +n1stCell +n) * ofun->evalDiff(*(N->ostates +n1stCell +n));

        updateGrads(n, N, M->outvals, M->ostates, dsdw, grad);
    }
}

void NormalLayer::backPropagate(Lab * M, Lab * N, Dsdw * dsdw, Grads * grad, Real* weights, Real* biases)
{
    #pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        if(!last)
            *(N->errvals +n1stNeuron +n) = 0.0; //here i truncate future errors
        for (const auto & l : *curr_output_links)
            addErrors(n, l, N, N->errvals, weights);
        
        *(N->errvals +n1stNeuron +n) *= func->evalDiff(*(N->in_vals +n1stNeuron +n));
        
        updateGrads(n, N, nullptr, nullptr, dsdw, grad);
    }
}

void LSTMLayer::backPropagate(Lab * M, Lab * N, Dsdw * dsdw, Grads * grad, Real* weights, Real* biases)
{
    #pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        if(!last)
            *(N->errvals +n1stNeuron +n) = 0.0; //here i truncate future errors
        for (const auto & l : *curr_output_links)
            addErrors(n, l, N, N->errvals, weights);
        
        *(N->eMCell +n1stCell+n) = ifun->evalDiff(*(N->in_vals +n1stNeuron +n)) * *(N->oIGates +n1stCell +n);
        *(N->eIGates+n1stCell+n) = sigm->evalDiff(*(N->iIGates +n1stCell   +n)) * *(N->oMCell  +n1stCell +n);
        *(N->eFGates+n1stCell+n) = sigm->evalDiff(*(N->iFGates +n1stCell   +n)) * *(M->ostates +n1stCell +n);
        *(N->eOGates+n1stCell+n) = sigm->evalDiff(*(N->iOGates +n1stCell   +n)) * ofun->eval(*(N->ostates +n1stCell +n)) * *(N->errvals +n1stNeuron +n);
        *(N->errvals +n1stNeuron +n) *= *(N->oOGates +n1stCell +n) * ofun->evalDiff(*(N->ostates +n1stCell +n)); /// multiply by * func->evalDiff(*(N->ostates)) in original lstm
        
        updateGrads(n, N, M->outvals, M->ostates, dsdw, grad);
    }
}

void NormalLayer::backPropagate(Lab * M, Lab * N, Grads * grad, Real* weights, Real* biases)
{
    #pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        if(!last)
            *(N->errvals +n1stNeuron +n) = 0.0; //here i truncate future errors
        for (const auto & l : *curr_output_links)
            addErrors(n, l, N, N->errvals, weights);
        
        *(N->errvals +n1stNeuron +n) *= func->evalDiff(*(N->in_vals +n1stNeuron +n));
        
        updateGrads(n, N, nullptr, nullptr, nullptr, grad);
    }
}

void LSTMLayer::backPropagate(Lab * M, Lab * N, Grads * grad, Real* weights, Real* biases)
{
    #pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        if(!last)
            *(N->errvals +n1stNeuron +n) = 0.0; //here i truncate future errors
        for (const auto & l : *curr_output_links)
            addErrors(n, l, N, N->errvals, weights);
        
        *(N->eMCell +n1stCell+n) = ifun->evalDiff(*(N->in_vals +n1stNeuron +n)) * *(N->oIGates +n1stCell +n);
        *(N->eIGates+n1stCell+n) = sigm->evalDiff(*(N->iIGates +n1stCell   +n)) * *(N->oMCell  +n1stCell +n);
        *(N->eFGates+n1stCell+n) = sigm->evalDiff(*(N->iFGates +n1stCell   +n)) * *(M->ostates +n1stCell +n);
        *(N->eOGates+n1stCell+n) = sigm->evalDiff(*(N->iOGates +n1stCell   +n)) * ofun->eval(*(N->ostates +n1stCell +n)) * *(N->errvals +n1stNeuron +n);
        
        *(N->errvals +n1stNeuron +n) *= *(N->oOGates +n1stCell +n) * ofun->evalDiff(*(N->ostates +n1stCell +n));
        
        //printf("output error %d is %f\n",n1stNeuron+n, *(N->errvals+n1stNeuron+n));
        updateGradsLight(n, N, M->outvals, M->ostates, grad);
    }
}

/* backPropagateDelta(Lab * prev, Lab * curr, Lab * next, Real* weights, Real* biases)
 *
 */

void NormalLayer::backPropagateDelta(Lab * prev, Lab * curr, Lab * next, Real* weights, Real* biases)
{
#pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        if(!last)
        *(curr->errvals +n1stNeuron +n) = 0.0;
        for (const auto & l : *curr_output_links) //nl_l_f
            addErrors(n, l, curr, curr->errvals, weights);
        for (const auto & l : *next_output_links) //nl_l_f
            addErrors(n, l, next, curr->errvals, weights);
        
        *(curr->errvals +n1stNeuron +n) *= func->evalDiff( *(curr->in_vals +n1stNeuron +n) );
        
        //printf("output error %d is %f\n",n1stNeuron+n, *(curr->errvals+n1stNeuron+n));
    }
}

void LSTMLayer::backPropagateDelta(Lab * prev, Lab * curr, Lab * next, Real* weights, Real* biases)
{
#pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        if(!last)
        *(curr->errvals +n1stNeuron +n) = 0.0;
        for (const auto & l : *curr_output_links)
        {
            //printf("curr link\n");
            addErrors(n, l, curr, curr->errvals, weights); //adderrors updates the fourth arg
        }
        for (const auto & l : *next_output_links)
        {
            //printf("next link\n");
            addErrors(n, l, next, curr->errvals, weights);
        }
        
        *(curr->eMCell +n1stCell+n) = ifun->evalDiff(*(curr->in_vals+n1stNeuron+n)) * *(curr->oIGates+n1stCell+n);
        *(curr->eIGates+n1stCell+n) = sigm->evalDiff(*(curr->iIGates+n1stCell  +n)) * *(curr->oMCell +n1stCell+n);
        *(curr->eFGates+n1stCell+n) = sigm->evalDiff(*(curr->iFGates+n1stCell  +n)) * *(prev->ostates+n1stCell+n);
        *(curr->eOGates+n1stCell+n) = sigm->evalDiff(*(curr->iOGates+n1stCell  +n)) * ofun->eval(*(curr->ostates+n1stCell+n)) * *(curr->errvals+n1stNeuron+n);
        
        //the final boss:
        *(curr->errvals+n1stNeuron+n) = *(curr->errvals+n1stNeuron+n) * *(curr->oOGates+n1stCell+n) * ofun->evalDiff(*(curr->ostates +n1stCell +n)) +
                                        *(next->errvals+n1stNeuron+n)* *(next->oFGates+n1stCell+n) +
                                          *(next->eIGates+n1stCell+n)* *(weights+n1stPeep+3*n)   +
                                          *(next->eFGates+n1stCell+n)* *(weights+n1stPeep+3*n+1) +
                                          *(curr->eOGates+n1stCell+n)* *(weights+n1stPeep+3*n+2);
        
        
        //printf("output error %d is %f\n",n1stNeuron+n, *(curr->errvals+n1stNeuron+n));
    }
}

void LSTMLayer::backPropagateDelta(Lab * prev, Lab * curr, Real* weights, Real* biases)
{
#pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        if(!last)
        *(curr->errvals +n1stNeuron +n) = 0.0;
        for (const auto & l : *curr_output_links)
        {
            //printf("curr link\n");
            addErrors(n, l, curr, curr->errvals, weights); //adderrors updates the fourth arg
        }
        
        *(curr->eMCell +n1stCell+n) = ifun->evalDiff(*(curr->in_vals+n1stNeuron+n)) * *(curr->oIGates+n1stCell+n);
        *(curr->eIGates+n1stCell+n) = sigm->evalDiff(*(curr->iIGates+n1stCell  +n)) * *(curr->oMCell +n1stCell+n);
        *(curr->eFGates+n1stCell+n) = sigm->evalDiff(*(curr->iFGates+n1stCell  +n)) * *(prev->ostates+n1stCell+n);
        *(curr->eOGates+n1stCell+n) = sigm->evalDiff(*(curr->iOGates+n1stCell  +n)) * ofun->eval(*(curr->ostates+n1stCell+n)) * *(curr->errvals+n1stNeuron+n);
        
        //the final boss:
        *(curr->errvals+n1stNeuron+n) = *(curr->errvals+n1stNeuron+n) * *(curr->oOGates+n1stCell+n) * ofun->evalDiff(*(curr->ostates +n1stCell +n));
    }
}
void NormalLayer::backPropagateDelta(Lab * prev, Lab * curr, Real* weights, Real* biases)
{
#pragma omp for
    for (int n=0; n<nNeurons; n++)
    {
        if(!last)
            *(curr->errvals +n1stNeuron +n) = 0.0;
        for (const auto & l : *curr_output_links) //nl_l_f
            addErrors(n, l, curr, curr->errvals, weights);
        
        *(curr->errvals +n1stNeuron +n) *= func->evalDiff( *(curr->in_vals +n1stNeuron +n) );
        
        //printf("output error %d is %f\n",n1stNeuron+n, *(curr->errvals+n1stNeuron+n));
    }
}

/* backPropagateGrads(Lab * prev, Lab * curr, Dsdw * dsdw, Grads * grad)
 *
 */

void NormalLayer::backPropagateGrads(Lab * M, Lab * N, Dsdw * dsdw, Grads * grad)
{
#pragma omp for
    for (int n=0; n<nNeurons; n++)
        updateGrads(n, N, M->outvals, M->ostates, dsdw, grad);
}

void LSTMLayer::backPropagateGrads(Lab * M, Lab * N, Dsdw * dsdw, Grads * grad)
{
#pragma omp for
    for (int n=0; n<nNeurons; n++)
        updateGrads(n, N, M->outvals, M->ostates, dsdw, grad);
}

void NormalLayer::backPropagateGradsLight(Lab * M, Lab * N, Grads * grad)
{
#pragma omp for
    for (int n=0; n<nNeurons; n++)
        updateGrads(n, N, nullptr, nullptr, nullptr, grad);
}

void LSTMLayer::backPropagateGradsLight(Lab * M, Lab * N, Grads * grad)
{
#pragma omp for
    for (int n=0; n<nNeurons; n++)
        updateGradsLight(n, N, M->outvals, M->ostates, grad);
}

/* Kernels:
 *
 */
KER1 void NormalLayer::updateInputs(const int n, Lab * N, Real * oldvals, Real* weights)
{
    Real *in;
    #ifndef SIMDKERNELSIN
    in  = (Real *) calloc(1,sizeof(Real)); //yeah it's silly
    #else
    _myallocate(in, SIMD)
    *in = 0.; //zero the first
    vec IN=SET0();
    #endif
    
    for (const auto & l : *curr_input_links)
    {
    #ifdef SIMDKERNELSIN
        if (l->first==false)
            addInputsSIMD(IN, n, l, N->outvals, weights);
        else
    #endif
            addInputs(in, n, l, N->outvals, weights);
    }
    
    *(N->in_vals +n1stNeuron +n) = *in; //first element of in
    #ifdef SIMDKERNELSIN
    STORE (in,IN);
    for(int t=0; t<SIMD; t++)
        *(N->in_vals +n1stNeuron +n) += *(in+t);
    _myfree(in);
    #else
    free (in);
    #endif
}

KER1 void LSTMLayer::updateInputs(  const int n, Lab * N, Real * oldvals, Real* weights)
{
    Real *tC, *tI, *tF, *tO;
    #ifndef SIMDKERNELSIN
    tC  = (Real *) calloc(1,sizeof(Real)); //might be silly
    tI  = (Real *) calloc(1,sizeof(Real)); //cant stop me now
    tF  = (Real *) calloc(1,sizeof(Real));
    tO  = (Real *) calloc(1,sizeof(Real));
    #else
    _myallocate(tC, SIMD)
    *tC = 0.; //zero the first
    _myallocate(tI, SIMD)
    *tI = 0.; //zero the first
    _myallocate(tF, SIMD)
    *tF = 0.; //zero the first
    _myallocate(tO, SIMD)
    *tO = 0.; //zero the first
    vec IN=SET0(); vec IG=SET0(); vec FG=SET0(); vec OG=SET0();
    #endif
    
    for (const auto & l : *curr_input_links)
    {
    #ifdef SIMDKERNELSIN
        if (l->first==false)
            addInputsSIMD(IN, IG, FG, OG, n, l, N->outvals, weights);
        else
    #endif
            addInputs(    tC, tI, tF, tO, n, l, N->outvals, weights);
    }
    for (const auto & l : *prev_input_links)
    #ifdef SIMDKERNELSIN
            addInputsSIMD(IN, IG, FG, OG, n, l, oldvals, weights);
    #else
            addInputs(    tC, tI, tF, tO, n, l, oldvals, weights);
    #endif
    
    *(N->in_vals +n1stNeuron +n) = *tC; *(N->iIGates +n1stCell +n) = *tI;
    *(N->iFGates +n1stCell +n) = *tF;   *(N->iOGates +n1stCell +n) = *tO;
    
    #ifdef SIMDKERNELSIN
    STORE (tC,IN); STORE (tI,IG); STORE (tF,FG); STORE (tO,OG);
    for(int t=1; t<SIMD; t++)
    {
        *tC += *(tC+t); *tI += *(tI+t); *tF += *(tF+t); *tO += *(tO+t);
    }
    *(N->in_vals +n1stNeuron +n) += *tC; *(N->iIGates +n1stCell +n) += *tI;
    *(N->iFGates +n1stCell +n) += *tF;   *(N->iOGates +n1stCell +n) += *tO;
    
    _myfree(tC); _myfree(tI); _myfree(tF); _myfree(tO);
    #else
    free(tC); free(tI); free(tF); free(tO);
    #endif
    
}

KER1 void NormalLayer::updateOutputs(const int n, Lab * N, Real * oldstates, Real* weights, Real* biases)
{
    *(N->in_vals +n1stNeuron +n) += *(biases +n1stBias +n);
    *(N->outvals +n1stNeuron +n) = func->eval( *(N->in_vals +n1stNeuron +n) );
}

KER1 void LSTMLayer::updateOutputs(  const int n, Lab * N, Real * oldstates, Real* weights, Real* biases)
{
    *(N->in_vals +n1stNeuron +n) += *(biases +n1stBias +n);
    *(N->iIGates +n1stCell +n) += *(oldstates +n1stCell +n) * *(weights +n1stPeep +3*n)    + *(biases +n1stBiasIG +n);
    *(N->iFGates +n1stCell +n) += *(oldstates +n1stCell +n) * *(weights +n1stPeep +3*n +1) + *(biases +n1stBiasFG +n);
    
    *(N->oMCell  +n1stCell +n) = ifun->eval(*(N->in_vals +n1stNeuron +n));
    *(N->oIGates +n1stCell +n) = sigm->eval(*(N->iIGates +n1stCell   +n));
    *(N->oFGates +n1stCell +n) = sigm->eval(*(N->iFGates +n1stCell   +n));
    
    *(N->ostates +n1stCell +n) = *(oldstates +n1stCell +n) * *(N->oFGates +n1stCell +n) +
    *(N->oMCell  +n1stCell +n) * *(N->oIGates +n1stCell +n);
    
    *(N->iOGates +n1stCell +n) += *(N->ostates +n1stCell +n) * *(weights +n1stPeep +3*n +2) + *(biases +n1stBiasOG +n);
    *(N->oOGates +n1stCell +n) = sigm->eval(*(N->iOGates +n1stCell +n) );
    *(N->outvals +n1stNeuron +n) = ofun->eval(*(N->ostates +n1stCell +n)) * *(N->oOGates +n1stCell +n);
}

KER1 void NormalLayer::updateGrads(const int n, Lab * N, Real * oldvals, Real * oldstates, Dsdw * dsdw, Grads * grad)
{
    *(grad->_B +n1stBias +n) = *(N->errvals +n1stNeuron +n);
    for (const auto & l : *curr_input_links)
    {
        #ifdef SIMDKERNELSG
        if (l->first==false)
            updateGradsSIMD(n, l, N, N->outvals, grad->_W, dsdw);
        else
        #endif
            updateGrads(n, l, N, N->outvals, grad->_W, dsdw);
    }
}

KER1 void LSTMLayer::updateGrads(  const int n, Lab * N, Real * oldvals, Real * oldstates, Dsdw * dsdw, Grads * grad)
{
    *(dsdw->DB +n1stdSdB +5*n)    = *(dsdw->DB +n1stdSdB +n*5)    * *(N->oFGates +n1stCell +n)
                                                                  + *(N->eMCell  +n1stCell +n);
    *(grad->_B +n1stBias   +n)    = *(dsdw->DB +n1stdSdB +n*5)    * *(N->errvals +n1stNeuron +n);
    
    *(dsdw->DB +n1stdSdB +5*n +1) = *(dsdw->DB +n1stdSdB +n*5 +1) * *(N->oFGates +n1stCell +n)
                                                                  + *(N->eIGates +n1stCell +n);
    *(grad->_B +n1stBiasIG +n)    = *(dsdw->DB +n1stdSdB +n*5 +1) * *(N->errvals +n1stNeuron +n);
    
    *(dsdw->DB +n1stdSdB +5*n +2) = *(dsdw->DB +n1stdSdB +n*5 +2) * *(N->oFGates +n1stCell +n)
                                                                  + *(N->eFGates +n1stCell +n);
    *(grad->_B +n1stBiasFG +n)    = *(dsdw->DB +n1stdSdB +n*5 +2) * *(N->errvals +n1stNeuron +n);
    
    *(grad->_B +n1stBiasOG +n)    = *(N->eOGates+n1stCell+n);
    
    *(dsdw->DB +n1stdSdB +5*n +3) = *(dsdw->DB +n1stdSdB +n*5 +3) * *(N->oFGates +n1stCell +n)
                                      + *(oldstates +n1stCell +n) * *(N->eIGates +n1stCell +n);
    *(grad->_W +n1stPeep +3*n)    = *(dsdw->DB +n1stdSdB +n*5 +3) * *(N->errvals +n1stNeuron +n);
    
    *(dsdw->DB +n1stdSdB +5*n +4) = *(dsdw->DB +n1stdSdB +n*5 +4) * *(N->oFGates +n1stCell +n)
                                      + *(oldstates +n1stCell +n) * *(N->eFGates +n1stCell +n);
    *(grad->_W +n1stPeep +3*n +1) = *(dsdw->DB +n1stdSdB +n*5 +4) * *(N->errvals +n1stNeuron +n);
    
    *(grad->_W +n1stPeep +3*n +2) =    *(N->ostates +n1stCell +n) * *(N->eOGates+n1stCell +n);
    
    for (const auto & l : *curr_input_links)
    {
        #ifdef SIMDKERNELSG
        if (l->first==false)
            updateGradsSIMD(n, l, N, N->outvals, grad->_W, dsdw);
        else
        #endif
            updateGrads(n, l, N, N->outvals, grad->_W, dsdw);
    }
    for (const auto & l : *prev_input_links)
    {
        #ifdef SIMDKERNELSG
        updateGradsSIMD(n, l, N, oldvals, grad->_W, dsdw);
        #else
        updateGrads(n, l, N, oldvals, grad->_W, dsdw);
        #endif
    }
}

KER1 void LSTMLayer::updateGradsLight(  const int n, Lab * N, Real * oldvals, Real * oldstates, Grads * grad)
{
    *(grad->_B +n1stBias   +n)    = *(N->eMCell  +n1stCell +n) * *(N->errvals +n1stNeuron +n);

    *(grad->_B +n1stBiasIG +n)    = *(N->eIGates +n1stCell +n) * *(N->errvals +n1stNeuron +n);

    *(grad->_B +n1stBiasFG +n)    = *(N->eFGates +n1stCell +n) * *(N->errvals +n1stNeuron +n);
    
    *(grad->_B +n1stBiasOG +n)    = *(N->eOGates+n1stCell+n);
    
    *(grad->_W +n1stPeep +3*n)    = *(oldstates +n1stCell +n) * *(N->eIGates +n1stCell +n) * *(N->errvals +n1stNeuron +n);
    
    *(grad->_W +n1stPeep +3*n +1) = *(oldstates +n1stCell +n) * *(N->eFGates +n1stCell +n) * *(N->errvals +n1stNeuron +n);
    
    *(grad->_W +n1stPeep +3*n +2) = *(N->ostates +n1stCell +n) * *(N->eOGates+n1stCell +n);
    
    for (const auto & l : *curr_input_links)
    {
        #ifdef SIMDKERNELSG
        if (l->first==false)
            updateGradsSIMDLight(n, l, N, N->outvals, grad->_W);
        else
        #endif
            updateGradsLight(n, l, N, N->outvals, grad->_W);
    }
    for (const auto & l : *prev_input_links)
    {
        #ifdef SIMDKERNELSG
        updateGradsSIMDLight(n, l, N, oldvals, grad->_W);
        #else
        updateGradsLight(n, l, N, oldvals, grad->_W);
        #endif
    }
}

KER2 void NormalLayer::addInputs(Real* tI, const int n, Link *l, Real * outvals, Real* weights)
{
    for (int i=0; i<l->nI; i++)
        *tI += *(outvals +l->iI +i) * *(weights +l->iW +n*l->nI +i);
}

KER2 void LSTMLayer::addInputs(Real* tC, Real* tI, Real* tF, Real* tO, const int n, Link *l, Real * outvals, Real* weights)
{
    for (int i=0; i<l->nI; i++)
    {
        *tC += *(outvals +l->iI +i) * *(weights +l->iW  +n*l->nI +i);
        *tI += *(outvals +l->iI +i) * *(weights +l->iWI +n*l->nI +i);
        *tF += *(outvals +l->iI +i) * *(weights +l->iWF +n*l->nI +i);
        *tO += *(outvals +l->iI +i) * *(weights +l->iWO +n*l->nI +i);
    }
}

#ifdef SIMDKERNELSIN
KER2 void NormalLayer::addInputsSIMD(vec & IN, const int n, Link *l, Real * outvals, Real* weights)
{
    assert(l->nI%SIMD == 0);

    for (int i=0; i<l->nI; i+=SIMD)
    {
        //FETCH((char*) outvals +l->iI +i + M_PF_O, M_POL_O);
        //FETCH((char*) weights +l->iW  +n*l->nI +i + M_PF_W, M_POL_W);
        IN = ADD(IN, MUL(LOAD(outvals +l->iI +i), LOAD(weights +l->iW  +n*l->nI +i)));
    }
    
    
}

KER2 void LSTMLayer::addInputsSIMD(vec & IN, vec & IG, vec & FG, vec & OG, const int n, Link *l, Real * outvals, Real* weights)
{
    assert(l->nI%SIMD == 0);
    
    for (int i=0; i<l->nI; i+=SIMD)
    {
        //FETCH((char*) outvals +l->iI +i + M_PF_O, M_POL_O);
        //FETCH((char*) weights +l->iW  +n*l->nI +i + M_PF_W, M_POL_W);
        //FETCH((char*) weights +l->iWI +n*l->nI +i + M_PF_W, M_POL_W);
        //FETCH((char*) weights +l->iWF +n*l->nI +i + M_PF_W, M_POL_W);
        //FETCH((char*) weights +l->iWO +n*l->nI +i + M_PF_W, M_POL_W);
        
        vec O = LOAD(outvals +l->iI +i);
        IN = ADD(IN, MUL(O, LOAD(weights +l->iW  +n*l->nI +i)));
        IG = ADD(IG, MUL(O, LOAD(weights +l->iWI +n*l->nI +i)));
        FG = ADD(FG, MUL(O, LOAD(weights +l->iWF +n*l->nI +i)));
        OG = ADD(OG, MUL(O, LOAD(weights +l->iWO +n*l->nI +i)));
    }
}
#endif

KER2 void NormalLayer::updateGrads(    const int n, Link *l, Lab * N, Real * outvals, Real* gradW, Dsdw * dsdw)
{
    for (int i=0; i<l->nI; i++)
        *(gradW +l->iW +n*l->nI +i) = *(outvals +l->iI +i) * *(N->errvals +l->iO +n);
    //printf("input errval %d is %f\n",l->iO +n, *(N->errvals +l->iO +n));
}

#ifdef SIMDKERNELSG
KER2 void NormalLayer::updateGradsSIMD(const int n, Link *l, Lab * N, Real * outvals, Real* gradW, Dsdw * dsdw)
{
    const vec E = BCAST(N->errvals +l->iO +n);
    for (int i=0; i<l->nI; i+=SIMD)
    {
        FETCH((char*) outvals +l->iI +i + M_PF_O, M_POL_O);
        STORE(gradW +l->iW +n*l->nI +i, MUL(E, LOAD(outvals +l->iI +i)));
    }
}

KER2 void LSTMLayer::updateGradsSIMD(  const int n, Link *l, Lab * N, Real * outvals, Real* gradW, Dsdw * dsdw)
{
    const vec FG = BCAST(N->oFGates+l->iC+n);
    const vec EN = BCAST(N->eMCell +l->iC+n);
    const vec EI = BCAST(N->eIGates+l->iC+n);
    const vec EF = BCAST(N->eFGates+l->iC+n);
    const vec EO = BCAST(N->eOGates+l->iC+n);
    const vec EC = BCAST(N->errvals+l->iO+n);
    
    for (int i=0; i<l->nI; i+=SIMD)
    {
        FETCH((char*) outvals +l->iI +i + M_PF_O, M_POL_O);
        
        FETCH((char*) dsdw->IN +l->idSdW +n*l->nI +i+ M_PF_DS, M_POL_DS);
        FETCH((char*) dsdw->IG +l->idSdW +n*l->nI +i+ M_PF_DS, M_POL_DS);
        FETCH((char*) dsdw->FG +l->idSdW +n*l->nI +i+ M_PF_DS, M_POL_DS);
        
        const vec O  = LOAD(outvals +l->iI +i);
        { //Cells input
            vec DSDW = ADD(MUL(LOAD(dsdw->IN +l->idSdW +n*l->nI +i),FG), MUL(EN,O));
            STORE(gradW +l->iW  +n*l->nI +i, MUL(DSDW, EC));
            STORE(dsdw->IN +l->idSdW +n*l->nI +i, DSDW);
        }
        { //Input gate
            vec DSDW = ADD(MUL(LOAD(dsdw->IG +l->idSdW +n*l->nI +i),FG), MUL(EI,O));
            STORE(gradW +l->iWI +n*l->nI +i, MUL(DSDW, EC));
            STORE(dsdw->IG +l->idSdW +n*l->nI +i, DSDW);
        }
        { //Forget gate
            vec DSDW = ADD(MUL(LOAD(dsdw->FG +l->idSdW +n*l->nI +i),FG), MUL(EF,O));
            STORE(gradW +l->iWF +n*l->nI +i, MUL(DSDW, EC));
            STORE(dsdw->FG +l->idSdW +n*l->nI +i, DSDW);
        }
        { //Output gate
            STORE(gradW +l->iWO +n*l->nI +i, MUL(EO, O));
        }
    }
}

KER2 void LSTMLayer::updateGradsSIMDLight(const int n, Link *l, Lab * N, Real * outvals, Real* gradW)
{
    const vec EN = BCAST(N->eMCell +l->iC+n);
    const vec EI = BCAST(N->eIGates+l->iC+n);
    const vec EF = BCAST(N->eFGates+l->iC+n);
    const vec EO = BCAST(N->eOGates+l->iC+n);
    const vec EC = BCAST(N->errvals+l->iO+n);
    
    for (int i=0; i<l->nI; i+=SIMD)
    {
        //FETCH((char*) outvals +l->iI +i + M_PF_O, M_POL_O);
        
        const vec O  = LOAD(outvals +l->iI +i);
        { //Cells input
            STORE(gradW +l->iW  +n*l->nI +i, MUL(MUL(EN,O), EC));
        }
        { //Input gate
            STORE(gradW +l->iWI +n*l->nI +i, MUL(MUL(EI,O), EC));
        }
        { //Forget gate
            STORE(gradW +l->iWF +n*l->nI +i, MUL(MUL(EF,O), EC));
        }
        { //Output gate
            STORE(gradW +l->iWO +n*l->nI +i, MUL(EO, O));
        }
    }
}
#endif

KER2 void LSTMLayer::updateGrads(      const int n, Link *l, Lab * N, Real * outvals, Real* gradW, Dsdw * dsdw)
{
    for (int i=0; i<l->nI; i++)
    {
        *(dsdw->IN +l->idSdW +n*l->nI +i) = *(dsdw->IN +l->idSdW +n*l->nI +i) * *(N->oFGates +l->iC +n)
                                                       + *(outvals +l->iI +i) * *(N->eMCell  +l->iC +n);
        *(gradW +l->iW       +n*l->nI +i) = *(dsdw->IN +l->idSdW +n*l->nI +i) * *(N->errvals +l->iO +n);

        *(dsdw->IG +l->idSdW +n*l->nI +i) = *(dsdw->IG +l->idSdW +n*l->nI +i) * *(N->oFGates +l->iC +n)
                                                       + *(outvals +l->iI +i) * *(N->eIGates +l->iC +n);
        *(gradW +l->iWI      +n*l->nI +i) = *(dsdw->IG +l->idSdW +n*l->nI +i) * *(N->errvals +l->iO +n);

        *(dsdw->FG +l->idSdW +n*l->nI +i) = *(dsdw->FG +l->idSdW +n*l->nI +i) * *(N->oFGates +l->iC +n)
                                                       + *(outvals +l->iI +i) * *(N->eFGates +l->iC +n);
        *(gradW +l->iWF      +n*l->nI +i) = *(dsdw->FG +l->idSdW +n*l->nI +i) * *(N->errvals +l->iO +n);

        *(gradW +l->iWO      +n*l->nI +i) =              *(outvals +l->iI +i) * *(N->eOGates +l->iC +n);
    }
}

KER2 void LSTMLayer::updateGradsLight( const int n, Link *l, Lab * N, Real * outvals, Real* gradW)
{
    for (int i=0; i<l->nI; i++)
    {
        *(gradW +l->iW       +n*l->nI +i) = *(outvals +l->iI +i) * *(N->eMCell  +l->iC +n) * *(N->errvals +l->iO +n);

        *(gradW +l->iWI      +n*l->nI +i) = *(outvals +l->iI +i) * *(N->eIGates +l->iC +n) * *(N->errvals +l->iO +n);

        *(gradW +l->iWF      +n*l->nI +i) = *(outvals +l->iI +i) * *(N->eFGates +l->iC +n) * *(N->errvals +l->iO +n);

        *(gradW +l->iWO      +n*l->nI +i) = *(outvals +l->iI +i) * *(N->eOGates +l->iC +n);
    }
    //printf("input errval %d is %f\n",l->iO +n, *(N->errvals +l->iO +n));
}

KER2 void NormalLayer::addErrors(const int n, Link *l, Lab * N, Real * errvals, Real* weights)
{
    Real err(0.);
    if (l->LSTM)
        for (int i=0; i<l->nO; i++)
        {
            //printf("input error %d is %f\n",l->iO +i, *(N->errvals +l->iO +i));
            err += *(N->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +n) + *(N->errvals +l->iO +i) * (
                                         *(N->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +n) +
                                         *(N->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +n) +
                                         *(N->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +n) );
        }
    else
        for (int i=0; i<l->nO; i++)
        {
            //printf("input error %d is %f\n",l->iO +i, *(N->errvals +l->iO +i));
            err  += *(N->errvals +l->iO +i) * *(weights +l->iW +i*l->nI +n);
        }
    
    *(errvals +l->iI +n) += err;
}
