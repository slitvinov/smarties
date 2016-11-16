/*
 *  Layers.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include "Optimizer.h"
#include <iomanip>      // std::setprecision
#include <iostream>     // std::cout, std::fixed
#include <cassert>

using namespace ErrorHandling;

Optimizer::Optimizer(Network * _net, Profiler * _prof, Settings  & settings) :
eta(settings.lRate), lambda(settings.nnLambda), alpha(0.5), net(_net), profiler(_prof),
nWeights(_net->getnWeights()), nBiases(_net->getnBiases()), nepoch(0)
{
    _allocateClean(_1stMomW, nWeights)
    _allocateClean(_1stMomB, nBiases)
}

AdamOptimizer::AdamOptimizer(Network * _net, Profiler * _prof, Settings  & settings) :
Optimizer(_net, _prof, settings), beta_1(0.9), beta_2(0.999), epsilon(1e-9),
beta_t_1(0.9), beta_t_2(0.999)
{
    _allocateClean(_2ndMomW, nWeights)
    _allocateClean(_2ndMomB, nBiases)
}

void Optimizer::stackGrads(Grads* const G, const Grads* const g) const
{
    for (int j=0; j<nWeights; j++) G->_W[j] += g->_W[j];
    for (int j=0; j<nBiases; j++)  G->_B[j] += g->_B[j];
}

void Optimizer::stackGrads(Grads* const G, const vector<Grads*> g) const
{
    const int nThreads = g.size();
    
    #pragma omp for nowait
    for (int j=0; j<nWeights; j++) 
    for (int k=1; k<nThreads; k++) {
        G->_W[j] += g[k]->_W[j];
        g[k]->_W[j] = 0.;
    }
    
    #pragma omp for
    for (int j=0; j<nBiases; j++) 
    for (int k=1; k<nThreads; k++) {
        G->_B[j] += g[k]->_B[j];
        g[k]->_B[j] = 0.;
    }
}

void Optimizer::stackGrads(const int thrID, Grads* const G, const vector<Grads*> g) const
{
    const int nThreads =g.size();
    
    vector<int> bndsW(nThreads+1), bndsB(nThreads+1);
    for (int k=1; k<nThreads; k++) {
        bndsW[k] = k*nWeights/Real(nThreads);
        bndsB[k] = k*nBiases/Real(nThreads);
    }
    bndsW.back() = nWeights; bndsB.back() = nBiases;
        
    for (int k=0; k<nThreads; k++) {
        const int beg = (k  +thrID)%nThreads;
        const int end = (beg+1==nThreads)?nThreads:(k+1+thrID)%nThreads;
        
        for (int j=bndsW[beg]; j<bndsW[end]; j++) {
            G->_W[j] += g[thrID]->_W[j];
            g[thrID]->_W[j] = 0.;
        }
        
        for (int j=bndsB[beg]; j<bndsB[end]; j++) {
            G->_B[j] += g[thrID]->_B[j];
            g[thrID]->_B[j] = 0.;
        }
        #pragma omp barrier
    }
}

void Optimizer::update(Grads* const G, const int batchsize)
{
    update(net->weights, G->_W, _1stMomW, nWeights, batchsize, lambda);
    update(net->biases,  G->_B, _1stMomB, nBiases, batchsize);
    #pragma omp barrier
}

void AdamOptimizer::update(Grads* const G, const int batchsize)
{
    update(net->weights, G->_W, _1stMomW, _2ndMomW, nWeights, batchsize, lambda);
    //Optimizer::update(net->biases,  G->_B, _1stMomB, nBiases, batchsize);
    update(net->biases,  G->_B, _1stMomB, _2ndMomB, nBiases, batchsize);
        
    #pragma omp barrier
    #pragma omp master
    {
        beta_t_1 *= beta_1;
        beta_t_2 *= beta_2;
    }
}

void Optimizer::update(Real* const dest, Real* const grad, Real* const _1stMom, const int N, const int batchsize, const Real _lambda) const
{
    const Real norm = 1./(Real)max(batchsize,1);
    const Real eta_ = eta*norm;
    const Real lambda_ = _lambda*eta;
    
    #pragma omp for nowait
    for (int i=0; i<N; i++) {
        const Real W = fabs(dest[i]);
        const Real M1 = alpha * _1stMom[i] + eta_ * grad[i];
        _1stMom[i] = std::max(std::min(M1,W),-W);
        grad[i] = 0.; //reset grads
        
        if (lambda_>0)
             dest[i] += _1stMom[i] + (dest[i]<0 ? lambda_ : -lambda_);
             //dest[i] += _1stMom[i] - dest[i]*lambda_;
        else dest[i] += _1stMom[i];
    }
}

void AdamOptimizer::update(Real* const dest, Real* const grad, Real* const _1stMom, Real* const _2ndMom, const int N, const int batchsize, const Real _lambda)
{
    const Real lambda_ = _lambda*eta;
    const Real norm = 1./(Real)max(batchsize,1);
    const Real eta_ = eta * sqrt(1.-beta_t_2)/(1.-beta_t_1);
    
    #pragma omp for nowait
    for (int i=0; i<N; i++) {
        const Real DW  = grad[i] *norm;
        const Real M1  = beta_1* _1stMom[i] +(1.-beta_1) *DW;
        const Real M2  = beta_2* _2ndMom[i] +(1.-beta_2) *DW*DW;
        const Real M1_ = std::min(std::max(M1,   -1e9),1e9);
        const Real M2_ = std::min(std::max(M2,epsilon),1e9);
        //slow down extreme updates (normalization):
        //const Real TOP = std::fabs(*(dest+i)) * std::sqrt(M2_) / fac12;
        //const Real M1_ = std::max(std::min(TOP,M1),-TOP);
        const Real DW_ = eta_ * M1_/sqrt(M2_);
        _1stMom[i] = M1_;
        _2ndMom[i] = M2_;
        grad[i] = 0.; //reset grads
        
        if (lambda_>0)
             dest[i] += DW_ + (dest[i]<0 ? lambda_ : -lambda_);
             //dest[i] += DW_ - dest[i]*lambda_;
        else dest[i] += DW_;
    }
}

void Optimizer::init(Real* const dest, const int N, const Real ini)
{
    for (int j=0; j<N; j++) dest[j] = ini;
}

/*
LMOptimizer::LMOptimizer(Network * _net, Profiler * _prof, Settings  & settings) : muMax(1e10), muMin(1e-6), muFactor(10), net(_net), profiler(_prof), nInputs(net->nInputs), nOutputs(net->nOutputs), iOutputs(net->iOutputs), nWeights(net->nWeights), nBiases(net->nBiases), totWeights(net->nWeights+net->nBiases), mu(0.1)
{
    dw.set_size(totWeights);
    Je.set_size(totWeights);
    diagJtJ.eye(totWeights, totWeights);
}

void LMOptimizer::stackGrads(Grads * g, const int k, const int i)
{
    #pragma omp parallel for nowait
    for (int j=0; j<nWeights; j++)
        J(i + k*nOutputs, j) = -*(g->_W + j);
    
    #pragma omp parallel for
    for (int j=0; j<nBiases; j++)
        J(i + k*nOutputs, j+nWeights) = -*(g->_B + j);
}

void LMOptimizer::tryNew()
{
    #pragma omp parallel for nowait
    for (int j=0; j<nWeights; j++)
        *(net->weights +j) += dw(j);
    
    #pragma omp parallel for
    for (int j=0; j<nBiases; j++)
        *(net->biases +j) += dw(j+nWeights);
}

void LMOptimizer::goBack()
{
    #pragma omp parallel for nowait
    for (int j=0; j<nWeights; j++)
        *(net->weights +j) -= dw(j);
    
    #pragma omp parallel for
    for (int j=0; j<nBiases; j++)
        *(net->biases +j) -= dw(j+nWeights);
}

void LMOptimizer::trainSeries(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE)
{
    trainMSE = 0.0;
    vector<Real> res;
    int nseries = inputs.size();
    net->allocateSeries(nseries+1);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    
    J.set_size(nOutputs*nseries, totWeights);
    e.set_size(nOutputs*nseries);
    
    #pragma omp parallel
    {
        //STEP 1: go through the data to compute predictions
        #pragma omp master
            profiler->start("F");
        
        for (int k=0; k<nseries; k++)
        {
            net->predict(inputs[k], res, net->series[k], net->series[k+1]);
            
            #pragma omp master
            for (int i=0; i<nOutputs; i++)
            { //put this loop here to slightly reduce overhead on second step
                Real err = *(net->series[k+1]->outvals+iOutputs+i) - targets[k][i];
                e(i + k*nOutputs) = err;
                *(net->series[k+1]->errvals +iOutputs+i) = 0.0;
                trainMSE += err*err;
            }
        }
        
        #pragma omp master
            profiler->stop("F");
        
        //STEP 2: go backwards to backpropagate deltas (errors)
        #pragma omp master
            profiler->start("B");
        
        net->clearErrors(net->series[nseries+1]); //there is a omp for in here
        for (int i=0; i<nOutputs; i++)
        {
            for (int k=nseries; k>=1; k--)
            {
                #pragma omp single
                for (int j=0; j<nOutputs; j++)
                    *(net->series[k]->errvals +iOutputs+i) = j==i;
                
                net->computeDeltasSeries(net->series, k);
            }
            
            net->clearDsdw();
            for (int k=1; k<=nseries; k++)
            {
                net->computeGradsSeries(net->series, k, net->grad);
                stackGrads(net->grad, k-1, i);
            }
        }
        #pragma omp master
            profiler->stop("B");
    }
    
    {
        Real Q = trainMSE+1.;
        
        JtJ = J.t() * J;
        Je  = J.t() * e;
        //diagJtJ = diagmat(JtJ);
        
        while (Q > trainMSE)
        {
            profiler->start("S");
            tmp = chol( JtJ + mu*diagJtJ );
            dw = solve(tmp, Je, arma::solve_opts::fast);
            profiler->stop("S");
            bool _nan = false;
            for (int w=0; w<totWeights; w++)
                if (std::isnan((dw(w))) || std::isinf((dw(w))))
                    _nan = true;
            if (_nan)
            {
                printf("Found nans :( \n");
                mu *= muFactor;
                Q = trainMSE+1.;
                continue;
            }
            //printf("Solved?\n");
            profiler->start("N");
            tryNew();
            profiler->stop("N");
            Q = 0;
            
            profiler->start("T");
            #pragma omp parallel
            for (int k=0; k<nseries; k++)
            {
                net->predict(inputs[k], res, net->series[k], net->series[k+1]);
                
                #pragma omp master
                for (int i=0; i<nOutputs; i++)
                { //put this loop here to slightly reduce overhead on second step
                    Real err = targets[k][i]- *(net->series[k+1]->outvals+iOutputs+i);
                    Q += err*err;
                }
            }
            profiler->stop("T");
            
            if (Q > trainMSE)
            {
                profiler->start("O");
                goBack();
                profiler->stop("O");
                
                printf("Nope \n");
                if (mu < muMax)
                    mu *= muFactor;
                else
                    break;
            }
            else
            printf("Yeap \n");
        }
        
        if (mu > muMin) mu /= muFactor;

    }

}
 */


