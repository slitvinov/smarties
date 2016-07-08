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

Optimizer::Optimizer(Network * _net, Profiler * _prof, Settings  & settings) : eta(settings.lRate), lambda(settings.nnLambda), alpha(0.5), net(_net), profiler(_prof), nInputs(_net->nInputs), nOutputs(_net->nOutputs), iOutputs(_net->iOutputs), nWeights(_net->nWeights), nBiases(_net->nBiases), nepoch(0)//,batchsize(0)
{
    _allocateClean(_1stMomW, nWeights)
    _allocateClean(_1stMomB, nBiases)
}

AdamOptimizer::AdamOptimizer(Network * _net, Profiler * _prof, Settings  & settings) : Optimizer(_net, _prof, settings), beta_1(0.9), beta_2(0.999), epsilon(1e-9), beta_t_1(0.9), beta_t_2(0.999)
{
    _allocateClean(_2ndMomW, nWeights)
    _allocateClean(_2ndMomB, nBiases)
}

void Optimizer::stackGrads(Grads* const G, const Grads* const g) const
{
    //batchsize++;
    const int WsizeSIMD=ceil(nWeights/(Real)SIMD)*SIMD;
    for (int j=0; j<WsizeSIMD; j+=SIMD) {
        #if SIMD==1
        *(G->_W + j) += *(g->_W + j);
        #else
        STORE (G->_W + j, ADD (LOAD(G->_W + j), LOAD(g->_W + j)));
        #endif
    }
    
    const int BsizeSIMD=ceil(nBiases/(Real)SIMD)*SIMD;
    for (int j=0; j<BsizeSIMD; j+=SIMD) {
        #if SIMD==1
        *(G->_B + j) += *(g->_B + j);
        #else
        STORE (G->_B + j, ADD (LOAD(G->_B + j), LOAD(g->_B + j)));
        #endif
    }
}

void Optimizer::stackGrads(Grads* const G, const vector<Grads*> g) const
{
    const int WsizeSIMD=ceil(nWeights/(Real)SIMD)*SIMD;
    const int BsizeSIMD=ceil(nBiases/(Real)SIMD)*SIMD;
    const int nThreads =g.size();
    #if SIMD > 1
    const vec zeros = SET0 ();
    #endif
    #pragma omp barrier
    #pragma omp for nowait
    for (int j=0; j<WsizeSIMD; j+=SIMD) {
        for (int k=0; k<nThreads; k++) {
            #if SIMD==1
            *(G->_W+j) += *(g[k]->_W+j);
            *(g[k]->_W+j) = 0.;
            #else
            STORE(G->_W+j, ADD (LOAD(G->_W+j), LOAD(g[k]->_W+j)));
            STORE(g[k]->_W+j,zeros); //reset grads
            #endif
        }
    }
    #pragma omp for
    for (int j=0; j<BsizeSIMD; j+=SIMD) {
        for (int k=0; k<nThreads; k++) {
            #if SIMD==1
            *(G->_B+j) += *(g[k]->_B+j);
            *(g[k]->_B+j) = 0.;
            #else
            STORE(G->_B+j, ADD (LOAD(G->_B+j), LOAD(g[k]->_B+j)));
            STORE(g[k]->_B+j,zeros); //reset grads
            #endif
        }
    }
}

void Optimizer::stackGrads(const int thrID, Grads* const G, const vector<Grads*> g) const
{
    const int WsizeSIMD=ceil(nWeights/(Real)SIMD)*SIMD;
    const int BsizeSIMD=ceil(nBiases/(Real)SIMD)*SIMD;
    const int nThreads =g.size();
    #if SIMD > 1
    const vec zeros = SET0 ();
    #endif
    #pragma omp barrier
    
    vector<int> bndsW(nThreads+1), bndsB(nThreads+1);
    for (int k=1; k<nThreads; k++){
        bndsW[k] = ceil(k*WsizeSIMD/Real(nThreads)/(Real)SIMD)*SIMD;
        bndsB[k] = ceil(k*BsizeSIMD/Real(nThreads)/(Real)SIMD)*SIMD;
    }
    bndsW.back() = WsizeSIMD; bndsB.back() = BsizeSIMD;
        
    for (int k=0; k<nThreads; k++) {
        const int beg = (k  +thrID)%nThreads;
        const int end = (beg+1==nThreads)?nThreads:(k+1+thrID)%nThreads;
        
        for (int j=bndsW[beg]; j<bndsW[end]; j++) { //j+=SIMD
        #if SIMD==1
            *(G->_W+j) += *(g[thrID]->_W+j);
            *(g[thrID]->_W+j) = 0.;
        #else
            STORE(G->_W+j, ADD (LOAD(G->_W+j), LOAD(g[k]->_W+j)));
            STORE(g[k]->_W+j,zeros); //reset grads
        #endif
        }
        
        for (int j=bndsB[beg]; j<bndsB[end]; j++) {
        #if SIMD==1
            *(G->_B+j) += *(g[thrID]->_B+j);
            *(g[thrID]->_B+j) = 0.;
        #else
            STORE(G->_B+j, ADD (LOAD(G->_B+j), LOAD(g[k]->_B+j)));
            STORE(g[k]->_B+j,zeros); //reset grads
        #endif
        }
        #pragma omp barrier
    }
}

void Optimizer::update(Real* const dest, Real* const grad, Real* const _1stMom, const int N, const int batchsize) const
{
    const Real norm = 1./(Real)max(batchsize,1);
    const Real _eta = (0.0001*exp(-nepoch/100.)+eta)*norm;
    #if SIMD > 1
    const vec ETA = SET1( _eta );
    const vec ALPHA = SET1(alpha);
    const vec NORM = SET1(norm);
    const vec zeros = SET0();
    #endif
    
    #pragma omp for nowait
    for (int i=0; i<N; i+=SIMD)
    {
        #if SIMD == 1
        const Real W = fabs(*(dest + i));
        const Real M1 = alpha * *(_1stMom + i) + _eta * *(grad + i);
        *(_1stMom + i) = std::max(std::min(M1,W),-W);
        *(dest + i) += *(_1stMom + i);
        *(grad + i) = 0.; //reset grads
        if (W>100) printf("W %d %f\n",i,W);
        #else
        const vec W = LOAD(dest+i);
        const vec M1 = ADD(MUL(ALPHA, LOAD(_1stMom+i)), MUL(ETA, LOAD(grad+i)));
        const vec DW = MIN(MAX(W,SUB(zeros,W)),MAX(MIN(W,SUB(zeros,W)),M1));
        STORE(dest+i, ADD(LOAD(dest+i), DW));
        STORE(_1stMom + i,DW);
        STORE(grad+i,zeros); //reset grads
        #endif
    }
}

void Optimizer::updateDecay(Real* const dest, Real* const grad, Real* const _1stMom, const int N, const int batchsize) const
{
    const Real _eta = (0.0001*exp(-nepoch/100.)+eta);
    const Real norm = 1./(Real)max(batchsize,1);
    #if SIMD > 1
    const vec ETA = SET1( _eta );
    const vec ALPHA = SET1(alpha);
    const vec LAMBDA = SET1(-lambda*_eta);
    const vec NORM = SET1(norm);
    const vec zeros = SET0();
    #endif
    
    #pragma omp for nowait
    for (int i=0; i<N; i+=SIMD)
    {
        #if SIMD == 1
        const Real W = fabs(*(dest + i));
        const Real M1 = alpha * *(_1stMom + i) + _eta * *(grad + i);
        *(_1stMom + i) = std::max(std::min(M1,W),-W);
        *(dest + i) += *(_1stMom + i) - _eta*lambda * *(dest + i);
        *(grad + i) = 0.; //reset grads
        #else
        const vec W = LOAD(dest+i);
        const vec M1 = ADD(MUL(ALPHA, LOAD(_1stMom+i)), MUL(ETA, LOAD(grad+i)));
        const vec DW = MIN(MAX(W,SUB(zeros,W)),MAX(MIN(W,SUB(zeros,W)),M1));
        STORE(dest+i, ADD(W, ADD(DW, MUL(LAMBDA,W))));
        STORE(_1stMom + i,DW);
        STORE(grad+i,zeros); //reset grads
        #endif
    }
}

void Optimizer::update(Grads* const G, const int batchsize)
{
    const int WsizeSIMD=ceil(nWeights/(Real)SIMD)*SIMD;
    const int BsizeSIMD=ceil(nBiases/(Real)SIMD)*SIMD;
    
    if (lambda>1e-9) {
        updateDecay(net->weights, G->_W, _1stMomW, WsizeSIMD, max(batchsize,1));
        updateDecay(net->biases,  G->_B, _1stMomB, BsizeSIMD, max(batchsize,1));
    } else {
        update(net->weights, G->_W, _1stMomW, WsizeSIMD, max(batchsize,1));
        update(net->biases,  G->_B, _1stMomB, BsizeSIMD, max(batchsize,1));
    }
    //batchsize=0;
    //nepoch++;
    #pragma omp barrier
}

void AdamOptimizer::update(Grads* const G, const int batchsize)
{
    const int WsizeSIMD=ceil(nWeights/(Real)SIMD)*SIMD;
    const int BsizeSIMD=ceil(nBiases/(Real)SIMD)*SIMD;
    
    if (lambda>1e-9) {
        updateDecay(net->weights, G->_W, _1stMomW, _2ndMomW, WsizeSIMD, max(batchsize,1));
        updateDecay(net->biases,  G->_B, _1stMomB, _2ndMomB, BsizeSIMD, max(batchsize,1));
    } else {
        update(net->weights, G->_W, _1stMomW, _2ndMomW, WsizeSIMD, max(batchsize,1));
        update(net->biases,  G->_B, _1stMomB, _2ndMomB, BsizeSIMD, max(batchsize,1));
    }
    //batchsize=0;
    beta_t_1 *= beta_1;
    beta_t_2 *= beta_2;
    //nepoch++;
    #pragma omp barrier
}

void AdamOptimizer::update(Real* const dest, Real* const grad, Real* const _1stMom, Real* const _2ndMom, const int N, const int batchsize)
{
    const Real norm = 1./(Real)max(batchsize,1);
    const Real fac12 = sqrt(1.-beta_t_2)/(1.-beta_t_1);
    //const Real eta_ = (0.0001*exp(-nepoch/100.)+eta);
    const Real eta_ = eta;
    #if SIMD > 1
    const vec B1 = SET1(beta_1);
    const vec B2 = SET1(beta_2);
    const vec _B1 =SET1(1.-beta_1);
    const vec _B2 =SET1(1.-beta_2);
    const vec F12 = SET1(fac12*eta_);
    const vec EPS = SET1(epsilon);
    const vec NORM = SET1(norm);
    const vec zeros = SET0();
    #endif
    
    #pragma omp for nowait
    for (int i=0; i<N; i+=SIMD) {
    #if SIMD == 1
        const Real DW = *(grad+i) *norm;
        const Real M1 = beta_1* *(_1stMom+i) +(1.-beta_1)*DW;
        const Real M2 = beta_2* *(_2ndMom+i) +(1.-beta_2)*DW*DW;
        
        //slow down extreme updates (normalization)
        const Real M2_ = std::max(M2,epsilon);
        const Real TOP = std::fabs(*(dest+i)) * std::sqrt(M2_) / fac12;
        const Real M1_ = std::max(std::min(TOP,M1),-TOP);
        const Real DW_ = eta_*fac12*M1_/sqrt(M2_);
        //printf("TOP %d W %e M1 %e M2 %e DW %e TOP %e \n",i,std::fabs(*(dest+i)),M1_,M2_,DW_,TOP);
        *(_1stMom + i) = M1_;
        *(_2ndMom + i) = M2_;
        *(dest + i) += DW_;
        *(grad + i) = 0.; //reset grads

        #else
        const vec W  = LOAD(dest+i);
        const vec DW_ = MUL(LOAD(grad+i),NORM);
        const vec _DW = MIN(MAX(W,SUB(zeros,W)),MAX(MIN(W,SUB(zeros,W)),DW_));
        const vec M1 = ADD( MUL(B1, LOAD(_1stMom+i)), MUL(_B1, _DW));
        const vec M2 = MAX(ADD( MUL(B2, LOAD(_2ndMom+i)), MUL(_B2, MUL(_DW,_DW))),EPS);
        const vec DW = MUL(MUL(F12, M1),RSQRT(M2));
        STORE(dest+i, ADD(W, DW));
        STORE(_1stMom + i,M1);
        STORE(_2ndMom + i,M2);
        STORE (grad+i,zeros); //reset grads
        #endif
    }
}

void AdamOptimizer::updateDecay(Real* const dest, Real* const grad, Real* const _1stMom, Real* const _2ndMom, const int N, const int batchsize) const
{
    //begin with an hardcoded bigger eta, then anneal to user's eta, which should be <= 1e-5
    const Real norm = 1./(Real)max(batchsize,1);
    const Real fac12 = sqrt(1.-beta_t_2)/(1.-beta_t_1);
    //const Real eta_ = (0.0001*exp(-nepoch/100.)+eta);
    const Real eta_ = eta;
    #if SIMD > 1
    const vec B1 = SET1(beta_1);
    const vec B2 = SET1(beta_2);
    const vec _B1 =SET1(1.-beta_1);
    const vec _B2 =SET1(1.-beta_2);
    const vec F12 = SET1(fac12);
    const vec EPS = SET1(epsilon);
    const vec NORM = SET1(norm);
    const vec LAMBDA = SET1(-lambda*eta);
    const vec zeros = SET0 ();
    #endif
    
    #pragma omp for nowait
    for (int i=0; i<N; i+=SIMD) {
        #if SIMD == 1
        const Real DW = *(grad+i) *norm;
        const Real M1 = beta_1* *(_1stMom+i) +(1.-beta_1)*DW;
        const Real M2 = beta_2* *(_2ndMom+i) +(1.-beta_2)*DW*DW;
        
        //slow down extreme updates (normalization)
        const Real M2_ = std::max(M2,epsilon);
        //const Real TOP = std::fabs(*(dest+i)) * std::sqrt(M2_) / fac12;
        //const Real M1_ = std::max(std::min(TOP,M1),-TOP);
        const Real DW_ = eta_*fac12*M1/sqrt(M2_);
        //printf("TOP %d W %e M1 %e M2 %e DW %e TOP %e \n",i,std::fabs(*(dest+i)),M1_,M2_,DW_,TOP);
        *(_1stMom + i) = M1;
        *(_2ndMom + i) = M2_;
        *(dest + i) += DW_ - *(dest + i)*lambda*eta;
        *(grad + i) = 0.; //reset grads
        #else
        const vec W  = LOAD(dest+i);
        const vec DW_ = MUL(LOAD(grad+i),NORM);
        const vec _DW = MIN(MAX(W,SUB(zeros,W)),MAX(MIN(W,SUB(zeros,W)),DW_));
        const vec M1 = ADD( MUL(B1, LOAD(_1stMom+i)), MUL(_B1, _DW));
        const vec M2 = MAX(ADD( MUL(B2, LOAD(_2ndMom+i)), MUL(_B2, MUL(_DW,_DW))),EPS);
        const vec DW = MUL(MUL(F12, M1),RSQRT(M2));
        STORE(dest+i, ADD(ADD(W, DW),MUL(LAMBDA,W)));
        STORE(_1stMom + i,M1);
        STORE(_2ndMom + i,M2);
        STORE (grad+i,zeros); //reset grads
        #endif
    }
}

void Optimizer::init(Real* const dest, const int N, const Real ini)
{
    #if SIMD > 1
    const vec zeros = SET1 (ini);
    #endif
    
    for (int j=0; j<N; j+=SIMD)
    {
        #if SIMD == 1
        *(dest +j) = ini;
        #else
        STORE (dest +j,zeros);
        #endif
    }
    
    #if SIMD > 1
    for (int j=int(N/SIMD)*SIMD; j<N; ++j)
        *(dest +j) = ini;
    #endif
}

/*
void AdamOptimizer::stackGrads(Real* const G, const Real* const g, Real* const _1stMom, Real* const _2ndMom, const int N)
{
    Real fac12 = sqrt(1.-beta_t_2)/(1.-beta_t_1);
    #if SIMD > 1
    const vec B1  = SET1 (beta_1);
    const vec B2  = SET1 (beta_2);
    const vec _B1 = SET1(1.-beta_1);
    const vec _B2 = SET1(1.-beta_2);
    const vec F12 = SET1(sqrt(1.-beta_t_2)/(1.-beta_t_1));
    const vec EPS = SET1(epsilon);
    #endif
    
    #pragma omp for nowait
    for (int i=0; i<N; i+=SIMD)
    {
        #if SIMD == 1
        *(_1stMom + i) = beta_1 * *(_1stMom + i) + (1.-beta_1) * *(g + i);
        *(_2ndMom + i) = beta_2 * *(_2ndMom + i) + (1.-beta_2) * *(g + i) * *(g + i);
        *(G + i) += *(_1stMom + i) * fac12 / sqrt(*(_2ndMom + i) + epsilon);
        #else
        vec _g = LOAD(g + i);
        vec M1 = ADD( MUL ( B1, LOAD(_1stMom + i)), MUL ( _B1, _g));
        vec M2 = ADD( MUL ( B2, LOAD(_2ndMom + i)), MUL ( _B2, MUL (_g,_g)));
        
        STORE(G+i, ADD( LOAD(G+i), MUL(MUL(M1, F12), RSQRT(ADD(M2,EPS))) ));
        STORE(_1stMom + i,M1);
        STORE(_2ndMom + i,M2);
        #endif
    }
}
void Optimizer::addUpdate(Grads* const G)
{
    update(net->weights, G->_W, _1stMomW, _2ndMomW, nWeights, eta);
    update(net->biases,  G->_B, _1stMomB, _2ndMomB, nBiases, eta);

    beta_t_1 *= beta_1;
    beta_t_2 *= beta_2;
}*/

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


