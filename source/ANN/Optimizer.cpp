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

AdamOptimizer::AdamOptimizer(Network * _net, Profiler * _prof, Settings  & settings) : eta(settings.lRate), beta_1(0.9), beta_2(0.999), epsilon(1e-8), lambda(settings.nnLambda), net(_net), profiler(_prof), nInputs(net->nInputs), nOutputs(net->nOutputs), iOutputs(net->iOutputs), nWeights(net->nWeights), nBiases(net->nBiases), beta_t_1(0.9), beta_t_2(0.999)
{
    //batchsize=0;
    nepoch=0;
    _allocateClean(_1stMomW, nWeights)
    init(_1stMomW, nWeights);
    _allocateClean(_1stMomB, nBiases)
    init(_1stMomB, nBiases);
    
    _allocateClean(_2ndMomW, nWeights)
    init(_2ndMomW, nWeights);
    _allocateClean(_2ndMomB, nBiases)
    init(_2ndMomB, nBiases);
}

void AdamOptimizer::trainBatch(const vector<const vector<Real>*>& inputs, const vector<const vector<Real>*>& targets, Real & trainMSE)
{/*
    trainMSE = 0.;
    int nseries = inputs.size();
    vector<Real> res(nOutputs);
    Grads * g = new Grads(nWeights,nBiases);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    
    net->clearDsdw();
    for (int k=0; k<nseries; k++) {
        profiler->start("F");
        net->predict(*(inputs[k]), res, net->series[0], net->series[1]);
        profiler->stop("F");
        
        profiler->start("B");
        for (int j =0; j<nOutputs; j++) {
            res[j] = (*(targets[k]))[j] - res[j];
            trainMSE += 0.5*res[j]*res[j];
        }
        
        net->computeGrads(res, net->series[0], net->series[1], g);
        profiler->stop("B");
        
        profiler->start("S");
        stackGrads(net->grad,g);
        profiler->stop("S");
        
        if (nseries>1)
            std::swap(net->series[1], net->series[0]);
    }

    profiler->start("W");
    update(net->grad);
    profiler->stop("W");
    
    trainMSE /= (Real)inputs.size();
    delete g;*/
}

void AdamOptimizer::trainSeries(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE)
{/*
    trainMSE = 0.0;
    vector<Real> res;
    int nseries = inputs.size();
    net->allocateSeries(nseries+1);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    Grads * g = new Grads(nWeights,nBiases);

    //STEP 1: go through the data to compute predictions
    profiler->start("F");
    for (int k=0; k<nseries; k++) {
        net->predict(inputs[k], res, net->series[k], net->series[k+1]);
        for (int i=0; i<nOutputs; i++) {
            Real err = targets[k][i]- *(net->series[k+1]->outvals+iOutputs+i);
            *(net->series[k+1]->errvals +iOutputs+i) = err;
            trainMSE += 0.5*err*err;
        }
    }
    profiler->stop("F");
    
    profiler->start("B");
    //STEP 2: go backwards to backpropagate deltas (errors)
    net->computeDeltasEnd(net->series, nseries);
    for (int k=nseries-1; k>=1; k--)
        net->computeDeltasSeries(net->series, k);
    profiler->stop("B");
    
    //STEP 3: go ahead again to compute the gradients with eligibility trace (dsdw(t) depends on dsdw(t-1))
    profiler->start("G");
    for (int k=1; k<=nseries; k++) {
        net->computeGradsLightSeries(net->series, k, g);
        stackGrads(net->grad,g);
    }
    profiler->stop("G");
    
    //STEP 4: finally update the weights
    profiler->start("W");
    update(net->grad);
    profiler->stop("W");

    //printf("%f\n",trainMSE);
    trainMSE /= (Real)inputs.size();
    delete g; */
}

void AdamOptimizer::checkGrads(const vector<vector<Real>>& inputs, const int lastn, const int ierr)
{
    std::cout << std::setprecision(9);
    int nseries = inputs.size();
    vector<Real> res;
    net->allocateSeries(nseries+1);
    
    Grads * g = new Grads(nWeights,nBiases);
    Grads * G = new Grads(nWeights,nBiases);
    const Real eps = 1e-6;
    
    net->predict(inputs[0], res, net->series[0]);
    for (int k=1; k<lastn; k++) {
        net->predict(inputs[k], res, net->series[k-1], net->series[k]);
        for (int i=0; i<nOutputs; i++)
            *(net->series[k]->errvals +iOutputs+i) = 0.;
    }

    *(net->series[lastn]->errvals +iOutputs+ierr) = -1.;//Errors[1*nOutputs + i];
    
    net->computeDeltasSeries(net->series, 0, lastn-1);
    
    for (int k=0; k<=lastn-2; k++) {
        net->computeGradsSeries(net->series, k, g);
        stackGrads(G,g);
    }

    for (int w=0; w<nWeights; w++) {
        *(g->_W+w) = 0;
        
        *(net->weights+w) += eps;
        net->predict(inputs[0], res, net->series[0]);
        for (int k=1; k<lastn; k++)
            net->predict(inputs[k], res, net->series[k-1], net->series[k]);
        const Real out1 = - *(net->series[lastn-1]->outvals+iOutputs+ierr);
        
        *(net->weights+w) -= 2*eps;
        net->predict(inputs[0], res, net->series[0]);
        for (int k=1; k<lastn; k++)
            net->predict(inputs[k], res, net->series[k-1], net->series[k]);
        const Real out2 = - *(net->series[lastn-1]->outvals+iOutputs+ierr);
        
        *(net->weights+w) += eps;
        *(g->_W+w) += (out1-out2)/(2*eps);
        
        //const Real scale = fabs(*(net->biases+w));
        const Real scale = max(fabs(*(G->_W+w)),fabs(*(g->_W+w)));
        const Real err = (*(G->_W+w)-*(g->_W+w))/scale;
        if (fabs(err)>1e-6) cout <<"W"<<w<<" "<<*(G->_W+w)<<" "<<*(g->_W+w)<<" "<<err<<endl;
    }
    
    for (int w=0; w<nBiases; w++) {
        *(g->_B+w) = 0;
        
        *(net->biases+w) += eps;
        net->predict(inputs[0], res, net->series[0]);
        for (int k=1; k<lastn; k++)
            net->predict(inputs[k], res, net->series[k-1], net->series[k]);
        const Real out1 = - *(net->series[lastn-1]->outvals+iOutputs+ierr);
        
        *(net->biases+w) -= 2*eps;
        net->predict(inputs[0], res, net->series[0]);
        for (int k=1; k<lastn; k++)
            net->predict(inputs[k], res, net->series[k-1], net->series[k]);
        const Real out2 = - *(net->series[lastn-1]->outvals+iOutputs+ierr);
        
        *(net->biases+w) += eps;
        *(g->_B+w) += (out1-out2)/(2*eps);
        
        //const Real scale = fabs(*(net->biases+w));
        const Real scale = max(fabs(*(G->_B+w)),fabs(*(g->_B+w)));
        const Real err = (*(G->_B+w)-*(g->_B+w))/scale;
        if (fabs(err)>1e-6) cout <<"B"<<w<<" "<<*(G->_B+w)<<" "<<*(g->_B+w)<<" "<<err<<endl;
    }
    printf("\n\n\n");
    abort();
}

void AdamOptimizer::addUpdate(Grads* const G)
{
    update(net->weights, G->_W, _1stMomW, _2ndMomW, nWeights, eta);
    update(net->biases,  G->_B, _1stMomB, _2ndMomB, nBiases, eta);

    beta_t_1 *= beta_1;
    beta_t_2 *= beta_2;
}

void AdamOptimizer::stackGrads(Grads* const G, const Grads* const g) const
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


void AdamOptimizer::stackGrads(Grads* const G, const vector<Grads*> g)
{
    const int WsizeSIMD=ceil(nWeights/(Real)SIMD)*SIMD;
    const int BsizeSIMD=ceil(nBiases/(Real)SIMD)*SIMD;
    const int nThreads =omp_get_num_threads();
    #if SIMD > 1
    const vec zeros = SET0 ();
    #endif
    
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

void AdamOptimizer::update(Grads* const G, const int batchsize)
{
    const Real etaBatch = (exp(-nepoch/200.) + eta)/Real(max(batchsize,1));
    const int WsizeSIMD=ceil(nWeights/(Real)SIMD)*SIMD;
    const int BsizeSIMD=ceil(nBiases/(Real)SIMD)*SIMD;
    
    if (lambda>1e-9) {
        updateDecay(net->weights, G->_W, _1stMomW, _2ndMomW, WsizeSIMD, etaBatch);
        updateDecay(net->biases,  G->_B, _1stMomB, _2ndMomB, BsizeSIMD, etaBatch);
    } else {
        update(net->weights, G->_W, _1stMomW, _2ndMomW, WsizeSIMD, etaBatch);
        update(net->biases,  G->_B, _1stMomB, _2ndMomB, BsizeSIMD, etaBatch);
    }
    //batchsize=0;
    beta_t_1 *= beta_1;
    beta_t_2 *= beta_2;
    //nepoch++;
}

void AdamOptimizer::update(Real* const dest, Real* const grad, Real* const _1stMom, Real* const _2ndMom, const int N, const Real _eta)
{
    Real fac12 = _eta*sqrt(1.-beta_t_2)/(1.-beta_t_1);
    #if SIMD > 1
    const vec B1 = SET1(beta_1);
    const vec B2 = SET1(beta_2);
    const vec _B1 =SET1(1.-beta_1);
    const vec _B2 =SET1(1.-beta_2);
    const vec F12 = SET1(_eta*sqrt(1.-beta_t_2)/(1.-beta_t_1));
    const vec EPS = SET1(epsilon);
    const vec zeros = SET0 ();
    #endif
    
    #pragma omp for nowait
    for (int i=0; i<N; i+=SIMD) {
        #if SIMD == 1
        *(_1stMom + i) = beta_1 * *(_1stMom + i) + (1.-beta_1) * *(grad + i);
        *(_2ndMom + i) = beta_2 * *(_2ndMom + i) + (1.-beta_2) * *(grad + i) * *(grad + i);

        *(grad + i) = 0.; //reset grads
        *(dest + i) += fac12 * *(_1stMom + i)  / sqrt(*(_2ndMom + i) + epsilon);
        #else
        vec _DW = LOAD(grad + i);
        //FETCH((char*)    grad +i + M_PF_G, M_POL_G);
        vec M1 = ADD( MUL ( B1, LOAD(_1stMom + i)), MUL ( _B1, _DW));
        //FETCH((char*) _1stMom +i + M_PF_G, M_POL_G);
        vec M2 = ADD( MUL ( B2, LOAD(_2ndMom + i)), MUL ( _B2, MUL (_DW,_DW)));
        //FETCH((char*) _2ndMom +i + M_PF_G, M_POL_G);
        STORE(dest+i, ADD( LOAD(dest+i), MUL( MUL(F12, M1), RSQRT(ADD(M2,EPS)) )));
        //FETCH((char*)    dest +i + M_PF_G, M_POL_G);
        
        STORE(_1stMom + i,M1);
        STORE(_2ndMom + i,M2);
        STORE (grad+i,zeros); //reset grads
        #endif
    }
}

void AdamOptimizer::updateDecay(Real* const dest, Real* const grad, Real* const _1stMom, Real* const _2ndMom, const int N, const Real _eta)
{
    Real fac12 = _eta*sqrt(1.-beta_t_2)/(1.-beta_t_1);
    #if SIMD > 1
    const vec B1 = SET1(beta_1);
    const vec B2 = SET1(beta_2);
    const vec _B1 =SET1(1.-beta_1);
    const vec _B2 =SET1(1.-beta_2);
    const vec F12 = SET1(_eta*sqrt(1.-beta_t_2)/(1.-beta_t_1));
    const vec EPS = SET1(epsilon);
    const vec LAMBDA = SET1(-lambda*_eta);
    const vec zeros = SET0 ();
    #endif
    
    #pragma omp for nowait
    for (int i=0; i<N; i+=SIMD) {
        #if SIMD == 1
        *(_1stMom + i) = beta_1 * *(_1stMom + i) + (1.-beta_1) * *(grad + i);
        *(_2ndMom + i) = beta_2 * *(_2ndMom + i) + (1.-beta_2) * *(grad + i) * *(grad + i);

        *(grad + i) = 0.; //reset grads
        *(dest + i) += fac12 * *(_1stMom + i)/sqrt(*(_2ndMom + i) + epsilon) -*(dest + i)*lambda*_eta;
        #else
        vec _DW = LOAD(grad + i);
        //FETCH((char*)    grad +i + M_PF_G, M_POL_G);
        vec M1 = ADD( MUL ( B1, LOAD(_1stMom + i)), MUL ( _B1, _DW));
        //FETCH((char*) _1stMom +i + M_PF_G, M_POL_G);
        vec M2 = ADD( MUL ( B2, LOAD(_2ndMom + i)), MUL ( _B2, MUL (_DW,_DW)));
        //FETCH((char*) _2ndMom +i + M_PF_G, M_POL_G);
        vec W = LOAD(dest + i);
        STORE(dest+i,ADD(W,ADD(MUL(MUL(F12, M1),RSQRT(ADD(M2,EPS))),MUL(LAMBDA,W))));
        //FETCH((char*)    dest +i + M_PF_G, M_POL_G);
        
        STORE(_1stMom + i,M1);
        STORE(_2ndMom + i,M2);
        STORE (grad+i,zeros); //reset grads
        #endif
    }
}

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

void AdamOptimizer::update(Real* const dest, Real* const grad, const int N, const Real _eta) const
{
    #if SIMD > 1
    const vec ETAB = SET1( _eta );
    const vec zeros = SET0 ();
    #endif
    
    #pragma omp for nowait
    for (int i=0; i<N; i+=SIMD)
    {
        #if SIMD == 1
        *(dest + i) += _eta * *(grad + i);
        *(grad + i) = 0.; //reset grads
        #else
        STORE(dest+i, ADD( LOAD(dest+i), MUL(ETAB, LOAD(grad + i))));
        STORE(grad+i,zeros); //reset grads
        #endif
    }
}

void AdamOptimizer::init(Real* const dest, const int N, const Real ini)
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


