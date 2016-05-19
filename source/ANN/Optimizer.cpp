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

AdamOptimizer::AdamOptimizer(Network * _net, Profiler * _prof, Settings  & settings) : eta(settings.nnEta), beta_1(0.9), beta_2(0.999), epsilon(1e-8), lambda(settings.nnLambda), net(_net), profiler(_prof), nInputs(net->nInputs), nOutputs(net->nOutputs), iOutputs(net->iOutputs), nWeights(net->nWeights), nBiases(net->nBiases), beta_t_1(0.9), beta_t_2(0.999), batchsize(0), nepoch(1)
{
    _myallocate(_1stMomW, nWeights)
    init(_1stMomW, nWeights);
    _myallocate(_1stMomB, nBiases)
    init(_1stMomB, nBiases);
    
    _myallocate(_2ndMomW, nWeights)
    init(_2ndMomW, nWeights);
    _myallocate(_2ndMomB, nBiases)
    init(_2ndMomB, nBiases);
}

void AdamOptimizer::trainBatch(const vector<const vector<Real>*>& inputs, const vector<const vector<Real>*>& targets, Real & trainMSE)
{
    trainMSE = 0.;
    int nseries = inputs.size();
    vector<Real> res(nOutputs);
    Grads * g = new Grads(nWeights,nBiases);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    
    net->clearDsdw();
    for (int k=0; k<nseries; k++)
    {
        profiler->start("F");
        net->predict(*(inputs[k]), res, net->series[0], net->series[1]);
        profiler->stop("F");
        
        profiler->start("B");
        for (int j =0; j<nOutputs; j++)
        {
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
    delete g;
}

void AdamOptimizer::trainSeries(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE)
{
    trainMSE = 0.0;
    vector<Real> res;
    int nseries = inputs.size();
    net->allocateSeries(nseries+1);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    Grads * g = new Grads(nWeights,nBiases);

    //STEP 1: go through the data to compute predictions
    profiler->start("F");
    for (int k=0; k<nseries; k++)
    {
        net->predict(inputs[k], res, net->series[k], net->series[k+1]);
        for (int i=0; i<nOutputs; i++)
        { //put this loop here to slightly reduce overhead on second step
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
    for (int k=1; k<=nseries; k++)
    {
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
    delete g;
}

void AdamOptimizer::checkGrads(const vector<vector<Real>>& inputs)
{
    std::cout << std::setprecision(9);
    int nseries = inputs.size();
    vector<Real> res, Errors(nseries*nOutputs);
    net->allocateSeries(nseries+1);

    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    
    Grads * g = new Grads(nWeights,nBiases);
    Grads * G = new Grads(nWeights,nBiases);
    for (int w=0; w<nWeights; w++)
    {   *(g->_W+w) = 0; *(G->_W+w) = 0; }
    for (int w=0; w<nBiases; w++)
    {   *(g->_B+w) = 0; *(G->_B+w) = 0; }
    const Real eps = 1e-6;
    int lastn = 2;
    
    for (int k=0; k<lastn; k++)
    {
        net->predict(inputs[k], res, net->series[k], net->series[k+1]);
        
        for (int i=0; i<nOutputs; i++)
            *(net->series[k+1]->errvals +iOutputs+i) = 0.;
    }

    for (int i=0; i<1; i++)
        *(net->series[lastn]->errvals +iOutputs+i) = -1.;//Errors[1*nOutputs + i];
    
    net->computeDeltasEnd(net->series, lastn);
    for (int k=lastn-1; k>=1; k--)
        net->computeDeltasSeries(net->series, k);
    
    for (int k=1; k<=lastn; k++)
    {
        net->computeGradsLightSeries(net->series, k, g);
        stackGrads(G,g);
    }

    
    for (int w=0; w<nWeights; w++)
    {
        *(g->_W+w) = 0;
        *(net->weights+w) += eps;
        
        for (int k=0; k<lastn; k++)
            net->predict(inputs[k], res, net->series[k], net->series[k+1]);
        
        for (int i=0; i<1; i++)
            Errors[(lastn-1)*nOutputs + i] = - *(net->series[lastn]->outvals+iOutputs+i);
        
        *(net->weights+w) -= 2*eps;
        
        for (int k=0; k<lastn; k++)
            net->predict(inputs[k], res, net->series[k], net->series[k+1]);
        
        for (int i=0; i<1; i++)
        {
            Real err = - *(net->series[lastn]->outvals+iOutputs+i);
            *(g->_W+w) += (Errors[(lastn-1)*nOutputs + i]-err)/(2*eps);
        }
        
        *(net->weights+w) += eps;
        if (fabs((*(G->_W+w)-*(g->_W+w))/max(fabs(*(G->_W+w)),fabs(*(g->_W+w)))) > 1e-4)
        cout << "W"<<w<<" "<< *(G->_W+w) << " " << *(g->_W+w)<<" "<< (*(G->_W+w)-*(g->_W+w))/max(fabs(*(G->_W+w)),fabs(*(g->_W+w)))<<endl;
    }
    
    for (int w=0; w<nBiases; w++)
    {
        *(g->_B+w) = 0;
        *(net->biases+w) += eps;
        
        for (int k=0; k<lastn; k++)
            net->predict(inputs[k], res, net->series[k], net->series[k+1]);
        
        for (int i=0; i<1; i++)
            Errors[(lastn-1)*nOutputs + i] = - *(net->series[lastn]->outvals+iOutputs+i);
        
        *(net->biases+w) -= 2*eps;
        
        for (int k=0; k<lastn; k++)
            net->predict(inputs[k], res, net->series[k], net->series[k+1]);
        
        for (int i=0; i<1; i++)
        {
            Real err = - *(net->series[lastn]->outvals+iOutputs+i);
            *(g->_B+w) += (Errors[(lastn-1)*nOutputs + i]-err)/(2*eps);
        }
        
        *(net->biases+w) += eps;
        
        //cout <<"B"<<w<<" "<< *(G->_B+w) << " " << *(g->_B+w) << endl;
        if (fabs((*(G->_B+w)-*(g->_B+w))/max(fabs(*(G->_B+w)),fabs(*(g->_B+w)))) > 1e-4)
        cout << "B"<<w<<" "<< *(G->_B+w) << " " << *(g->_B+w)<<" "<< (*(G->_B+w)-*(g->_B+w))/max(fabs(*(G->_B+w)),fabs(*(g->_B+w))) <<endl;
    }
    printf("\n\n\n");
    abort();
}

void AdamOptimizer::addUpdate(Grads * G)
{
    update(net->weights, G->_W, _1stMomW, _2ndMomW, nWeights, eta);
    update(net->biases,  G->_B, _1stMomB, _2ndMomB, nBiases, eta);

    beta_t_1 *= beta_1;
    beta_t_2 *= beta_2;
}

#if 1

void AdamOptimizer::stackGrads(Grads * G, Grads * g)
{
    batchsize++;

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int j=0; j<nWeights; j+=SIMD)
        {
            #if SIMD==1
            *(G->_W + j) += *(g->_W + j);
            #else
            STORE (G->_W + j, ADD (LOAD(G->_W + j), LOAD(g->_W + j)));
            #endif
        }
        
        #if SIMD>1
        #pragma omp single nowait
        for (int j=int(nWeights/SIMD)*SIMD ; j<nWeights; ++j)
            *(G->_W + j) += *(g->_W + j);
        #endif
        
        #pragma omp for nowait
        for (int j=0; j<nBiases; j+=SIMD)
        {
            #if SIMD==1
            *(G->_B + j) += *(g->_B + j);
            #else
            STORE (G->_B + j, ADD (LOAD(G->_B + j), LOAD(g->_B + j)));
            #endif
        }
        
        #if SIMD>1
        #pragma omp single
        for (int j=int(nBiases/SIMD)*SIMD ; j<nBiases; ++j)
            *(G->_B + j) += *(g->_B + j);
        #endif
    }
    
}

void AdamOptimizer::update(Grads * G)
{
    if (lambda>1e-9)
    {
        updateDecay(net->weights, G->_W, _1stMomW, _2ndMomW, nWeights, eta);
        updateDecay(net->biases,  G->_B, _1stMomB, _2ndMomB, nBiases, eta);
    }
    else
    {
        update(net->weights, G->_W, _1stMomW, _2ndMomW, nWeights, eta);
        update(net->biases,  G->_B, _1stMomB, _2ndMomB, nBiases, eta);
    }
    
    beta_t_1 *= beta_1;
    beta_t_2 *= beta_2;
    nepoch++;
}

#else

void AdamOptimizer::stackGrads(Grads * G, Grads * g)
{
    batchsize++;
    
    stackGrads(G->_W, g->_W, _1stMomW, _2ndMomW, nWeights);
    stackGrads(G->_B, g->_B, _1stMomB, _2ndMomB, nBiases);
    
    beta_t_1 *= beta_1;
    beta_t_2 *= beta_2;
}

void AdamOptimizer::update(Grads * G)
{
    update(net->weights, G->_W, nWeights, eta);
    update(net->biases,  G->_B, nBiases, eta);
}

#endif

void AdamOptimizer::update(Real* dest, Real* grad, Real* _1stMom, Real* _2ndMom, const int N, Real _eta)
{
    #pragma omp parallel
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
        for (int i=0; i<N; i+=SIMD)
        {
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
        
        #if SIMD > 1
        #pragma omp single nowait
        for (int i=int(N/SIMD)*SIMD ; i<N; ++i)
        {
            *(_1stMom + i) = beta_1 * *(_1stMom + i) + (1.-beta_1) * *(grad + i);
            *(_2ndMom + i) = beta_2 * *(_2ndMom + i) + (1.-beta_2) * *(grad + i) * *(grad + i);
            
            *(grad + i) = 0.; //reset grads
            *(dest + i) += fac12 * *(_1stMom + i)  / sqrt(*(_2ndMom + i) + epsilon);
        }
        #endif
    }
}

void AdamOptimizer::updateDecay(Real* dest, Real* grad, Real* _1stMom, Real* _2ndMom, const int N, Real _eta)
{
    #pragma omp parallel
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
        for (int i=0; i<N; i+=SIMD)
        {
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
        
        #if SIMD > 1
        #pragma omp single nowait
        for (int i=int(N/SIMD)*SIMD ; i<N; ++i)
        {
            *(_1stMom + i) = beta_1 * *(_1stMom + i) + (1.-beta_1) * *(grad + i);
            *(_2ndMom + i) = beta_2 * *(_2ndMom + i) + (1.-beta_2) * *(grad + i) * *(grad + i);
            
            *(grad + i) = 0.; //reset grads
            *(dest + i) += fac12 * *(_1stMom + i)/sqrt(*(_2ndMom + i)+epsilon) -*(dest + i)*lambda*_eta;
        }
        #endif
    }
}

void AdamOptimizer::stackGrads(Real * G, Real * g, Real* _1stMom, Real* _2ndMom, const int N)
{
    #pragma omp parallel
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
        
        #if SIMD > 1
        #pragma omp single nowait
        for (int i=int(N/SIMD)*SIMD ; i<N ; ++i)
        {
            *(_1stMom + i) = beta_1 * *(_1stMom + i) + (1.-beta_1) * *(g + i);
            *(_2ndMom + i) = beta_2 * *(_2ndMom + i) + (1.-beta_2) * *(g + i) * *(g + i);
            *(G + i) += *(_1stMom + i) * fac12 / sqrt(*(_2ndMom + i) + epsilon);
        }
        #endif
    }
}

void AdamOptimizer::update(Real* dest, Real* grad, const int N, Real _eta)
{
    #pragma omp parallel
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
        
        #if SIMD > 1
        #pragma omp single nowait
        for (int i=int(N/SIMD)*SIMD; i<N; ++i)
        {
            *(dest + i) += _eta * *(grad + i);
            *(grad + i) = 0.;
        }
        #endif
    }
}

void AdamOptimizer::init(Real* dest, const int N, Real ini)
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


