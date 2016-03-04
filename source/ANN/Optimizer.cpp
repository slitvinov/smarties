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

AdamOptimizer::AdamOptimizer(Network * _net, Profiler * _prof, Settings  & settings) : eta(settings.nnEta), beta_1(0.8), beta_2(0.999), epsilon(1e-8), net(_net), profiler(_prof), nInputs(net->nInputs), nOutputs(net->nOutputs), iOutputs(net->iOutputs), nWeights(net->nWeights), nBiases(net->nBiases), beta_t_1(0.8), beta_t_2(0.999), batchsize(0), nepoch(1)
{
    _myallocate(_1stMomW, nWeights)
    init(_1stMomW, nWeights);
    _myallocate(_1stMomB, nBiases)
    init(_1stMomB, nBiases);
    _myallocate(_2ndMomW, nWeights)
    init(_2ndMomW, nWeights);
    _myallocate(_2ndMomB, nBiases)
    init(_2ndMomB, nBiases);
    _myallocate(_etaW, nWeights)
    init(_etaW, nWeights, eta);
    _myallocate(_etaB, nBiases)
    init(_etaB, nBiases,  eta);
}

void AdamOptimizer::trainBatch(const vector<const vector<Real>*>& inputs, const vector<const vector<Real>*>& targets, Real & trainMSE)
{
    trainMSE = 0.;
    int nseries = inputs.size();
    vector<Real> res(nOutputs);
    Grads * g = new Grads(nWeights,nBiases);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    
    #pragma omp parallel
    {
        net->clearDsdw();
        for (int k=0; k<nseries; k++)
        {
            #pragma omp master
                profiler->start("F");
            
            net->predict(*(inputs[k]), res, net->series[0], net->series[1]);
            
            #pragma omp master
            {
                profiler->stop("F");
                profiler->start("B");
            }
            #pragma omp single
            for (int j =0; j<nOutputs; j++)
            {
                res[j] = (*(targets[k]))[j] - res[j];
                trainMSE += 0.5*res[j]*res[j];
            }
            
            net->computeGrads(res, net->series[0], net->series[1], g);
            
            #pragma omp master
            {
                profiler->stop("B");
                profiler->start("S");
            }
            
            stackGrads(net->grad,g);
            
            #pragma omp master
                profiler->stop("S");
            
            #pragma omp single
            if (nseries>1)
                std::swap(net->series[1], net->series[0]);
        }
        #pragma omp master
            profiler->start("W");
        
        update(net->grad);
        
        #pragma omp master
            profiler->stop("W");
    }
    
    trainMSE /= (Real)inputs.size();
    delete g;
}

void AdamOptimizer::trainSeries(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE)
{
    trainMSE = 0.0;
    vector<Real> res;
    int nseries = inputs.size();
    for (int k=0; k<=nseries; k++)
        net->freshSeries(k+1);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    Grads * g = new Grads(nWeights,nBiases);
    
    //printf("\n\n\n NEW SERIES \n");
    #pragma omp parallel
    {
        //STEP 1: go through the data to compute predictions 
        #pragma omp master
        profiler->start("F");
        
        for (int k=0; k<nseries; k++)
        {
            net->predict(inputs[k], res, net->series[k], net->series[k+1]);
            
            #pragma omp master
            {
            for (int i=0; i<nOutputs; i++)
            { //put this loop here to slightly reduce overhead on second step
                Real err = targets[k][i]- *(net->series[k+1]->outvals+iOutputs+i);
                //printf("%f ",err);
                *(net->series[k+1]->errvals +iOutputs+i) = err;
                trainMSE += 0.5*err*err;
            }
                //printf("\n");
            }
        }
        
        #pragma omp master
        {
            profiler->stop("F");
            profiler->start("B");
        }
        
        //STEP 2: go backwards to backpropagate deltas (errors)
        net->clearErrors(net->series[nseries+1]); //there is a omp for in here
        for (int k=nseries; k>=1; k--)
            net->computeDeltasSeries(net->series, k);
        
        #pragma omp master
        profiler->stop("B");
        
        //STEP 3: go ahead again to compute the gradients with eligibility trace (dsdw(t) depends on dsdw(t-1))
        #pragma omp master
        profiler->start("G");
        
        //net->clearDsdw();
        for (int k=1; k<=nseries; k++)
        {
            //for (int i=0; i<nInputs; i++)
            //    printf("%f ",inputs[k-1][i]);
            //printf("\n");
            net->computeGradsLightSeries(net->series, k, g);
            stackGrads(net->grad,g);
        }
        
        #pragma omp master
        profiler->stop("G");
        
        //STEP 4: finally update the weights
        #pragma omp master
        profiler->start("W");
        
        update(net->grad);
        
        #pragma omp master
        profiler->stop("W");
    }
    //printf("%f\n",trainMSE);
    trainMSE /= (Real)inputs.size();
    delete g;
}

void AdamOptimizer::checkGrads(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE)
{
    std::cout << std::setprecision(9);
    int nseries = inputs.size();
    vector<Real> res, Errors(nseries*nOutputs);
    for (int k=0; k<=nseries; k++)
    {
        net->freshSeries(k+1);
        //net->clearErrors(net->series[k+1]); //there is a omp for in here
    }
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    
    Grads * g = new Grads(nWeights,nBiases);
    Grads * G = new Grads(nWeights,nBiases);
    for (int w=0; w<nWeights; w++)
    {
        *(g->_W+w) = 0;
        *(G->_W+w) = 0;
    }
    for (int w=0; w<nBiases; w++)
    {
        *(g->_B+w) = 0;
        *(G->_B+w) = 0;
    }
    const double eps = 1e-6;
    int lastn = 2;
    #pragma omp parallel
    {
        for (int k=0; k<lastn; k++)
        {
            //printf("\n Series %d\n",k+1);
            
            net->predict(inputs[k], res, net->series[k], net->series[k+1]);
            
            #pragma omp master
            for (int i=0; i<nOutputs; i++)
                *(net->series[k+1]->errvals +iOutputs+i) = 0.;
        }

        for (int i=0; i<1; i++)
        {
            Errors[(lastn-1)*nOutputs + i] = targets[lastn-1][i]- *(net->series[lastn]->outvals+iOutputs+i);
            *(net->series[lastn]->errvals +iOutputs+i) = -1.;//Errors[1*nOutputs + i];
        }
        net->computeDeltasEnd(net->series, lastn);
        for (int k=lastn-1; k>=1; k--)
        {
            net->computeDeltasSeries(net->series, k);
            printf("\n\n\n");
        }
        
        for (int k=1; k<=lastn; k++)
        {
            net->computeGradsLightSeries(net->series, k, g);
            stackGrads(G,g);
        }
    }
    
    for (int w=0; w<nWeights; w++)
    {
        *(g->_W+w) = 0;
        *(net->weights+w) += eps;
        
        for (int k=0; k<lastn; k++)
        {
            net->predict(inputs[k], res, net->series[k], net->series[k+1]);
        }
        
        for (int i=0; i<1; i++)
        {
            Errors[(lastn-1)*nOutputs + i] = targets[(lastn-1)][i]- *(net->series[lastn]->outvals+iOutputs+i);
        }
        
        *(net->weights+w) -= 2*eps;
        
        for (int k=0; k<lastn; k++)
        {
            net->predict(inputs[k], res, net->series[k], net->series[k+1]);
        }
        
        for (int i=0; i<1; i++)
        {
            Real err = targets[(lastn-1)][i]- *(net->series[lastn]->outvals+iOutputs+i);
            *(g->_W+w) += (Errors[(lastn-1)*nOutputs + i]-err)/(2*eps);
        }
        
        *(net->weights+w) += eps;
        //cout <<"W"<<w<<" "<< *(G->_W+w) << " " << *(g->_W+w) << endl;
        cout << "W"<<w<<" "<< *(G->_W+w) << " " << *(g->_W+w)<<" "<< (*(G->_W+w)-*(g->_W+w))/max(fabs(*(G->_W+w)),fabs(*(g->_W+w)))<<endl;
    }
    
    for (int w=0; w<nBiases; w++)
    {
        *(g->_B+w) = 0;
        *(net->biases+w) += eps;
        
        for (int k=0; k<lastn; k++)
        {
            net->predict(inputs[k], res, net->series[k], net->series[k+1]);
        }
        
        for (int i=0; i<1; i++)
        {
            Errors[(lastn-1)*nOutputs + i] = targets[(lastn-1)][i]- *(net->series[lastn]->outvals+iOutputs+i);
        }
        
        *(net->biases+w) -= 2*eps;
        
        for (int k=0; k<lastn; k++)
        {
            net->predict(inputs[k], res, net->series[k], net->series[k+1]);
        }
        
        for (int i=0; i<1; i++)
        {
            Real err = targets[(lastn-1)][i]- *(net->series[lastn]->outvals+iOutputs+i);
            *(g->_B+w) += (Errors[(lastn-1)*nOutputs + i]-err)/(2*eps);
        }
        
        *(net->biases+w) += eps;
        
        //cout <<"B"<<w<<" "<< *(G->_B+w) << " " << *(g->_B+w) << endl;
        cout << "B"<<w<<" "<< *(G->_B+w) << " " << *(g->_B+w)<<" "<< (*(G->_B+w)-*(g->_B+w))/max(fabs(*(G->_B+w)),fabs(*(g->_B+w))) <<endl;
    }
    printf("\n\n\n");
    printf("\n\n\n");
    abort();
}

void AdamOptimizer::trainSeries3(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE)
{
    trainMSE = 0.0;
    int nseries = inputs.size();
    vector<Real> res(nOutputs);
    //Grads * g = new Grads(nWeights,nBiases);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    
    //printf("\n\n\n NEW SERIES 3\n");
    #pragma omp parallel
    {
        net->clearDsdw();
        //STEP 1: go through the data to compute predictions
        for (int k=0; k<nseries; k++)
        {
            #pragma omp master
                profiler->start("F");
            
            net->predict(inputs[k], res, net->series[0], net->series[1]);
            
            #pragma omp master
            {
                profiler->stop("F");
                profiler->start("B");
            }
            #pragma omp single
            {
            for (int j =0; j<nOutputs; j++)
            {
                res[j] = targets[k][j] - res[j];
                //printf("%f ",res[j]);
                trainMSE += 0.5*res[j]*res[j];
            }
                //printf("\n");
            }
            
            net->computeGrads(res, net->series[0], net->series[1], net->grad);
            
            #pragma omp master
            {
                profiler->stop("B");
                profiler->start("W");
            }
            
            addUpdate(net->grad);
            
            #pragma omp master
                profiler->stop("W");
            
            #pragma omp single
            std::swap(net->series[1], net->series[0]);
        }
    }
    //printf("%f\n",trainMSE);
    trainMSE /= (Real)inputs.size();
    //delete g;
}

void AdamOptimizer::trainSeries2(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE)
{
    trainMSE = 0.0;
    vector<Real> res;
    int nseries = inputs.size();
    for (int k=0; k<=nseries; k++)
        net->freshSeries(k+1);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    
    #pragma omp parallel
    {
        net->clearErrors(net->series[nseries+1]); //there is a omp for in here
        //net->clearDsdw();
        for (int e=1; e<=nseries; e++)
        {
            //STEP 1: go through the data to compute predictions
            #pragma omp master
            profiler->start("F");
            
            for (int k=e; k<=nseries; k++)
            {
                net->predict(inputs[k-1], res, net->series[k-1], net->series[k]);
                
                #pragma omp master
                for (int i=0; i<nOutputs; i++)
                { //put this loop here to slightly reduce overhead on second step
                    Real err = targets[k-1][i]- *(net->series[k]->outvals+iOutputs+i);
                    *(net->series[k]->errvals +iOutputs+i) = err;
                    if(e==k) trainMSE += 0.5*err*err;
                }
            }
            
            #pragma omp master
            profiler->stop("F");
            
            //STEP 2: go backwards to backpropagate deltas (errors)
            #pragma omp master
            profiler->start("B");
            
            for (int k=nseries; k>=e; k--)
                net->computeDeltasSeries(net->series, k);
            
            #pragma omp master
            profiler->stop("B");
            
            //STEP 3: go ahead again to compute the gradients with eligibility trace (dsdw(t) depends on dsdw(t-1))
            #pragma omp master
            profiler->start("G");
            
            net->computeGradsLightSeries(net->series, e, net->grad);

            #pragma omp master
            profiler->stop("G");
            
            //STEP 4: finally update the weights
            #pragma omp master
            profiler->start("W");
            
            addUpdate(net->grad);
            
            #pragma omp master
            profiler->stop("W");
        }
    }
    
    trainMSE /= (Real)inputs.size();
}

void AdamOptimizer::trainSeries4(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE)
{
    trainMSE = 0.0;
    vector<Real> res;
    int nseries = inputs.size();
    for (int k=0; k<=nseries; k++)
        net->freshSeries(k+1);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    
    #pragma omp parallel
    {
        for (int e=1; e<=nseries; e++)
        {
            //STEP 1: go through the data to compute predictions
            #pragma omp master
            profiler->start("F");
            
            for (int k=max(1,e-1); k<=min(nseries,e+1); k++)
            {
                net->predict(inputs[k-1], res, net->series[k-1], net->series[k]);
                
                #pragma omp master
                for (int i=0; i<nOutputs; i++)
                { //put this loop here to slightly reduce overhead on second step
                    Real err = targets[k-1][i]- *(net->series[k]->outvals+iOutputs+i);
                    *(net->series[k]->errvals +iOutputs+i) = err;
                    if(e==k) trainMSE += 0.5*err*err;
                }
            }
            
            #pragma omp master
            profiler->stop("F");
            
            //STEP 2: go backwards to backpropagate deltas (errors)
            #pragma omp master
            profiler->start("B");
            
            if (e<nseries)
            {
                net->computeDeltasEnd(net->series, e+1);
                net->computeDeltasSeries(net->series, e);
            }
            else
                net->computeDeltasEnd(net->series, e);
            
            #pragma omp master
            profiler->stop("B");
            
            //STEP 3: go ahead again to compute the gradients with eligibility trace (dsdw(t) depends on dsdw(t-1))
            #pragma omp master
            profiler->start("G");
            
            net->computeGradsLightSeries(net->series, e, net->grad);

            #pragma omp master
            profiler->stop("G");
            
            //STEP 4: finally update the weights
            #pragma omp master
            profiler->start("W");
            
            addUpdate(net->grad);
            
            #pragma omp master
            profiler->stop("W");
        }
    }
    
    trainMSE /= (Real)inputs.size();
}

void AdamOptimizer::trainSeries5(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE)
{
    trainMSE = 0.0;
    vector<Real> res;
    int nseries = inputs.size();
    for (int k=0; k<=nseries; k++)
        net->freshSeries(k+1);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    int first =1;
    #pragma omp parallel
    {
        for (int e=nseries; e>=1; e--)
        {
            //STEP 1: go through the data to compute predictions
            #pragma omp master
            profiler->start("F");
            
            for (int k=first; k<=min(nseries,e+1); k++)
            {
                net->predict(inputs[k-1], res, net->series[k-1], net->series[k]);
                
                #pragma omp master
                for (int i=0; i<nOutputs; i++)
                { //put this loop here to slightly reduce overhead on second step
                    Real err = targets[k-1][i]- *(net->series[k]->outvals+iOutputs+i);
                    *(net->series[k]->errvals +iOutputs+i) = err;
                    if(e==k) trainMSE += 0.5*err*err;
                }
            }
            
            #pragma omp master
            profiler->stop("F");
            
            //STEP 2: go backwards to backpropagate deltas (errors)
            #pragma omp master
            profiler->start("B");
            
            if (e==nseries)
                net->computeDeltasEnd(net->series, e);
            else if (e+1==nseries)
            {
                net->computeDeltasEnd(net->series, e+1);
                net->computeDeltasSeries(net->series, e);
            }
            else
            {
                net->computeDeltasSeries(net->series, e+1);
                net->computeDeltasSeries(net->series, e);
            }
            
            #pragma omp master
            profiler->stop("B");
            
            //STEP 3: go ahead again to compute the gradients with eligibility trace (dsdw(t) depends on dsdw(t-1))
            #pragma omp master
            profiler->start("G");
            
            net->computeGradsLightSeries(net->series, e, net->grad);

            #pragma omp master
            profiler->stop("G");
            
            //STEP 4: finally update the weights
            #pragma omp master
            profiler->start("W");
            
            addUpdate(net->grad);
            
            #pragma omp master
            profiler->stop("W");
            
            #pragma omp single
            first = max(1,e-1);
        }
    }
    
    trainMSE /= (Real)inputs.size();
}

void AdamOptimizer::addUpdate(Grads * G)
{
    update(net->weights, G->_W, _1stMomW, _2ndMomW, _etaW, nWeights);
    update(net->biases,  G->_B, _1stMomB, _2ndMomB, _etaB, nBiases);
    
    #pragma omp single
    {
        beta_t_1 *= beta_1;
        beta_t_2 *= beta_2;
    }
}

#if 1

void AdamOptimizer::stackGrads(Grads * G, Grads * g)
{
    #pragma omp master
    batchsize++;
    #if SIMD > 1
    vec _m01 = SET1 (-1.);
    vec _p01 = SET1 (1.);
    #endif

    #pragma omp for nowait
    for (int j=0; j<nWeights; j+=SIMD)
        #if SIMD == 1
        *(G->_W + j) += *(g->_W + j);//max(min(*(g->_W + j),1.),-1.);
        #else
        STORE (G->_W + j, ADD (LOAD(G->_W + j), MAX(MIN(LOAD(g->_W + j),_p01),_m01)));
        #endif

    #pragma omp for
    for (int j=0; j<nBiases; j+=SIMD)
        #if SIMD == 1
        *(G->_B + j) += *(g->_B + j);//max(min(*(g->_B + j),1.),-1.);//*(g->_B + j);
        #else
        STORE (G->_B + j, ADD (LOAD(G->_B + j), MAX(MIN(LOAD(g->_B + j),_p01),_m01)));
        #endif
}

void AdamOptimizer::update(Grads * G)
{
    Real etaB = eta;// *0.5*(1.+1./pow(nepoch,0.5));///max(batchsize,1);
    
    update(net->weights, G->_W, _1stMomW, _2ndMomW, nWeights, etaB);
    update(net->biases,  G->_B, _1stMomB, _2ndMomB, nBiases, etaB);
    
    #pragma omp single
    {
        beta_t_1 *= beta_1;
        beta_t_2 *= beta_2;
        nepoch++;
        
        //for (int i =0 ; i<nWeights; i++) printf("%d %f %f %f %f %f\n",i,*(net->weights + i),*(G->_W + i),*(_1stMomW + i), *(_2ndMomW + i),etaB);
    }
}

#else

void AdamOptimizer::stackGrads(Grads * G, Grads * g)
{
    #pragma omp master
    batchsize++;
    
    stackGrads(G->_W, g->_W, _1stMomW, _2ndMomW, nWeights);
    stackGrads(G->_B, g->_B, _1stMomB, _2ndMomB, nBiases);
    
    #pragma omp master
    {
        beta_t_1 *= beta_1;
        beta_t_2 *= beta_2;
    }
}

void AdamOptimizer::update(Grads * G)
{
    Real etaB = eta;///max(batchsize,1);
    
    update(net->weights, G->_W, nWeights, etaB);
    update(net->biases,  G->_B, nBiases, etaB);
}

#endif

void AdamOptimizer::update(Real* dest, Real* grad, Real* _1stMom, Real* _2ndMom, Real* _eta, const int N)
{
    #if SIMD > 1
    const vec _05 = SET1( 0.5 );
    const vec _1 = SET1( 1.0 );
    //const vec M = SET1( 0.1 );
    const vec B = SET1( 0.99 );
    const vec _B = SET1( 0.01 );
    const vec EPS = SET1( 1e-4 );
    const vec zeros = SET0 ();
    #endif
    
    #pragma omp for nowait
    for (int i=0; i<N; i+=SIMD)
    {
        #if SIMD == 1
        
        *(_2ndMom + i) = 0.9 * *(_2ndMom + i) + 0.1 * *(grad + i) * *(grad + i);
        *(_eta    + i) = *(_eta+i) * max(.5, 1+ .5 * *(grad + i) * *(_1stMom + i) / *(_2ndMom + i));
        *(_1stMom + i) = 0.9 * *(_1stMom + i) + 0.1 * *(grad + i);
        
        *(dest + i) += *(_eta+i) * *(grad + i);
        *(grad + i) = 0;
        #else
        
        vec _DW = LOAD(grad + i);
        vec M1 = LOAD(_1stMom + i);
        vec M2 = ADD( MUL (B, LOAD(_2ndMom+i)), MUL (_B, MUL(_DW,_DW)));
        
        vec ETA = MUL(LOAD(_eta+i), MAX(_05, ADD(_1, MUL(MUL(_B, MUL(M1, _DW)), RCP(ADD(M2,EPS))))));
        
        STORE(dest+i,ADD(LOAD(dest+i), MUL(ETA,_DW)));
        STORE(grad + i,zeros);
        
        STORE(_1stMom+i, ADD( MUL (B, M1), MUL (_B, _DW)));
        STORE(_2ndMom+i, M2);
        STORE(_eta+i, ETA);
        #endif
        /*
        #if SIMD == 1
        Real tmp = *(_1stMom + i) * *(grad + i) + 0.5* *(_1stMom + i);
        *(_1stMom + i) = _eta * *(grad + i) + 0.5* *(_1stMom + i);
        *(dest + i) += *(_1stMom + i);
        *(grad + i) = 0;
        #else
        vec _DW = ADD(MUL(ETAB, LOAD(grad + i)), MUL(ALFA,LOAD(_1stMom + i)) );
        STORE(dest+i, ADD( LOAD(dest+i), _DW));
        STORE(grad + i,zeros);
        STORE(_1stMom + i,_DW);
        #endif
         */
    }
}

void AdamOptimizer::update(Real* dest, Real* grad, Real* _1stMom, Real* _2ndMom, const int N, Real _eta)
{
    #if SIMD == 1
    Real fac1 = 1./(1.-beta_t_1);
    Real fac2 = 1./(1.-beta_t_2);
    //Real fac12 = (1.-beta_t_1)*(1.-beta_t_1)/(1.-beta_t_2);
    #else
    const vec B1 = SET1(beta_1);
    const vec B2 = SET1(beta_2);
    const vec _B1 =SET1(1.-beta_1);
    const vec _B2 =SET1(1.-beta_2);
    const vec F1 = SET1(_eta/(1.-beta_t_1));
    const vec F2 = SET1(1./(1.-beta_t_2));
//    const vec F2 = SET1( (1.-beta_t_1)*(1.-beta_t_1)/(1.-beta_t_2) );
//    const vec ETAB = SET1( _eta );
    const vec EPS = SET1(epsilon);
    const vec zeros = SET0 ();
    #endif
    
    #pragma omp for nowait
    for (int i=0; i<N; i+=SIMD)
    {
        #if SIMD == 1
        *(_1stMom + i) = beta_1 * *(_1stMom + i) + (1.-beta_1) * *(grad + i);
        *(_2ndMom + i) = beta_2 * *(_2ndMom + i) + (1.-beta_2) * *(grad + i) * *(grad + i);
        //*(dest + i) += _eta * *(_1stMom + i) * fac1 / (sqrt(*(_2ndMom + i) * fac2) + epsilon);
        *(dest + i) += _eta * *(_1stMom + i) / (sqrt(*(_2ndMom + i)) + epsilon);
        *(grad + i) = 0.; //reset grads
        
        //*(dest + i) += _eta * *(_1stMom + i)  / (sqrt(*(_2ndMom + i) * fac12 + epsilon));
        #else
        vec _DW = LOAD(grad + i);
        //FETCH((char*)    grad +i + M_PF_G, M_POL_G);
        vec M1 = ADD( MUL ( B1, LOAD(_1stMom + i)), MUL ( _B1, _DW));
        //FETCH((char*) _1stMom +i + M_PF_G, M_POL_G);
        vec M2 = ADD( MUL ( B2, LOAD(_2ndMom + i)), MUL ( _B2, MUL (_DW,_DW)));
        //FETCH((char*) _2ndMom +i + M_PF_G, M_POL_G);
        STORE(dest+i,ADD(LOAD(dest+i),MUL( MUL(F1,M1), RSQRT(ADD(MUL(M2,F2),EPS)))));
        //STORE(dest+i, ADD( LOAD(dest+i),MUL( MUL(ETAB, M1), RSQRT(ADD(MUL(M2,F2),EPS)))));
        //FETCH((char*)    dest +i + M_PF_G, M_POL_G);
        
        STORE(_1stMom + i,M1);
        STORE(_2ndMom + i,M2);
        STORE (grad+i,zeros); //reset grads
        #endif
    }
}

void AdamOptimizer::stackGrads(Real * G, Real * g, Real* _1stMom, Real* _2ndMom, const int N)
{
#if SIMD == 1
    Real fac1 = 1./(1.-beta_t_1);
    Real fac2 = 1./(1.-beta_t_2);
    //Real fac12 = (1.-beta_t_1)*(1.-beta_t_1)/(1.-beta_t_2);
#else
    const vec B1 = SET1 (beta_1);
    const vec B2 = SET1 (beta_2);
    const vec _B1 =SET1(1.-beta_1);
    const vec _B2 =SET1(1.-beta_2);
    const vec F1 = SET1(1./(1.-beta_t_1));
    const vec F2 = SET1(1./(1.-beta_t_2));
    //const vec F2 = SET1( (1.-beta_t_1)*(1.-beta_t_1)/(1.-beta_t_2) );
    const vec EPS = SET1(epsilon);
#endif
    
#pragma omp for nowait
    for (int i=0; i<N; i+=SIMD)
    {
#if SIMD == 1
        *(_1stMom + i) = beta_1 * *(_1stMom + i) + (1.-beta_1) * *(g + i);
        *(_2ndMom + i) = beta_2 * *(_2ndMom + i) + (1.-beta_2) * *(g + i) * *(g + i);
        *(G + i) += *(_1stMom + i) * fac1 / (sqrt(*(_2ndMom + i) * fac2) + epsilon);
        //*(G + i) += *(_1stMom + i)  / (sqrt(*(_2ndMom + i) * fac12 + epsilon));
#else
        vec _g = LOAD(g + i);
        vec M1 = ADD( MUL ( B1, LOAD(_1stMom + i)), MUL ( _B1, _g));
        vec M2 = ADD( MUL ( B2, LOAD(_2ndMom + i)), MUL ( _B2, MUL (_g,_g)));
        STORE(G+i, ADD( LOAD(G+i), MUL(MUL(M1, F1), RSQRT(ADD(MUL(M2,F2),EPS)))));
        
        //STORE(G+i, ADD( LOAD(G+i),MUL(M1, RSQRT(ADD(MUL(M2,F2),EPS)))));
        STORE(_1stMom + i,M1);
        STORE(_2ndMom + i,M2);
        //STORE (grad+i,zeros); //reset grads
#endif
    }
}

void AdamOptimizer::update(Real* dest, Real* grad, const int N, Real _eta)
{
#if SIMD > 1
    const vec ETAB = SET1( _eta );
    const vec zeros = SET0 ();
#endif
    
#pragma omp for
    for (int i=0; i<N; i+=SIMD)
    {
#if SIMD == 1
        *(dest + i) += _eta * *(grad + i);
        *(grad + i) = 0.; //reset grads
        //*(dest + i) += _eta * *(_1stMom + i)  / (sqrt(*(_2ndMom + i) * fac12 + epsilon));
#else
        STORE(dest+i, ADD( LOAD(dest+i), MUL(ETAB, LOAD(grad + i))));
        STORE(grad+i,zeros); //reset grads
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
        *(dest +j) = 0.;
        #else
        STORE (dest +j,zeros);
        #endif
    }
}

/*
 void AdamOptimizer::trainSeries2(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE)
 {
 trainMSE = 0.0;
 vector<Real> res;
 int nseries = inputs.size();
 for (int k=0; k<=nseries; k++)
 net->freshSeries(k+1);
 net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
 
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
 Real err = targets[k][i]- *(net->series[k+1]->outvals+iOutputs+i);
 *(net->series[k+1]->errvals +iOutputs+i) = err;
 trainMSE += 0.5*err*err;
 }
 }
 
 #pragma omp master
 profiler->stop("F");
 
 //STEP 2: go backwards to backpropagate deltas (errors)
 #pragma omp master
 profiler->start("B");
 
 net->clearErrors(net->series[nseries+1]); //there is a omp for in here
 for (int k=nseries; k>=1; k--)
 net->computeDeltasSeries(net->series, k); //requires k-1, k, k+1
 
 #pragma omp master
 profiler->stop("B");
 
 //STEP 3: go ahead again to compute the gradients with eligibility trace (dsdw(t) depends on dsdw(t-1))
 #pragma omp master
 profiler->start("G");
 
 net->clearDsdw();
 net->computeGradsSeries(net->series, 1, net->grad);
 
 #pragma omp master
 profiler->stop("G");
 
 #pragma omp master
 profiler->start("W");
 
 update(net->grad);
 
 #pragma omp master
 profiler->stop("W");
 
 for (int k=1; k<nseries; k++)
 {
 #pragma omp master
 profiler->start("F");
 
 net->predict(inputs[k], res, net->series[k], net->series[k+1]);
 
 #pragma omp single
 for (int i=0; i<nOutputs; i++)
 { //put this loop here to slightly reduce overhead on second step
 Real err = targets[k][i] - *(net->series[k+1]->outvals+iOutputs+i);
 *(net->series[k+1]->errvals +iOutputs+i) = err;
 batchsize = 10;
 }
 
 #pragma omp master
 profiler->stop("F");
 
 #pragma omp master
 profiler->start("B");
 
 net->computeDeltasSeries(net->series, k+1);
 
 #pragma omp master
 profiler->stop("B");
 
 #pragma omp master
 profiler->start("G");
 
 net->computeGradsSeries(net->series, k+1, net->grad);
 
 #pragma omp master
 profiler->stop("G");
 
 #pragma omp master
 profiler->start("W");
 
 update(net->grad);
 
 #pragma omp master
 profiler->stop("W");
 }
 }
 }
 
LMOptimizer::LMOptimizer(Network * _net, Profiler * _prof, Settings  & settings) : muMax(1e10), muMin(1e-6), muFactor(10), net(_net), profiler(_prof), nInputs(net->nInputs), nOutputs(net->nOutputs), iOutputs(net->iOutputs), nWeights(net->nWeights), nBiases(net->nBiases), totWeights(net->nWeights+net->nBiases), mu(0.1)
{
    dw.set_size(totWeights);
    Je.set_size(totWeights);
    diagJtJ.eye(totWeights, totWeights);
}

void LMOptimizer::stackGrads(Grads * g, const int k, const int i)
{
    #pragma omp for nowait
    for (int j=0; j<nWeights; j++)
        J(i + k*nOutputs, j) = -*(g->_W + j);
    
    #pragma omp for
    for (int j=0; j<nBiases; j++)
        J(i + k*nOutputs, j+nWeights) = -*(g->_B + j);
}

void LMOptimizer::tryNew()
{
    #pragma omp for nowait
    for (int j=0; j<nWeights; j++)
        *(net->weights +j) += dw(j);
    
    #pragma omp for
    for (int j=0; j<nBiases; j++)
        *(net->biases +j) += dw(j+nWeights);
}

void LMOptimizer::goBack()
{
    #pragma omp for nowait
    for (int j=0; j<nWeights; j++)
        *(net->weights +j) -= dw(j);
    
    #pragma omp for
    for (int j=0; j<nBiases; j++)
        *(net->biases +j) -= dw(j+nWeights);
}

void LMOptimizer::trainSeries(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE)
{
    trainMSE = 0.0;
    vector<Real> res;
    int nseries = inputs.size();
    for (int k=0; k<=nseries; k++)
        net->freshSeries(k+1);
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