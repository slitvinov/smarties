/*
 *  LSTMNet.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "Network.h"
#include "../ErrorHandling.h"
#include <cassert>
#define SCAL 3.
using namespace ErrorHandling;

void Network::orthogonalize(int nO, int nI, int n0)
{
    if (nI>=nO)
    for (int i=1; i<nO; i++)
    for (int j=0; j<i;  j++)
    {
        Real u_d_u = 0.0;
        Real v_d_u = 0.0;
        for (int k=0; k<nI; k++)
        {
            u_d_u += *(weights +n0 +j*nI +k)* *(weights +n0 +j*nI +k);
            v_d_u += *(weights +n0 +j*nI +k)* *(weights +n0 +i*nI +k);
        }
        if(u_d_u>0) //die("WTF did you do %d %d %d %d %d???\n",i,j,nO,nI,n0);
        for (int k=0; k<nI; k++)
            *(weights +n0 +i*nI +k) -= (v_d_u/u_d_u) * *(weights +n0 +j*nI +k);
    }
}

void Network::initializeWeights(Graph & g, mt19937 & gen)
{
    uniform_real_distribution<Real> dis(-1.,1.);
    
    for (int w=g.wPeep; w<(g.wPeep + 3*g.recurrSize); w++)
        *(weights +w) = dis(gen)*sqrt(SCAL)/Real(g.recurrSize);
    
    for (const auto & l : *g.nl_c_l)
    {
        for (int w=l->iW ; w<(l->iW + l->nO*l->nI); w++)
            *(weights +w) = dis(gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iW);
    }
    
    for (const auto & l : *g.nl_o_l)
    {
        for (int w=l->iW ; w<(l->iW + l->nO*l->nI); w++)
            *(weights +w) = dis(gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iW);
    }
    
    for (const auto & l : *g.rl_c_l)
    {
        for (int w=l->iW ; w<(l->iW + l->nO*l->nI); w++)
            *(weights +w) = dis(gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iW);
        
        for (int w=l->iWI; w<(l->iWI+ l->nO*l->nI); w++)
            *(weights +w) = dis(gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWI);
        
        for (int w=l->iWF; w<(l->iWF+ l->nO*l->nI); w++)
            *(weights +w) = dis(gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWF);
        
        for (int w=l->iWO; w<(l->iWO+ l->nO*l->nI); w++)
            *(weights +w) = dis(gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWO);
    }
    
    for (const auto & l : *g.rl_o_l)
    {
        for (int w=l->iW ; w<(l->iW + l->nO*l->nI); w++)
            *(weights +w) = dis(gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iW);
        
        for (int w=l->iWI; w<(l->iWI+ l->nO*l->nI); w++)
            *(weights +w) = dis(gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWI);
        
        for (int w=l->iWF; w<(l->iWF+ l->nO*l->nI); w++)
            *(weights +w) = dis(gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWF);
        
        for (int w=l->iWO; w<(l->iWO+ l->nO*l->nI); w++)
            *(weights +w) = dis(gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWO);
    }

    for (int w=g.biasHL; w<g.biasHL+g.normalSize; w++)
        *(biases +w) = dis(gen)*sqrt(SCAL)/Real(g.normalSize);
    
    for (int w=g.biasIN; w<g.biasIN+g.recurrSize; w++)
        *(biases +w) = dis(gen)*sqrt(SCAL)/Real(g.recurrSize);
        
    for (int w=g.biasIG; w<g.biasIG+g.recurrSize; w++)
        *(biases +w) =
                        //-2.; /* IG starts decisively closed */
                        dis(gen)*sqrt(SCAL)/Real(g.recurrSize);
        
    for (int w=g.biasFG; w<g.biasFG+g.recurrSize; w++)
        *(biases +w) =
                        // 2.; /* FG starts decisively open */
                         dis(gen)*sqrt(SCAL)/Real(g.recurrSize);
    
    for (int w=g.biasOG; w<g.biasOG+g.recurrSize; w++)
        *(biases +w) =
                        //-2.; /* OG starts decisively closed */
                         dis(gen)*sqrt(SCAL)/Real(g.recurrSize);
}

void Network::addNormal(Graph * p, Graph * g, bool first, bool last)
{
    int normalSize_SIMD = ceil((Real)g->normalSize/SIMD)*SIMD;
    if(!last && !first) g->normalSize = normalSize_SIMD;
    
    if (g->normalSize>0)
    {
        g->normalPos = nNeurons;
        nNeurons += normalSize_SIMD;
        g->biasHL = nBiases;
        nBiases += normalSize_SIMD;
        
        if (p->recurrSize>0)
        { //conntected to previous recurrent layer
            Link * link = new Link(p->recurrSize,p->recurrPos,g->normalSize,g->normalPos,nWeights,p->first);
            
            g->nl_c_l->push_back(link);
            p->rl_l_c->push_back(link);
            
            nWeights += p->recurrSize*g->normalSize;
        }
        
        if (p->normalSize>0)
        { //conntected to previous normal layer
            Link * link = new Link(p->normalSize,p->normalPos,g->normalSize,g->normalPos,nWeights,p->first);
            
            g->nl_c_l->push_back(link);
            p->nl_l_c->push_back(link);
            
            nWeights += p->normalSize*g->normalSize;
        }
        
        if (g->recurrSize>0)
        { //conntected to previous normal layer
            Link * link = new Link(g->recurrSize,g->recurrPos,g->normalSize,g->normalPos,nWeights,g->first);
            
            g->nl_c_l->push_back(link);
            g->rl_l_c->push_back(link);
            
            nWeights += g->recurrSize*g->normalSize;
        }
        
        //if (false)
        { //conntected to previous normal layer
            Link * link = new Link(g->normalSize,g->normalPos,g->normalSize,g->normalPos,nWeights,g->first);
            
            g->nl_o_l->push_back(link);
            g->nl_l_f->push_back(link);
            
            nWeights += g->normalSize*g->normalSize;
        }
        
        NormalLayer * l;
        ActivationFunction * f;
        
        //if (last) f = new Linear;
        //else
            f = new SoftSign;
        
        l = new NormalLayer(g->normalSize, g->normalPos, g->biasHL, g->nl_c_l, g->nl_l_c, g->nl_l_f, f, last);
        l->prev_input_links = g->nl_o_l;
        layers.push_back(l);
    }
}

void Network::addLSTM(Graph * p, Graph * g, bool first, bool last)
{
    if (last) die("How the fudge did you get here?\n");
    int normalSize_SIMD = ceil((Real)g->normalSize/SIMD)*SIMD; //g->normalSize
    int recurrSize_SIMD = ceil((Real)g->recurrSize/SIMD)*SIMD; //g->recurrSize
    if(!last && !first)          g->recurrSize = recurrSize_SIMD;
    if (last && g->normalSize>0) g->recurrSize = recurrSize_SIMD;
    g->normalSize = normalSize_SIMD; g->recurrSize = recurrSize_SIMD; //todo
    if (g->recurrSize>0)
    {
        {
            g->recurrPos = nNeurons;
            nNeurons += g->recurrSize;
            g->indState = nStates;
            nStates  += g->recurrSize;
            
            g->biasIN = nBiases;
            g->biasIG = g->biasIN + g->recurrSize;
            g->biasFG = g->biasIG + g->recurrSize;
            g->biasOG = g->biasFG + g->recurrSize;
            nBiases += 4*g->recurrSize;
            
            g->wPeep  = nWeights;
            nWeights+= 3*g->recurrSize;
            g->dSdB   = ndSdB;
            ndSdB += 5 * g->recurrSize;
        }
        
        if (p->recurrSize>0)
        { //conntected to previous recurrent layer
            int WeightHL = nWeights;
            nWeights += p->recurrSize*g->recurrSize;
            int WeightIG = nWeights;
            nWeights += p->recurrSize*g->recurrSize;
            int WeightFG = nWeights;
            nWeights += p->recurrSize*g->recurrSize;
            int WeightOG = nWeights;
            nWeights += p->recurrSize*g->recurrSize;
            
            int idSdW  = ndSdW;
            ndSdW += p->recurrSize * g->recurrSize;
            
            Link * link = new Link(p->recurrSize,p->recurrPos,g->recurrSize,g->recurrPos,g->indState,WeightHL,WeightIG,WeightFG,WeightOG,idSdW,p->first);
            g->rl_c_l->push_back(link);
            p->rl_l_c->push_back(link);
        }
        if (p->normalSize>0)
        { //conntected to previous normal layer
            int WeightHL = nWeights;
            nWeights += p->normalSize*g->recurrSize;
            int WeightIG = nWeights;
            nWeights += p->normalSize*g->recurrSize;
            int WeightFG = nWeights;
            nWeights += p->normalSize*g->recurrSize;
            int WeightOG = nWeights;
            nWeights += p->normalSize*g->recurrSize;
            
            int idSdW  = ndSdW;
            ndSdW += p->normalSize * g->recurrSize;
            
            Link * link = new Link(p->normalSize,p->normalPos,g->recurrSize,g->recurrPos,g->indState,WeightHL,WeightIG,WeightFG,WeightOG,idSdW,p->first);
            g->rl_c_l->push_back(link);
            p->nl_l_c->push_back(link);
        }

        { //conntected to past realization of recurrent layer
            int WeightHL = nWeights;
            nWeights += g->recurrSize*g->recurrSize;
            int WeightIG = nWeights;
            nWeights += g->recurrSize*g->recurrSize;
            int WeightFG = nWeights;
            nWeights += g->recurrSize*g->recurrSize;
            int WeightOG = nWeights;
            nWeights += g->recurrSize*g->recurrSize;
            
            int idSdW  = ndSdW;
            ndSdW += g->recurrSize * g->recurrSize;
            
            Link * link = new Link(g->recurrSize,g->recurrPos,g->recurrSize,g->recurrPos,g->indState,WeightHL,WeightIG,WeightFG,WeightOG,idSdW);
            g->rl_o_l->push_back(link);
            g->rl_l_f->push_back(link);
        }
        if (g->normalSize>0)
        { //conntected to past realization of normal layer
            int WeightHL = nWeights;
            nWeights += g->normalSize*g->recurrSize;
            int WeightIG = nWeights;
            nWeights += g->normalSize*g->recurrSize;
            int WeightFG = nWeights;
            nWeights += g->normalSize*g->recurrSize;
            int WeightOG = nWeights;
            nWeights += g->normalSize*g->recurrSize;
            
            int idSdW  = ndSdW;
            ndSdW += g->normalSize * g->recurrSize;
            
            Link * link = new Link(g->normalSize,g->normalPos,g->recurrSize,g->recurrPos,g->indState,WeightHL,WeightIG,WeightFG,WeightOG,idSdW);
            g->rl_o_l->push_back(link);
            g->nl_l_f->push_back(link);
        }
        
        NormalLayer * l;
        ActivationFunction * f = new SoftSign;
        ActivationFunction * h = new SoftSigm;
        l = new LSTMLayer(g->recurrSize, g->recurrPos, g->indState, g->wPeep, g->biasIN, g->biasIG, g->biasFG, g->biasOG, g->dSdB, g->rl_c_l, g->rl_o_l, g->rl_l_c, g->rl_l_f, f, h, last);
        layers.push_back(l);
    }
}

Network::Network(vector<int>& normalSize, vector<int>& recurrSize, Settings & settings, int nAgents) : nInputs(0), nOutputs(0), nLayers(0), nNeurons(0), nWeights(0), nBiases(0), ndSdW(0), ndSdB(0), nStates(0), iOutputs(0), allocatedFrozenWeights(false)
{
    if(normalSize.size()<3)
        die("Put at least one hidden layer, would you kindly? \n");
    if(recurrSize.back()>0)
        die("Put just a normal layer as output, would you kindly? \n");
    if(recurrSize.front()>0)
        die("Put just a normal layer as input, would you kindly? \n");
    mt19937 gen(settings.randSeed);
    int nMixedLayers = normalSize.size();
    nInputs = normalSize.front();
    nOutputs = normalSize.back();
    
    {
        Graph * g = new Graph();
        g->first = true;
        g->normalSize = nInputs;
        nNeurons += ceil((Real)nInputs/SIMD)*SIMD;
        G.push_back(g);
    }
    
    for (int i=1; i<nMixedLayers; i++)
    { //layer 0 is the input layer
        Graph * g = new Graph();
        bool first = i==1; bool last = i+1==nMixedLayers;
        g->recurrSize = recurrSize[i];
        g->normalSize = normalSize[i];
        if (recurrSize[i]>0)
            addLSTM(G.back(),g,first,last);
        if (normalSize[i]>0)
            addNormal(G.back(),g,first,last);
        G.push_back(g);
    }
    
    iOutputs = G.back()->normalPos;
    nLayers = layers.size();
    printf("nNeurons= %d, nWeights= %d, nBiases= %d, ndSdW= %d, ndSdB= %d, nStates= %d \n", nNeurons, nWeights, nBiases, ndSdW, ndSdB, nStates);
    
    for (int i=0; i<nAgents; ++i)
    {
        Mem * m = new Mem(nNeurons, nStates);
        mem.push_back(m);
    }
    
    dsdw = new Dsdw(ndSdW,ndSdB);
    grad = new Grads(nWeights,nBiases);
    
    freshSeries(0);
    freshSeries(1);
    
    _myallocate(weights, nWeights)
    _myallocate(biases,   nBiases)
    
    for (int i=1; i<static_cast<int>(G.size()); i++)
        initializeWeights(*G[i], gen);
}

void Network::save(string fname)
{
    debug1("Saving into %s\n", fname.c_str());
    
    string nameBackup = fname + "_tmp";
    ofstream out(nameBackup.c_str());
    
    if (!out.good()) die("Unable to open save into file %s\n", fname.c_str());
    
    out.precision(20);
    
    out << nWeights << " "  << nBiases << " " << nLayers  << " " << nNeurons << endl;
    
    for (int i=0; i<nWeights; i++)
        out << *(weights + i) << " ";
    
    for (int i=0; i<nBiases; i++)
        out << *(biases + i) << " ";

    out.flush();
    out.close();
    
    // Prepare copying command
    string command = "cp ";
    string nameOriginal = fname;
    command = command + nameBackup + " " + nameOriginal;
    
    // Submit the command to the system
    system(command.c_str());
}

bool Network::restart(string fname)
{
    string nameBackup = fname + "_tmp";
    
    ifstream in(nameBackup.c_str());
    debug1("Reading from %s\n", nameBackup.c_str());
    if (!in.good())
    {
        error("WTF couldnt open file %s (ok keep going mofo)!\n", fname.c_str());
        return false;
    }
    
    int readTotWeights, readTotBiases, readNNeurons, readNLayers;
    in >> readTotWeights  >> readTotBiases >> readNLayers >> readNNeurons;
    
    if (readTotWeights != nWeights || readTotBiases != nBiases || readNLayers != nLayers || readNNeurons != nNeurons)
    die("Network parameters differ!");
    
    for (int i=0; i<nWeights; i++)
        in >> *(weights + i);
    
    for (int i=0; i<nBiases; i++)
        in >> *(biases + i);

    in.close();
    return true;
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output, Lab * _M, Lab * _N, Real * _weights, Real * _biases)
{
    #pragma omp single
    for (int j=0; j<nInputs; j++)
        *(_N->outvals +j) = _input[j];
    
    for (int j=0; j<nLayers; j++)
        layers[j]->propagate(_M,_N,_weights,_biases);
    
    #pragma omp single
    if (static_cast<int>(_output.size())==nOutputs)
    for (int j=0; j<nOutputs; j++)
        _output[j] = *(_N->outvals +iOutputs +j);
}

void Network::computeGrads(const vector<Real>& _error, Lab * _M, Lab * _N, Grads * _grad)
{
    #pragma omp single
    for (int j=0; j<nOutputs; j++)
        *(_N->errvals +iOutputs +j) = _error[j];
    
    for (int j=1; j<=nLayers; j++)
        //layers[nLayers-j]->backPropagate(_M,_N,_grad,weights,biases);
        layers[nLayers-j]->backPropagate(_M,_N,dsdw,_grad,weights,biases);
}

void Network::computeDeltasSeries(vector<Lab*>& _series, const int k)
{
    //printf("\n Series %d\n",k);
    for (int i=1; i<=nLayers; i++)
        layers[nLayers-i]->backPropagateDelta(series[k-1],series[k],series[k+1],weights,biases);
}

void Network::computeDeltasEnd(vector<Lab*>& _series, const int k)
{
    for (int i=1; i<=nLayers; i++)
        layers[nLayers-i]->backPropagateDelta(series[k-1],series[k],weights,biases);
}

void Network::computeGradsSeries(vector<Lab*>& _series, const int k, Grads * _grad)
{
    for (int i=0; i<nLayers; i++)
        layers[i]->backPropagateGrads(series[k-1],series[k],dsdw,_grad);
}

void Network::computeGradsLightSeries(vector<Lab*>& _series, const int k, Grads * _grad)
{
    //printf("\n Series %d\n",k);
    for (int i=0; i<nLayers; i++)
        layers[i]->backPropagateGradsLight(series[k-1],series[k],_grad);
}

void Network::updateFrozenWeights()
{
    #pragma omp single
    if (allocatedFrozenWeights==false)
    {
        _myallocate(frozen_weights, nWeights)
        _myallocate(frozen_biases,   nBiases)
        allocatedFrozenWeights = true;
    }
    #pragma omp for nowait
    for (int j=0; j<nWeights; j+=SIMD)
        #if SIMD == 1
        *(frozen_weights + j) = *(weights + j);
        #else
        STORE (frozen_weights + j, LOAD(weights + j));
        #endif
    
    #pragma omp for
    for (int j=0; j<nBiases; j+=SIMD)
        #if SIMD == 1
        *(frozen_biases + j) = *(biases + j);
        #else
        STORE (frozen_biases + j, LOAD(biases + j));
        #endif
}

void Network::expandMemory(Mem * _M, Lab * _N)
{
    std::swap(_N->outvals,_M->outvals);
    std::swap(_N->ostates,_M->ostates);
}

void Network::freshSeries(int _k, vector<Lab*> & _series)
{
    if (static_cast<int>(_series.size()) <= _k)
    {
        Lab * ns = new Lab(nNeurons,nStates);
        series.push_back(ns);
    }
}

void Network::clearInputs(Lab * _N)
{
    #if SIMD == 1
    for (int j=0; j<nNeurons; j++)
        *(_N->in_vals +j) = 0.; /* everything here is a += */
    for (int j=0; j<nStates; j++)
    {
        *(_N->iIGates +j) = 0.; /* everything here is a += */
        *(_N->iFGates +j) = 0.;
        *(_N->iOGates +j) = 0.;
    }
    #else
    const vec zeros = SET0 ();
    for (int j=0; j<nNeurons; j+=SIMD)
        STREAM (_N->in_vals +j,zeros);
    for (int j=0; j<nStates; j+=SIMD)
    {
        STREAM (_N->iIGates +j,zeros);
        STREAM (_N->iFGates +j,zeros);
        STREAM (_N->iOGates +j,zeros);
    }
    #endif
}

void Network::clearErrors(Lab * _N)
{
    #if SIMD != 1
    const vec zeros = SET0 ();
    #endif
    #pragma omp for nowait
    for (int j=0; j<nNeurons; j+=SIMD)
    {
        #if SIMD == 1
        *(_N->errvals +j) = 0.; /* everything here is a += */
        #else
        STREAM (_N->errvals +j,zeros);
        #endif
    }
    #pragma omp for
    for (int j=0; j<nStates; j+=SIMD)
    {
        #if SIMD == 1
        *(_N->eOGates +j) = 0.; /* everything here is a += */
        *(_N->eIGates +j) = 0.;
        *(_N->eFGates +j) = 0.;
        *(_N->eMCell +j) = 0.;
        #else
        STREAM (_N->eOGates +j,zeros);
        STREAM (_N->eIGates +j,zeros);
        STREAM (_N->eFGates +j,zeros);
        STREAM (_N->eMCell +j,zeros);
        #endif
    }
}

void Network::clearDsdw(Dsdw * _dsdw)
{
    #if SIMD != 1
    const vec zeros = SET0 ();
    #endif
    #pragma omp for nowait
    for (int j=0; j<ndSdW; j+=SIMD)
    {
        #if SIMD == 1
        *(_dsdw->IN +j) = 0.;
        *(_dsdw->IG +j) = 0.;
        *(_dsdw->FG +j) = 0.;
        #else
        STREAM (_dsdw->IN +j,zeros);
        STREAM (_dsdw->IG +j,zeros);
        STREAM (_dsdw->FG +j,zeros);
        #endif
    }
    
    #pragma omp for
    for (int n=0; n<ndSdB; n+=SIMD)
    {
        #if SIMD == 1
        *(_dsdw->DB +n) = 0.;
        #else
        const vec zeros = SET0 ();
        STREAM (_dsdw->DB +n,zeros);
        #endif
    }
}

void Network::clearMemory(Real * _outvals, Real * _ostates)
{
    #if SIMD != 1
    const vec zeros = SET0 ();
    #endif
    #pragma omp for nowait
    for (int j=0; j<nNeurons; j+=SIMD)
        #if SIMD == 1
        *(_outvals +j) = 0.;
        #else
        STREAM (_outvals +j,zeros);
        #endif
    
    #pragma omp for nowait
    for (int j=0; j<nStates; j+=SIMD)
        #if SIMD == 1
        *(_ostates +j) = 0.;
        #else
        STREAM (_ostates +j,zeros);
        #endif
}

/*
 void inline Network::predict(const vector<Real>& _input, vector<Real>& _output, Lab * _M, Lab * _N)
 {
 #pragma omp single
 for (int j=0; j<nInputs; j++)
 *(_N->outvals +j) = _input[j];
 
 for (int j=0; j<nLayers; j++)
 layers[j]->propagate(_M,_N,weights,biases);
 
 #pragma omp single
 if (_output.size()==nOutputs)
 for (int j=0; j<nOutputs; j++)
 _output[j] = *(_N->outvals +iOutputs +j);
 }
 */
