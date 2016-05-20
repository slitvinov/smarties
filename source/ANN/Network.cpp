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
#define SCAL 6.
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
        if(u_d_u>0)
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
        *(biases +w) =  //-2.; /* IG starts decisively closed */
                        dis(gen)*sqrt(SCAL)/Real(g.recurrSize);
        
    for (int w=g.biasFG; w<g.biasFG+g.recurrSize; w++)
        *(biases +w) =  // 2.; /* FG starts decisively open */
                         dis(gen)*sqrt(SCAL)/Real(g.recurrSize);
    
    for (int w=g.biasOG; w<g.biasOG+g.recurrSize; w++)
        *(biases +w) =  //-2.; /* OG starts decisively closed */
                         dis(gen)*sqrt(SCAL)/Real(g.recurrSize);
}

void Network::addNormal(Graph * p, Graph * g, bool first, bool last)
{
#ifdef SIMDKERNELS
    int normalSize_SIMD = ceil((Real)g->normalSize/SIMD)*SIMD;
#else
    int normalSize_SIMD = g->normalSize;
#endif
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
        { //conntected  to current realization of current rec layer
            Link * link = new Link(g->recurrSize,g->recurrPos,g->normalSize,g->normalPos,nWeights,g->first);
            
            g->nl_c_l->push_back(link);
            g->rl_l_c->push_back(link);
            
            nWeights += g->recurrSize*g->normalSize;
        }
        
        if (false)//(!last)
        { //conntected  to past realization of current normal layer
            Link * link = new Link(g->normalSize,g->normalPos,g->normalSize,g->normalPos,nWeights,g->first);
            
            g->nl_o_l->push_back(link);
            g->nl_l_f->push_back(link);
            
            nWeights += g->normalSize*g->normalSize;
        }
        
        NormalLayer * l;
        ActivationFunction * f;
        
#ifndef _scaleR_
        if (last)
            f = new Linear;
        else
#endif
            f = new SoftSign;
        
        l = new NormalLayer(g->normalSize, g->normalPos, g->biasHL, g->nl_c_l, g->nl_l_c, g->nl_l_f, f, last);
        l->prev_input_links = g->nl_o_l;
        layers.push_back(l);
    }
}

void Network::addLSTM(Graph * p, Graph * g, bool first, bool last)
{
    if (last) die("How the fudge did you get here?\n");
    
#ifdef SIMDKERNELS
    int normalSize_SIMD = ceil((Real)g->normalSize/SIMD)*SIMD; //g->normalSize
    int recurrSize_SIMD = ceil((Real)g->recurrSize/SIMD)*SIMD; //g->recurrSize
#else
    int normalSize_SIMD = g->normalSize;
    int recurrSize_SIMD = g->recurrSize;
#endif
    
    if(!last && !first)          g->recurrSize = recurrSize_SIMD;
    if (last && g->normalSize>0) g->recurrSize = recurrSize_SIMD;
    g->normalSize = normalSize_SIMD;
    g->recurrSize = recurrSize_SIMD; //TODO
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

        { //conntected to past realization of current recurrent layer
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
        if (false)//(g->normalSize>0) //NOT VALIDATED/SOMWETHING WRONG IN THE EQs
        { //conntected to past realization of current normal layer
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
        
        l = new LSTMLayer(g->recurrSize, g->recurrPos, g->indState, g->wPeep, g->biasIN, g->biasIG, g->biasFG, g->biasOG, g->dSdB, g->rl_c_l, g->rl_o_l, g->rl_l_c, g->rl_l_f, f, last);
        layers.push_back(l);
    }
}

Network::Network(vector<int>& normalSize, vector<int>& recurrSize, Settings & settings) :
Pdrop(settings.nnPdrop), nInputs(0), nOutputs(0), nLayers(0), nNeurons(0), nWeights(0), nBiases(0), ndSdW(0), ndSdB(0), nStates(0), iOutputs(0),
allocatedFrozenWeights(false), allocatedDroputWeights(false), backedUp(false), gen(settings.randSeed), bDump(not settings.bTrain)
{
    if(normalSize.size()<3)
        die("Put at least one hidden layer, would you kindly? \n");
    if(recurrSize.back()>0)
        die("Put just a normal layer as output, would you kindly? TODO \n");
    if(recurrSize.front()>0)
        die("Put just a normal layer as input, would you kindly? \n");
    mt19937 gen(settings.randSeed);
    int nMixedLayers = normalSize.size();
    nInputs = normalSize.front();
    nOutputs = (normalSize.back()==0) ? recurrSize.back() : normalSize.back();
    
    {
        Graph * g = new Graph();
        g->first = true;
        g->normalSize = nInputs;
        #ifdef SIMDKERNELS
        nNeurons += ceil((Real)nInputs/SIMD)*SIMD;
        #else
        nNeurons += nInputs;
        #endif
        G.push_back(g);
    }
    
    for (int i=1; i<nMixedLayers; i++)
    { //layer 0 is the input layer
        Graph * g = new Graph();
        bool first = i==1; bool last = i+1==nMixedLayers;
        g->recurrSize = recurrSize[i];
        g->normalSize = normalSize[i];
        if (recurrSize[i]>0) addLSTM(G.back(),g,first,last);
        if (normalSize[i]>0) addNormal(G.back(),g,first,last);
        G.push_back(g);
    }
    
    iOutputs = (normalSize.back()==0) ? G.back()->recurrPos : G.back()->normalPos;
    nLayers = layers.size();
    printf("nNeurons= %d, nWeights= %d, nBiases= %d, ndSdW= %d, ndSdB= %d, nStates= %d iOutputs = %d\n, nInputs = %d, nOutputs = %d ",
           nNeurons, nWeights, nBiases, ndSdW, ndSdB, nStates, iOutputs, nInputs, nOutputs);
    
    for (int i=0; i<settings.nAgents; ++i)
    {
        Mem * m = new Mem(nNeurons, nStates);
        clearMemory(m->outvals, m->ostates);
        mem.push_back(m);
    }
    dump_ID.resize(settings.nAgents);
    dsdw = new Dsdw(ndSdW,ndSdB);
    grad = new Grads(nWeights,nBiases);

    allocateSeries(1);
    
    _myallocate(weights, nWeights)
    _myallocate(biases,   nBiases)
    
    for (int i=1; i<static_cast<int>(G.size()); i++) initializeWeights(*G[i], gen);
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
    {
        if (std::isnan(*(weights + i)) || std::isinf(*(weights + i)))
            *(weights + i) = 0.0;
        out << *(weights + i) << "\n";
    }
    
    for (int i=0; i<nBiases; i++)
    {
        if (std::isnan(*(biases + i)) || std::isinf(*(biases + i)))
            *(biases + i) = 0.0;
        out << *(biases + i) << "\n";
    }

    out.flush();
    out.close();
    
    // Prepare copying command
    string command = "cp ";
    string nameOriginal = fname;
    command = command + nameBackup + " " + nameOriginal;
    
    // Submit the command to the system
    system(command.c_str());
}

void Network::dump(const int agentID)
{
    if (not bDump) return;
    char buf[500];
    sprintf(buf, "%07d", (int)dump_ID[agentID]);
    string nameNeurons  = "neuronOuts_" + to_string(agentID) + "_" + string(buf) + ".dat";
    string nameMemories = "cellStates_" + to_string(agentID) + "_" + string(buf) + ".dat";
    string nameOut_Mems = "out_states_" + to_string(agentID) + "_" + string(buf) + ".dat";
    {
        ofstream out(nameOut_Mems.c_str());
        if (!out.good()) die("Unable to open save into file %s\n", nameOut_Mems.c_str());
        for (int j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << "\n";
        for (int j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << "\n";
        out.close();
    }
    {
        ofstream out(nameNeurons.c_str());
        if (!out.good()) die("Unable to open save into file %s\n", nameNeurons.c_str());
        for (int j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << "\n";
        out.close();
    }
    {
        ofstream out(nameMemories.c_str());
        if (!out.good()) die("Unable to open save into file %s\n", nameMemories.c_str());
        for (int j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << "\n";
        out.close();
    }
    dump_ID[agentID]++;
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
    for (int j=0; j<nInputs; j++)
        *(_N->outvals +j) = _input[j];
    
    for (int j=0; j<nLayers; j++)
        layers[j]->propagate(_M,_N,_weights,_biases);
    
    if (static_cast<int>(_output.size())==nOutputs)
    for (int j=0; j<nOutputs; j++)
        _output[j] = *(_N->outvals +iOutputs +j);
}

void Network::computeGrads(const vector<Real>& _error, Lab * _M, Lab * _N, Grads * _grad)
{
    for (int j=0; j<nOutputs; j++)
        *(_N->errvals +iOutputs +j) = _error[j];
    
    for (int j=1; j<=nLayers; j++)
        layers[nLayers-j]->backPropagate(_M,_N,_grad,weights,biases);
}

void Network::computeDeltasSeries(vector<Lab*>& _series, const int k)
{
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
    for (int i=0; i<nLayers; i++)
        layers[i]->backPropagateGradsLight(series[k-1],series[k],_grad);
}

void Network::updateFrozenWeights()
{
    if (allocatedFrozenWeights==false)
    {
        _myallocate(frozen_weights, nWeights)
        _myallocate(frozen_biases,   nBiases)
        allocatedFrozenWeights = true;
    }
    
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int j=0; j<nWeights; j+=SIMD)
        {
            #if SIMD == 1
            *(frozen_weights + j) = *(weights + j);
            #else
            STORE (frozen_weights + j, LOAD(weights + j));
            #endif
        }

        #if SIMD > 1
        #pragma omp single nowait
        for (int j=int(nWeights/SIMD)*SIMD ; j<nWeights; ++j)
            *(frozen_weights + j) = *(weights + j);
        #endif

        
        #pragma omp for nowait
        for (int j=0; j<nBiases; j+=SIMD)
        {
            #if SIMD == 1
            *(frozen_biases + j) = *(biases + j);
            #else
            STORE (frozen_biases + j, LOAD(biases + j));
            #endif
        }
        
        #if SIMD > 1
        #pragma omp single
        for (int j=int(nBiases/SIMD)*SIMD ; j<nBiases; ++j)
            *(frozen_biases + j) = *(biases + j);
        #endif
    }
}

void Network::expandMemory(Mem * _M, Lab * _N)
{
    std::swap(_N->outvals,_M->outvals);
    std::swap(_N->ostates,_M->ostates);
}

void Network::allocateSeries(int _k, vector<Lab*> & _series)
{
    for (int j=static_cast<int>(_series.size()); j<=_k; j++)
    {
        Lab * ns = new Lab(nNeurons,nStates);
        series.push_back(ns);
    }
}

void Network::clearInputs(Lab * _N)
{
    #pragma omp parallel
    {
        #if SIMD > 1
        const vec zeros = SET0 ();
        #endif
        
        #pragma omp for nowait
        for (int j=0; j<nNeurons; j+=SIMD)
            #if SIMD == 1
            *(_N->in_vals +j) = 0.;
            #else
            STREAM (_N->in_vals +j,zeros);
            #endif

        #if SIMD > 1
        #pragma omp single nowait
        for (int j=int(nNeurons/SIMD)*SIMD ; j<nNeurons; ++j)
            *(_N->in_vals +j) = 0.;
        #endif
        
        #pragma omp for nowait
        for (int j=0; j<nStates; j+=SIMD)
        {
            #if SIMD == 1
            *(_N->iIGates +j) = 0.;
            *(_N->iFGates +j) = 0.;
            *(_N->iOGates +j) = 0.;
            #else
            STREAM (_N->iIGates +j,zeros);
            STREAM (_N->iFGates +j,zeros);
            STREAM (_N->iOGates +j,zeros);
            #endif
        }
        
        #if SIMD > 1
        #pragma omp single
        for (int j=int(nStates/SIMD)*SIMD ; j<nStates; ++j)
        {
            *(_N->iIGates +j) = 0.;
            *(_N->iFGates +j) = 0.;
            *(_N->iOGates +j) = 0.;
        }
        #endif
    }
}

void Network::clearErrors(Lab * _N)
{
    #pragma omp parallel
    {
        #if SIMD > 1
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
        
        #if SIMD > 1
        #pragma omp single nowait
        for (int j=int(nNeurons/SIMD)*SIMD ; j<nNeurons; ++j)
            *(_N->errvals +j) = 0.;
        #endif
        
        #pragma omp for nowait
        for (int j=0; j<nStates; j+=SIMD)
        {
            #if SIMD == 1
            *(_N->eOGates +j) = 0.; // everything here is a +=
            *(_N->eIGates +j) = 0.;
            *(_N->eFGates +j) = 0.;
            *(_N->eMCell  +j) = 0.;
            #else
            STREAM (_N->eOGates +j,zeros);
            STREAM (_N->eIGates +j,zeros);
            STREAM (_N->eFGates +j,zeros);
            STREAM (_N->eMCell +j,zeros);
            #endif
        }
        
        #if SIMD > 1
        #pragma omp single
        for (int j=int(nStates/SIMD)*SIMD ; j<nStates; ++j)
        {
            *(_N->eOGates +j) = 0.; // everything here is a +=
            *(_N->eIGates +j) = 0.;
            *(_N->eFGates +j) = 0.;
            *(_N->eMCell  +j) = 0.;
        }
        #endif
    }
}

void Network::clearDsdw(Dsdw * _dsdw)
{
    #pragma omp parallel
    {
        #if SIMD > 1
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
        
        #if SIMD > 1
        #pragma omp single nowait
        for (int j=int(ndSdW/SIMD)*SIMD ; j<ndSdW; ++j)
        {
            *(_dsdw->IN +j) = 0.;
            *(_dsdw->IG +j) = 0.;
            *(_dsdw->FG +j) = 0.;
        }
        #endif
        
        #pragma omp for
        for (int j=0; j<ndSdB; j+=SIMD)
        {
            #if SIMD == 1
            *(_dsdw->DB +j) = 0.;
            #else
            const vec zeros = SET0 ();
            STREAM (_dsdw->DB +j,zeros);
            #endif
        }
        
        #if SIMD > 1
        #pragma omp single nowait
        for (int j=int(ndSdB/SIMD)*SIMD ; j<ndSdB; ++j)
            *(_dsdw->DB +j) = 0.;
        #endif
    }
}

void Network::clearMemory(Real * _outvals, Real * _ostates)
{
    #pragma omp parallel
    {
        #if SIMD > 1
        const vec zeros = SET0 ();
        #endif
        
        #pragma omp for nowait
        for (int j=0; j<nNeurons; j+=SIMD)
            #if SIMD == 1
            *(_outvals +j) = 0.;
            #else
            STREAM (_outvals +j,zeros);
            #endif

        #if SIMD > 1
        #pragma omp single nowait
        for (int j=int(nNeurons/SIMD)*SIMD ; j<nNeurons; ++j)
            *(_outvals +j) = 0.;
        #endif
        
        #pragma omp for nowait
        for (int j=0; j<nStates; j+=SIMD)
            #if SIMD == 1
            *(_ostates +j) = 0.;
            #else
            STREAM (_ostates +j,zeros);
            #endif
        
        #if SIMD > 1
        #pragma omp single
        for (int j=int(nStates/SIMD)*SIMD ; j<nStates; ++j)
            *(_ostates +j) = 0.;
        #endif
    }
}

void Network::assignDropoutMask()
{
    if (Pdrop > 0)
    {
        assert(Pdrop>0 && Pdrop<1 && backedUp==false);
        if (allocatedDroputWeights==false)
        {
            _myallocate(weights_DropoutBackup, nWeights)
            allocatedDroputWeights = true;
        }
        //backup the weights
        swap(weights_DropoutBackup,weights);
        backedUp = true;
        //probability of having a true in the bernoulli distrib:
        Real Pkeep = 1. - Pdrop;
        Real fac = 1./Pkeep; //the others have to compensate
        
        //seeds for the rng
#ifdef  _useOMP_
        const int nSeeds = omp_get_max_threads();
        std::vector<unsigned> Seeds(nSeeds);
        { //soo many seeds
            if (nSeeds<1) die("ma povcaputtana\n");
            std::vector<unsigned> seedsSeeds(nSeeds);
            uniform_real_distribution<Real> dis(-1e6,1e6);
            for (int i(0); i<nSeeds; i++) seedsSeeds[i] = dis(gen);
            std::seed_seq genSeeds(seedsSeeds.begin(),seedsSeeds.end());
            genSeeds.generate(Seeds.begin(),Seeds.end());
        }
        #pragma omp parallel
        {
            const int me = omp_get_thread_num();
            if (me>=nSeeds) die("ma povcamiseria\n");
            bernoulli_distribution dis(Pkeep);
            mt19937 g(Seeds[me]);
            #pragma omp for
            for (int j=0; j<nWeights; j++) //TODO: betterer, simder, paralleler
            {
                bool res = dis(g);
                *(weights + j) = (res) ? *(weights_DropoutBackup + j)*fac : 0.;
            }
        }
#else
        bernoulli_distribution dis(Pkeep);
        for (int j=0; j<nWeights; j++) //TODO: betterer, simder, paralleler
        {
            bool res = dis(gen);
            *(weights + j) = (res) ? *(weights_DropoutBackup + j)*fac : 0.;
        }
#endif
    }
}

void Network::removeDropoutMask()
{
    if (allocatedDroputWeights && backedUp)
    {
        swap(weights_DropoutBackup,weights);
        backedUp = false;
    }
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
