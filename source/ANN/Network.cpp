/*
 *  LSTMNet.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include "Network.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>
#define SCAL 6.
using namespace ErrorHandling;

void Network::orthogonalize(const int nO, const int nI, const int n0, Real* const _weights)
{
    return;
    if (nI>=nO)
    for (int i=1; i<nO; i++)
    for (int j=0; j<i;  j++) {
        Real u_d_u = 0.0;
        Real v_d_u = 0.0;
        for (int k=0; k<nI; k++) {
            u_d_u += *(_weights +n0 +j*nI +k)* *(_weights +n0 +j*nI +k);
            v_d_u += *(_weights +n0 +j*nI +k)* *(_weights +n0 +i*nI +k);
        }
        if(u_d_u>0)
            for (int k=0; k<nI; k++)
            *(_weights+n0+i*nI+k) -= (v_d_u/u_d_u) * *(_weights+n0+j*nI+k);
    }
}

void Network::initializeWeights(Graph & g, Real* const _weights, Real* const _biases)
{
    uniform_real_distribution<Real> dis(-1.,1.);
    
    for (int w=g.wPeep; w<(g.wPeep + 3*g.recurrSize); w++)
        *(_weights +w) = dis(*gen)*sqrt(SCAL)/Real(g.recurrSize);
    
    for (const auto & l : *g.nl_c_l)
    {
        for (int w=l->iW ; w<(l->iW + l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iW,_weights);
    }
    
    for (const auto & l : *g.nl_o_l)
    {
        for (int w=l->iW ; w<(l->iW + l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iW,_weights);
    }
    
    for (const auto & l : *g.rl_c_l)
    {
        for (int w=l->iW ; w<(l->iW + l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iW,_weights);
        
        for (int w=l->iWI; w<(l->iWI+ l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWI,_weights);
        
        for (int w=l->iWF; w<(l->iWF+ l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWF,_weights);
        
        for (int w=l->iWO; w<(l->iWO+ l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWO,_weights);
    }
    
    for (const auto & l : *g.rl_o_l)
    {
        for (int w=l->iW ; w<(l->iW + l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iW,_weights);
        
        for (int w=l->iWI; w<(l->iWI+ l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWI,_weights);
        
        for (int w=l->iWF; w<(l->iWF+ l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWF,_weights);
        
        for (int w=l->iWO; w<(l->iWO+ l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen)*sqrt(SCAL)/Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWO,_weights);
    }

    for (int w=g.biasHL; w<g.biasHL+g.normalSize; w++)
        *(_biases +w) = dis(*gen)*sqrt(SCAL)/Real(g.normalSize);
    
    for (int w=g.biasIN; w<g.biasIN+g.recurrSize; w++)
        *(_biases +w) = dis(*gen)*sqrt(SCAL)/Real(g.recurrSize);
        
    for (int w=g.biasIG; w<g.biasIG+g.recurrSize; w++)
        *(_biases +w) = dis(*gen)*sqrt(SCAL)/Real(g.recurrSize)+2.;
        
    for (int w=g.biasFG; w<g.biasFG+g.recurrSize; w++)
        *(_biases +w) = dis(*gen)*sqrt(SCAL)/Real(g.recurrSize)+2.;
    
    for (int w=g.biasOG; w<g.biasOG+g.recurrSize; w++)
        *(_biases +w) = dis(*gen)*sqrt(SCAL)/Real(g.recurrSize)+2.;
}

void Network::addNormal(Graph* const p, Graph* const g, const bool first, const bool last)
{
#ifdef SIMDKERNELS
    g->normalSize_SIMD = ceil((Real)g->normalSize/SIMD)*SIMD;
#else
    g->normalSize_SIMD = g->normalSize;
#endif
    
    if (g->normalSize>0)
    {
        g->normalPos = nNeurons;
        nNeurons += g->normalSize_SIMD;
        g->biasHL = nBiases;
        nBiases += g->normalSize_SIMD;
        
        if (p->recurrSize>0) { //conntected to previous recurrent layer
            Link * link = new Link(p->recurrSize_SIMD,p->recurrPos,g->normalSize,g->normalPos,nWeights);
            g->nl_c_l->push_back(link);
            p->rl_l_c->push_back(link);
            nWeights += p->recurrSize_SIMD*g->normalSize;
        }
        
        if (p->normalSize>0) { //conntected to previous normal layer
            Link * link = new Link(p->normalSize_SIMD,p->normalPos,g->normalSize,g->normalPos,nWeights);
            g->nl_c_l->push_back(link);
            p->nl_l_c->push_back(link);
            nWeights += p->normalSize_SIMD*g->normalSize;
        }
        
        if (g->recurrSize>0) { //conntected  to current realization of current rec layer
            Link * link = new Link(g->recurrSize_SIMD,g->recurrPos,g->normalSize,g->normalPos,nWeights);
            g->nl_o_l->push_back(link);
            g->rl_l_c->push_back(link);
            nWeights += g->recurrSize_SIMD*g->normalSize;
        }
        
        if (false) { //(!last) //conntected  to past realization of current normal layer
            Link * link = new Link(g->normalSize_SIMD,g->normalPos,g->normalSize,g->normalPos,nWeights);
            g->nl_o_l->push_back(link);
            g->nl_l_f->push_back(link);
            nWeights += g->normalSize_SIMD*g->normalSize;
        }
        
#ifndef _scaleR_
        const Activation * f = (last) ? new Activation : new SoftSign;
#else
        const Activation * f = new SoftSign;
#endif

        NormalLayer * l = new NormalLayer(g->normalSize, g->normalPos, g->biasHL, g->nl_c_l, g->nl_o_l, g->nl_l_c, g->nl_l_f, f, last);
        l->prev_input_links = g->nl_o_l;
        layers.push_back(l);
    }
}

void Network::addLSTM(Graph* const p, Graph* const g, const bool first, const bool last)
{
#ifdef SIMDKERNELS
    g->recurrSize_SIMD = ceil((Real)g->recurrSize/SIMD)*SIMD;
#else
    g->recurrSize_SIMD = g->recurrSize;
#endif
    
    if (g->recurrSize>0) {
        {
            g->recurrPos = nNeurons;
            nNeurons += g->recurrSize_SIMD;
            g->indState = nStates;
            nStates  += g->recurrSize_SIMD;
            
            g->biasIN = nBiases;
            g->biasIG = g->biasIN + g->recurrSize_SIMD;
            g->biasFG = g->biasIG + g->recurrSize_SIMD;
            g->biasOG = g->biasFG + g->recurrSize_SIMD;
            nBiases += 4*g->recurrSize;
            
            //g->wPeep  = nWeights;
            //nWeights+= 3*g->recurrSize_SIMD; //SIMD HAZARD
        }
        
        if (p->recurrSize>0) { //conntected to previous recurrent layer
            int WeightHL = nWeights;
            nWeights += p->recurrSize_SIMD*g->recurrSize;
            int WeightIG = nWeights;
            nWeights += p->recurrSize_SIMD*g->recurrSize;
            int WeightFG = nWeights;
            nWeights += p->recurrSize_SIMD*g->recurrSize;
            int WeightOG = nWeights;
            nWeights += p->recurrSize_SIMD*g->recurrSize;
            
            Link * link = new Link(p->recurrSize_SIMD,p->recurrPos,g->recurrSize,g->recurrPos,g->indState,WeightHL,WeightIG,WeightFG,WeightOG);
            g->rl_c_l->push_back(link);
            p->rl_l_c->push_back(link);
        }
        if (p->normalSize>0) { //conntected to previous normal layer
            int WeightHL = nWeights;
            nWeights += p->normalSize_SIMD*g->recurrSize;
            int WeightIG = nWeights;
            nWeights += p->normalSize_SIMD*g->recurrSize;
            int WeightFG = nWeights;
            nWeights += p->normalSize_SIMD*g->recurrSize;
            int WeightOG = nWeights;
            nWeights += p->normalSize_SIMD*g->recurrSize;
            
            Link * link = new Link(p->normalSize_SIMD,p->normalPos,g->recurrSize,g->recurrPos,g->indState,WeightHL,WeightIG,WeightFG,WeightOG);
            g->rl_c_l->push_back(link);
            p->nl_l_c->push_back(link);
        }

        { //conntected to past realization of current recurrent layer
            int WeightHL = nWeights;
            nWeights += g->recurrSize_SIMD*g->recurrSize;
            int WeightIG = nWeights;
            nWeights += g->recurrSize_SIMD*g->recurrSize;
            int WeightFG = nWeights;
            nWeights += g->recurrSize_SIMD*g->recurrSize;
            int WeightOG = nWeights;
            nWeights += g->recurrSize_SIMD*g->recurrSize;
            
            Link * link = new Link(g->recurrSize_SIMD,g->recurrPos,g->recurrSize,g->recurrPos,g->indState,WeightHL,WeightIG,WeightFG,WeightOG);
            g->rl_o_l->push_back(link);
            g->rl_l_f->push_back(link);
        }
        if (false)//(g->normalSize>0) //NOT VALIDATED/SOMWETHING WRONG IN THE EQs
        { //conntected to past realization of current normal layer
            int WeightHL = nWeights;
            nWeights += g->normalSize_SIMD*g->recurrSize;
            int WeightIG = nWeights;
            nWeights += g->normalSize_SIMD*g->recurrSize;
            int WeightFG = nWeights;
            nWeights += g->normalSize_SIMD*g->recurrSize;
            int WeightOG = nWeights;
            nWeights += g->normalSize_SIMD*g->recurrSize;
            
            Link * link = new Link(g->normalSize_SIMD,g->normalPos,g->recurrSize,g->recurrPos,g->indState,WeightHL,WeightIG,WeightFG,WeightOG);
            g->rl_o_l->push_back(link);
            g->nl_l_f->push_back(link);
        }
        
#ifndef _scaleR_
        const Activation * fI = (last) ? new Activation : new SoftSign2;
        const Activation * fG = new SoftSigm;
        const Activation * fO = (last) ? new Activation : new HardSign(2.);
#else
        const Activation * fI = new SoftSign2;
        const Activation * fG = new SoftSigm;
        const Activation * fO = new HardSign(2.);
#endif
        
        NormalLayer * l = new LSTMLayer(g->recurrSize, g->recurrPos, g->indState, g->wPeep, g->biasIN, g->biasIG, g->biasFG, g->biasOG, g->rl_c_l, g->rl_o_l, g->rl_l_c, g->rl_l_f, fI, fG, fO, last);
        layers.push_back(l);
    }
}

Network::Network(const vector<int>& normalSize, const vector<int>& recurrSize, const Settings & settings) :
Pdrop(settings.nnPdrop), nInputs(normalSize.front()), nOutputs(normalSize.back()),
nLayers(0), nNeurons(0), nWeights(0), nBiases(0), nStates(0), iOutputs(0),
allocatedFrozenWeights(false), allocatedDroputWeights(false), backedUp(false),
gen(settings.gen), bDump(not settings.bTrain)
{
    if(normalSize.size()<3) die("Put at least one hidden layer. \n");
    if(recurrSize.front()>0) die("Put just a normal layer as input. \n");
    const int nMixedLayers = normalSize.size();
    
    {
        Graph * g = new Graph();
        g->first = true;
        g->normalSize = nInputs;
        #ifdef SIMDKERNELS
        nNeurons += ceil((Real)nInputs/SIMD)*SIMD;
        #else
        nNeurons += nInputs;
        #endif
        g->normalSize_SIMD = g->normalSize_SIMD;
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
    printf("nNeurons= %d, nWeights= %d, nBiases= %d, nStates= %d iOutputs = %d\n, nInputs = %d, nOutputs = %d \n",
           nNeurons, nWeights, nBiases, nStates, iOutputs, nInputs, nOutputs);
    
    for (int i=0; i<settings.nAgents; ++i) {
        Mem * m = new Mem(nNeurons, nStates);
        clearMemory(m->outvals, m->ostates);
        mem.push_back(m);
    }
    
    dump_ID.resize(settings.nAgents);
    grad = new Grads(nWeights,nBiases);
    
    if (settings.nThreads>1) {
        Vgrad.resize(settings.nThreads);
        //Vbiases.resize(settings.nThreads);
        //Vweights.resize(settings.nThreads);
        //VFbiases.resize(settings.nThreads);
        //VFweights.resize(settings.nThreads);
        for (int i=0; i<settings.nThreads; ++i) {
            Vgrad[i] = new Grads(nWeights,nBiases);
            //_allocateQuick(VFweights[i], nWeights)
            //_allocateQuick(Vweights[i], nWeights)
            //_allocateQuick(VFbiases[i], nBiases)
            //_allocateQuick(Vbiases[i], nBiases)
        }
    } else {
        _grad = new Grads(nWeights,nBiases);
    }
    
    allocateSeries(3);
    
    _allocateClean(weights, nWeights)
    _allocateClean(biases,   nBiases)
    
    /*
     _allocateClean(frozen_weights, nWeights)
     _allocateClean(frozen_biases, nBiases)
     allocatedFrozenWeights = true;
     */
    
    for (int i=1; i<static_cast<int>(G.size()); i++) {
        initializeWeights(*G[i], weights, biases);
        //initializeWeights(*G[i], frozen_weights, frozen_biases);
    }
    updateFrozenWeights();
    //synchronizeWeights();
}

Network::Network(const vector<int>& layerSize, const bool bLSTM, const Settings & settings) :
Pdrop(settings.nnPdrop), nInputs(layerSize.front()), nOutputs(layerSize.back()),
nLayers(0), nNeurons(0), nWeights(0), nBiases(0), nStates(0), iOutputs(0),
allocatedFrozenWeights(false), allocatedDroputWeights(false), backedUp(false),
gen(settings.gen), bDump(not settings.bTrain)
{
    const int nMixedLayers = layerSize.size();
    if(nMixedLayers<3) die("Put at least one hidden layer. \n");
    if(nInputs<1) die("No inputs. \n");
    
    {
        Graph * g = new Graph();
        g->first = true;
        g->normalSize = nInputs;
        #ifdef SIMDKERNELS
        g->normalSize_SIMD = ceil((Real)nInputs/SIMD)*SIMD;
        nNeurons += ceil((Real)nInputs/SIMD)*SIMD;
        #else
        g->normalSize_SIMD = nInputs;
        nNeurons += nInputs;
        #endif
        G.push_back(g);
    }
    
    for (int i=1; i<nMixedLayers; i++) { //layer 0 is the input layer
        Graph * g = new Graph();
        bool first = i==1; bool last = i+1==nMixedLayers;
        if (bLSTM && not last) { //
            g->recurrSize = layerSize[i];
            g->normalSize = 0;
            addLSTM(G.back(),g,first,last);
        } else {
            g->normalSize = layerSize[i];
            g->recurrSize = 0;
            addNormal(G.back(),g,first,last);
        }
        G.push_back(g);
    }
    
    //iOutputs = (bLSTM) ? G.back()->recurrPos : G.back()->normalPos;
    iOutputs = G.back()->normalPos;
    nLayers = layers.size();
    printf("nNeurons= %d, nWeights= %d, nBiases= %d, nStates= %d iOutputs = %d\n, nInputs = %d, nOutputs = %d \n", 
           nNeurons, nWeights, nBiases, nStates, iOutputs, nInputs, nOutputs);
    
    for (int i=0; i<settings.nAgents; ++i) {
        Mem * m = new Mem(nNeurons, nStates);
        clearMemory(m->outvals, m->ostates);
        mem.push_back(m);
    }
    dump_ID.resize(settings.nAgents);
    allocateSeries(3);
    
    grad = new Grads(nWeights,nBiases);
    _allocateClean(weights, nWeights)
    _allocateClean(biases, nBiases)
    
    if (settings.nThreads>1) {
        Vgrad.resize(settings.nThreads);
        //Vbiases.resize(settings.nThreads);
        //Vweights.resize(settings.nThreads);
        //VFbiases.resize(settings.nThreads);
        //VFweights.resize(settings.nThreads);
        for (int i=0; i<settings.nThreads; ++i) {
            Vgrad[i] = new Grads(nWeights,nBiases);
            //_allocateQuick(VFweights[i], nWeights)
            //_allocateQuick(Vweights[i], nWeights)
            //_allocateQuick(VFbiases[i], nBiases)
            //_allocateQuick(Vbiases[i], nBiases)
        }
    } else {
        _grad = new Grads(nWeights,nBiases);
    }
    
    
    /*
    _allocateClean(frozen_weights, nWeights)
    _allocateClean(frozen_biases, nBiases)
    allocatedFrozenWeights = true;
     */
    
    for (int i=1; i<static_cast<int>(G.size()); i++) {
        initializeWeights(*G[i], weights, biases);
        //initializeWeights(*G[i], frozen_weights, frozen_biases);
    }
    updateFrozenWeights();
    //synchronizeWeights();
}

void Network::save(const string fname)
{
    printf("Saving into %s\n", fname.c_str());
    fflush(0);
    string nameBackup = fname + "_tmp";
    ofstream out(nameBackup.c_str());
    
    if (!out.good()) die("Unable to open save into file %s\n", fname.c_str());
    
    out.precision(20);
    out << nWeights << " "  << nBiases << " " << nLayers  << " " << nNeurons << endl;
    
    for (int i=0; i<nWeights; i++) {
        if (std::isnan(*(weights + i)) || std::isinf(*(weights + i))) {
            *(weights + i) = 0.0;
            out << 0.0 << "\n";
        } else {
            out << *(weights + i) << "\n";
        }
    }
    
    for (int i=0; i<nBiases; i++) {
       if (std::isnan(*(biases + i)) || std::isinf(*(biases + i))) {
           *(biases + i) = 0.0;
            out << 0.0 << "\n";
        } else {
            out << *(biases + i) << "\n";
        }
    }

    out.flush();
    out.close();
    
    //Prepare copying command
    string command = "cp " + nameBackup + " " + fname;
    
    //Submit the command to the system
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
        for (int j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << " ";
        for (int j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << " ";
        out << "\n";
        out.close();
    }
    {
        ofstream out(nameNeurons.c_str());
        if (!out.good()) die("Unable to open save into file %s\n", nameNeurons.c_str());
        for (int j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << " ";
        out << "\n";
        out.close();
    }
    {
        ofstream out(nameMemories.c_str());
        if (!out.good()) die("Unable to open save into file %s\n", nameMemories.c_str());
        for (int j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << " ";
        out << "\n";
        out.close();
    }
    dump_ID[agentID]++;
}

bool Network::restart(const string fname)
{
    string nameBackup = fname + "_tmp";
    
    ifstream in(nameBackup.c_str());
    debug1("Reading from %s\n", nameBackup.c_str());
    if (!in.good()) {
        error("WTF couldnt open file %s (ok keep going mofo)!\n", fname.c_str());
        return false;
    }
    
    int readTotWeights, readTotBiases, readNNeurons, readNLayers;
    in >> readTotWeights  >> readTotBiases >> readNLayers >> readNNeurons;
    
    if (readTotWeights != nWeights || readTotBiases != nBiases || readNLayers != nLayers || readNNeurons != nNeurons)
    die("Network parameters differ!");
    
    Real tmp;
    for (int i=0; i<nWeights; i++) {
        in >> tmp;
        if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
        *(weights + i) = tmp;
    }
    
    for (int i=0; i<nBiases; i++) {
        in >> tmp;
        if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
        *(biases + i) = tmp;
    }

    in.close();
    
    updateFrozenWeights();
    //synchronizeWeights();
    return true;
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output, const Lab* const _M, Lab* const _N, const Real* const _weights, const Real* const _biases) const
{
    for (int j=0; j<nInputs; j++)
        *(_N->outvals +j) = _input[j];
    
    for (int j=0; j<nLayers; j++)
        layers[j]->propagate(_M,_N,_weights,_biases);
    
    assert(static_cast<int>(_output.size())==nOutputs);
    
    for (int j=0; j<nOutputs; j++)
        _output[j] = *(_N->outvals +iOutputs +j);
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output, Lab* const _N, const Real* const _weights, const Real* const _biases) const
{
    for (int j=0; j<nInputs; j++)
    *(_N->outvals +j) = _input[j];
    
    for (int j=0; j<nLayers; j++)
    layers[j]->propagate(_N,_weights,_biases);
    
    assert(static_cast<int>(_output.size())==nOutputs);
    
    for (int j=0; j<nOutputs; j++)
    _output[j] = *(_N->outvals +iOutputs +j);
}

void Network::computeGrads(const vector<Real>& _error, const Lab* const _M, Lab* const _N, Grads* const _Grad) const
{
    for (int j=0; j<nOutputs; j++)
        *(_N->errvals +iOutputs +j) = _error[j];
    
    for (int j=1; j<=nLayers; j++)
        layers[nLayers-j]->backPropagate(_M,_N,_Grad,weights,biases);
}

void Network::computeDeltasInputs(vector<Lab*>& _series, const int k, const Real* const _weights, const Real* const _biases) const
{//no weight grad to care about, no recurrent links
    for (int n=0; n<nInputs; n++) {
        Real err = 0.;
        for (const auto & l : *(G[0]->nl_l_c)) {
            if (l->LSTM)
                for (int i=0; i<l->nO; i++)
                    err+=*(series[k]->eOGates+l->iC+i)* *(_weights+l->iWO+i*l->nI+n)+
                    *(series[k]->errvals+l->iO+i)* (
                    *(series[k]->eMCell +l->iC+i)* *(_weights+l->iW +i*l->nI+n)+
                    *(series[k]->eIGates+l->iC+i)* *(_weights+l->iWI+i*l->nI+n)+
                    *(series[k]->eFGates+l->iC+i)* *(_weights+l->iWF+i*l->nI+n));
            else
                for (int i=0; i<l->nO; i++)
                    err+=*(series[k]->errvals+l->iO+i)* *(_weights+l->iW+i*l->nI +n);
        }
        *(series[k]->errvals +n) = err;
    }
}

void Network::computeDeltasSeries(vector<Lab*>& _series, const int first, const int last, const Real* const _weights, const Real* const _biases) const
{
#ifdef _BPTT_
    for (int i=1; i<=nLayers; i++) {
        layers[nLayers-i]->backPropagateDeltaLast(series[last-1],series[last],_weights,_biases);
    }
    
    for (int k=last-1; k>=first+1; k--) {
        for (int i=1; i<=nLayers; i++)
        layers[nLayers-i]->backPropagateDelta(series[k-1],series[k],series[k+1],_weights,_biases);
    }
    
    for (int i=1; i<=nLayers; i++) {
        layers[nLayers-i]->backPropagateDeltaFirst(series[first],series[first+1],_weights,_biases);
    }
#else
    for (int k=first; k>=last; k--) {
        for (int i=1; i<=nLayers; i++)
            layers[nLayers-i]->backPropagateDelta(series[k],_weights,_biases);
    }
#endif
}

void Network::computeDeltas(Lab* const _series, const Real* const _weights, const Real* const _biases) const
{
    for (int i=1; i<=nLayers; i++)
        layers[nLayers-i]->backPropagateDelta(_series,_weights,_biases);
}

void Network::computeGradsSeries(const vector<Lab*>& _series, const int k, Grads* const _Grad) const
{
    for (int i=0; i<nLayers; i++)
        layers[i]->backPropagateGrads(series[k-1],series[k],_Grad);
}

void Network::computeGrads(const Lab* const _series, Grads* const _Grad) const
{
    for (int i=0; i<nLayers; i++)
        layers[i]->backPropagateGrads(_series,_Grad);
}

void Network::computeAddGradsSeries(const vector<Lab*>& _series, const int first, const int last, Grads* const _Grad) const
{
    for (int i=0; i<nLayers; i++)
        layers[i]->backPropagateAddGrads(series[first],_Grad);
    
    for (int k=first+1; k<=last; k++) {
        for (int i=0; i<nLayers; i++)
            layers[i]->backPropagateAddGrads(series[k-1],series[k],_Grad);
    }
}

void Network::computeAddGrads(const Lab* const _series, Grads* const _Grad) const
{
    for (int i=0; i<nLayers; i++)
    layers[i]->backPropagateAddGrads(_series,_Grad);
}

void Network::updateFrozenWeights()
{
    if (allocatedFrozenWeights==false) {
        _allocateQuick(frozen_weights, nWeights)
        _allocateQuick(frozen_biases,   nBiases)
        allocatedFrozenWeights = true;
    }
    
    #pragma omp parallel
    {
        const int WsizeSIMD=ceil(nWeights/(Real)SIMD)*SIMD;
        #pragma omp for nowait
        for (int j=0; j<WsizeSIMD; j+=SIMD) {
            #if SIMD == 1
            *(frozen_weights + j) = *(weights + j);
            #else
            STORE (frozen_weights + j, LOAD(weights + j));
            #endif
        }
        
        const int BsizeSIMD=ceil(nBiases/(Real)SIMD)*SIMD;
        #pragma omp for nowait
        for (int j=0; j<BsizeSIMD; j+=SIMD) {
            #if SIMD == 1
            *(frozen_biases + j) = *(biases + j);
            #else
            STORE (frozen_biases + j, LOAD(biases + j));
            #endif
        }
    }
}

/*
void Network::synchronizeWeights()
{
    const int WsizeSIMD=ceil(nWeights/(Real)SIMD)*SIMD;
    const int BsizeSIMD=ceil(nBiases/(Real)SIMD)*SIMD;
    const int nThreads = Vweights.size();
    #if SIMD > 1
    const vec zeros = SET0 ();
    #endif
    
    #pragma omp for nowait
    for (int j=0; j<WsizeSIMD; j+=SIMD) {
        for (int k=0; k<nThreads; k++) {
            #if SIMD==1
            *(Vweights[k]+j) = *(weights+j);
            *(VFweights[k]+j) = *(frozen_weights+j);
            #else
            STORE(Vweights[k]+j, LOAD(weights+j));
            STORE(VFweights[k]+j, LOAD(frozen_weights+j));
            #endif
        }
    }
    
    #pragma omp for
    for (int j=0; j<BsizeSIMD; j+=SIMD) {
        for (int k=0; k<nThreads; k++) {
            #if SIMD==1
            *(Vbiases[k]+j) = *(biases+j);
            *(VFbiases[k]+j) = *(frozen_biases+j);
            #else
            STORE(Vbiases[k]+j, LOAD(biases+j));
            STORE(VFbiases[k]+j, LOAD(frozen_biases+j));
            #endif
        }
    }
}
 */

void Network::moveFrozenWeights(const Real alpha)
{
    if (allocatedFrozenWeights==false) updateFrozenWeights();
    
    const Real _alpha = 1. - alpha;
    #if SIMD > 1
    const vec B1 = SET1(alpha);
    const vec B2 = SET1(_alpha);
    #endif
    #pragma omp parallel
    {
        const int WsizeSIMD=ceil(nWeights/(Real)SIMD)*SIMD;
        #pragma omp for nowait
        for (int j=0; j<WsizeSIMD; j+=SIMD) {
            #if SIMD == 1
            *(frozen_weights + j) = *(frozen_weights + j)*_alpha + *(weights + j)*alpha;
            #else
            STORE(frozen_weights+j,ADD(MUL(B2,LOAD(frozen_weights+j)),MUL(B1,LOAD(weights+j))));
            #endif
        }

        const int BsizeSIMD=ceil(nBiases/(Real)SIMD)*SIMD;
        #pragma omp for nowait
        for (int j=0; j<BsizeSIMD; j+=SIMD) {
            #if SIMD == 1
            *(frozen_biases + j) = *(frozen_biases + j)*_alpha + *(biases + j)*alpha;
            #else
            STORE(frozen_biases+j,ADD(MUL(B2,LOAD(frozen_biases+j)),MUL(B1,LOAD(biases+j))));
            #endif
        }
    }
}

void Network::expandMemory(Mem * _M, Lab * _N) const
{
    std::swap(_N->outvals,_M->outvals);
    std::swap(_N->ostates,_M->ostates);
}

void Network::allocateSeries(int _k, vector<Lab*> & _series)
{
    for (int j=static_cast<int>(_series.size()); j<=_k; j++) {
        Lab * ns = new Lab(nNeurons,nStates);
        series.push_back(ns);
    }
}

void Network::clearInputs(Lab * _N)
{
    #if SIMD > 1
    const vec zeros = SET0 ();
    #endif

    for (int j=0; j<nNeurons; j+=SIMD)
        #if SIMD == 1
        *(_N->in_vals +j) = 0.;
        #else
        STREAM (_N->in_vals +j,zeros);
        #endif

    #if SIMD > 1
    for (int j=int(nNeurons/SIMD)*SIMD ; j<nNeurons; ++j)
        *(_N->in_vals +j) = 0.;
    #endif

    for (int j=0; j<nStates; j+=SIMD) {
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
    for (int j=int(nStates/SIMD)*SIMD ; j<nStates; ++j) {
        *(_N->iIGates +j) = 0.;
        *(_N->iFGates +j) = 0.;
        *(_N->iOGates +j) = 0.;
    }
    #endif
}

void Network::clearErrors(Lab * _N) const
{
    #if SIMD > 1
    const vec zeros = SET0 ();
    #endif

    for (int j=0; j<nNeurons; j+=SIMD)
    {
        #if SIMD == 1
        *(_N->errvals +j) = 0.; /* everything here is a += */
        #else
        STREAM (_N->errvals +j,zeros);
        #endif
    }
    
    #if SIMD > 1
    for (int j=int(nNeurons/SIMD)*SIMD ; j<nNeurons; ++j)
        *(_N->errvals +j) = 0.;
    #endif

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
    for (int j=int(nStates/SIMD)*SIMD ; j<nStates; ++j)
    {
        *(_N->eOGates +j) = 0.; // everything here is a +=
        *(_N->eIGates +j) = 0.;
        *(_N->eFGates +j) = 0.;
        *(_N->eMCell  +j) = 0.;
    }
    #endif
}

void Network::clearMemory(Real * _outvals, Real * _ostates) const
{
    #if SIMD > 1
    const vec zeros = SET0 ();
    #endif

    for (int j=0; j<nNeurons; j+=SIMD)
        #if SIMD == 1
        *(_outvals +j) = 0.;
        #else
        STREAM (_outvals +j,zeros);
        #endif

    #if SIMD > 1
    for (int j=int(nNeurons/SIMD)*SIMD ; j<nNeurons; ++j)
        *(_outvals +j) = 0.;
    #endif
    
    for (int j=0; j<nStates; j+=SIMD)
        #if SIMD == 1
        *(_ostates +j) = 0.;
        #else
        STREAM (_ostates +j,zeros);
        #endif
    
    #if SIMD > 1
    for (int j=int(nStates/SIMD)*SIMD ; j<nStates; ++j)
        *(_ostates +j) = 0.;
    #endif
}

void Network::assignDropoutMask()
{
    if (Pdrop > 0)
    {
        assert(Pdrop>0 && Pdrop<1 && backedUp==false);
        if (allocatedDroputWeights==false) {
            _allocateQuick(weights_DropoutBackup, nWeights)
            allocatedDroputWeights = true;
        }
        //backup the weights
        swap(weights_DropoutBackup,weights);
        backedUp = true;
        //probability of having a true in the bernoulli distrib:
        Real Pkeep = 1. - Pdrop;
        Real fac = 1./Pkeep; //the others have to compensate
        
        //seeds for the rng
        /*
#ifdef  _useOMP_
        const int nSeeds = omp_get_max_threads();
        std::vector<unsigned> Seeds(nSeeds);
        { //soo many seeds
            if (nSeeds<1) die("ma povcaputtana\n");
            std::vector<unsigned> seedsSeeds(nSeeds);
            uniform_real_distribution<Real> dis(-1e6,1e6);
            for (int i(0); i<nSeeds; i++) seedsSeeds[i] = dis(*gen);
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
        */
        bernoulli_distribution dis(Pkeep);
        for (int j=0; j<nWeights; j++) //TODO: betterer, simder, paralleler
        {
            bool res = dis(*gen);
            *(weights + j) = (res) ? *(weights_DropoutBackup + j)*fac : 0.;
        }
//#endif
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

void Network::checkGrads(const vector<vector<Real>>& inputs, const int lastn, const int ierr)
{
    //std::cout << std::setprecision(9);
    int nseries = inputs.size();
    vector<Real> res(nOutputs);
    allocateSeries(nseries+1);
    
    Grads * g = new Grads(nWeights,nBiases);
    Grads * G = new Grads(nWeights,nBiases);
    const Real eps = 1e-6;
    
    predict(inputs[0], res, series[0]);
    for (int k=1; k<lastn; k++) {
        predict(inputs[k], res, series[k-1], series[k]);
        for (int i=0; i<nOutputs; i++)
            *(series[k]->errvals +iOutputs+i) = 0.;
    }

    *(series[lastn-1]->errvals +iOutputs+ierr) = -1.;//Errors[1*nOutputs + i];
    
    computeDeltasSeries(series, 0, lastn-1);
    computeAddGradsSeries(series, 0, lastn-1, G);
    //for (int k=0; k<=lastn-2; k++) {
    //    computeGradsSeries(series, k, g);
    //    stackGrads(G,g);
    //}

    for (int w=0; w<nWeights; w++) {
        *(weights+w) += eps;
        predict(inputs[0], res, series[0]);
        for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
        const Real out1 = - *(series[lastn-1]->outvals+iOutputs+ierr);
        
        *(weights+w) -= 2*eps;
        predict(inputs[0], res, series[0]);
        for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
        const Real out2 = - *(series[lastn-1]->outvals+iOutputs+ierr);
        
        *(weights+w) += eps;
        *(g->_W+w) += (out1-out2)/(2*eps);
        
        //const Real scale = fabs(*(biases+w));
        const Real scale = max(fabs(*(G->_W+w)),fabs(*(g->_W+w)));
        const Real err = (*(G->_W+w)-*(g->_W+w))/scale;
        if (fabs(err)>1e-4) cout <<"W"<<w<<" "<<*(G->_W+w)<<" "<<*(g->_W+w)<<" "<<err<<endl;
    }
    
    for (int w=0; w<nBiases; w++) {
        *(biases+w) += eps;
        predict(inputs[0], res, series[0]);
        for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
        const Real out1 = - *(series[lastn-1]->outvals+iOutputs+ierr);
        
        *(biases+w) -= 2*eps;
        predict(inputs[0], res, series[0]);
        for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
        const Real out2 = - *(series[lastn-1]->outvals+iOutputs+ierr);
        
        *(biases+w) += eps;
        *(g->_B+w) += (out1-out2)/(2*eps);
        
        //const Real scale = fabs(*(biases+w));
        const Real scale = max(fabs(*(G->_B+w)),fabs(*(g->_B+w)));
        const Real err = (*(G->_B+w)-*(g->_B+w))/scale;
        if (fabs(err)>1e-4) cout <<"B"<<w<<" "<<*(G->_B+w)<<" "<<*(g->_B+w)<<" "<<err<<endl;
    }
    printf("\n\n\n");
    abort();
}
