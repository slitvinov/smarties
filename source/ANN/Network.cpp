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
    uniform_real_distribution<Real> dis(-sqrt(12.),sqrt(12.));
    
    for (const auto & l : *(g.nl_inputs_vec))
    {
        for (int w=l->iW ; w<(l->iW + l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen) / Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iW,_weights);
    }
    
    {
        const Link* const l = g.nl_recurrent;
        for (int w=l->iW ; w<(l->iW + l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen) / Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iW,_weights);
    }
    
    for (const auto & l : *(g.rl_inputs_vec))
    {
        for (int w=l->iW ; w<(l->iW + l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen) / Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iW,_weights);
        
        for (int w=l->iWI; w<(l->iWI+ l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen) / Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWI,_weights);
        
        for (int w=l->iWF; w<(l->iWF+ l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen) / Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWF,_weights);
        
        for (int w=l->iWO; w<(l->iWO+ l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen) / Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWO,_weights);
    }
    
    {
        const Link* const l = g.rl_recurrent;
        for (int w=l->iW ; w<(l->iW + l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen) / Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iW,_weights);
        
        for (int w=l->iWI; w<(l->iWI+ l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen) / Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWI,_weights);
        
        for (int w=l->iWF; w<(l->iWF+ l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen) / Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWF,_weights);
        
        for (int w=l->iWO; w<(l->iWO+ l->nO*l->nI); w++)
            *(_weights +w) = dis(*gen) / Real(l->nO + l->nI);
        orthogonalize(l->nO,l->nI,l->iWO,_weights);
    }
    
    //if (not g.last)
    for (int w=g.biasHL; w<g.biasHL+g.normalSize; w++)
        *(_biases +w) = dis(*gen) / Real(g.normalSize);
    
    //if (not g.last)
    for (int w=g.biasIN; w<g.biasIN+g.recurrSize; w++)
        *(_biases +w) = dis(*gen) / Real(g.recurrSize);
        
    for (int w=g.biasIG; w<g.biasIG+g.recurrSize; w++)
        *(_biases +w) = dis(*gen) / Real(g.recurrSize) - .5;
        
    for (int w=g.biasFG; w<g.biasFG+g.recurrSize; w++)
        *(_biases +w) = dis(*gen) / Real(g.recurrSize) + .5;
    
    for (int w=g.biasOG; w<g.biasOG+g.recurrSize; w++)
        *(_biases +w) = dis(*gen) / Real(g.recurrSize) - .5;
}

void Network::addNormal(Graph* const p, Graph* const g, const bool first, const bool last)
{
    if (g->normalSize>0) {
        g->last = last;
        g->normalPos = nNeurons;
        nNeurons += g->normalSize;
        g->biasHL = nBiases;
        nBiases += g->normalSize;
        
        if (p->recurrSize>0)
        { //conntected to previous recurrent layer
            //g->nl_inputs->set( p->recurrSize, p->recurrPos, g->normalSize, g->normalPos, nWeights);
            //p->rl_outputs->set(p->recurrSize, p->recurrPos, g->normalSize, g->normalPos, nWeights);
            
            Link* tmp=new Link(p->recurrSize, p->recurrPos, g->normalSize, g->normalPos, nWeights);
            g->nl_inputs_vec->push_back(tmp);
            p->rl_outputs_vec->push_back(tmp);
            
            nWeights += p->recurrSize*g->normalSize;
        }
        else if (p->normalSize>0)
        { //conntected to previous normal layer
            //g->nl_inputs->set( p->normalSize, p->normalPos, g->normalSize, g->normalPos, nWeights);
            //p->nl_outputs->set(p->normalSize, p->normalPos, g->normalSize, g->normalPos, nWeights);
            
            Link* tmp=new Link(p->normalSize, p->normalPos, g->normalSize, g->normalPos, nWeights);
            g->nl_inputs_vec->push_back(tmp);
            p->nl_outputs_vec->push_back(tmp);
            
            nWeights += p->normalSize*g->normalSize;
        }
        else die("Unlinked to inputs");
        
        if (false) { //(!last) //conntected  to past realization of current normal layer
            g->nl_recurrent->set(g->normalSize, g->normalPos, g->normalSize, g->normalPos, nWeights);
            nWeights += g->normalSize*g->normalSize;
        }
        
        #ifndef _scaleR_
        const Response * f = (last) ? new Response : new SoftSign;
        if (last) printf( "Linear output\n");
        #else
        const Response * f = new SoftSign;
        if (last) printf( "Logic output\n");
        #endif

        //NormalLayer * l = new NormalLayer(g->normalSize, g->normalPos, g->biasHL, g->nl_inputs, g->nl_recurrent, g->nl_outputs, f, last);
        NormalLayer * l = new NormalLayer(g->normalSize, g->normalPos, g->biasHL, g->nl_inputs_vec, g->nl_recurrent, g->nl_outputs_vec, f, last);
        layers.push_back(l);
    }
}

void Network::addLSTM(Graph* const p, Graph* const g, const bool first, const bool last)
{
    if (g->recurrSize>0) {
        g->last = last;
        g->recurrPos = nNeurons;
        nNeurons += g->recurrSize;
        g->indState = nStates;
        nStates  += g->recurrSize;
        
        g->biasIN = nBiases;
        g->biasIG = g->biasIN + g->recurrSize;
        g->biasFG = g->biasIG + g->recurrSize;
        g->biasOG = g->biasFG + g->recurrSize;
        nBiases += 4*g->recurrSize;
        
        if (p->recurrSize>0)      //conntected to previous recurrent layer
        {
            int WeightHL = nWeights;
            nWeights += p->recurrSize*g->recurrSize;
            int WeightIG = nWeights;
            nWeights += p->recurrSize*g->recurrSize;
            int WeightFG = nWeights;
            nWeights += p->recurrSize*g->recurrSize;
            int WeightOG = nWeights;
            nWeights += p->recurrSize*g->recurrSize;
            /*
            g->rl_inputs->set (p->recurrSize, p->recurrPos, g->recurrSize, g->recurrPos,
                               g->indState, WeightHL, WeightIG, WeightFG, WeightOG);
            p->rl_outputs->set(p->recurrSize, p->recurrPos, g->recurrSize, g->recurrPos,
                               g->indState, WeightHL, WeightIG, WeightFG, WeightOG);
            */
            Link* tmp=new Link(p->recurrSize, p->recurrPos, g->recurrSize, g->recurrPos, g->indState, WeightHL, WeightIG, WeightFG, WeightOG);
            g->rl_inputs_vec->push_back(tmp);
            p->rl_outputs_vec->push_back(tmp);
        }
        else if (p->normalSize>0) //conntected to previous normal layer
        {
            int WeightHL = nWeights;
            nWeights += p->normalSize*g->recurrSize;
            int WeightIG = nWeights;
            nWeights += p->normalSize*g->recurrSize;
            int WeightFG = nWeights;
            nWeights += p->normalSize*g->recurrSize;
            int WeightOG = nWeights;
            nWeights += p->normalSize*g->recurrSize;
            /*
            g->rl_inputs->set (p->normalSize, p->normalPos, g->recurrSize, g->recurrPos,
                               g->indState, WeightHL, WeightIG, WeightFG, WeightOG);
            p->nl_outputs->set(p->normalSize, p->normalPos, g->recurrSize, g->recurrPos,
                               g->indState, WeightHL, WeightIG, WeightFG, WeightOG);
            */
            Link* tmp=new Link(p->normalSize, p->normalPos, g->recurrSize, g->recurrPos, g->indState, WeightHL, WeightIG, WeightFG, WeightOG);
            g->rl_inputs_vec->push_back(tmp);
            p->nl_outputs_vec->push_back(tmp);
        }
        else die("Unlinked to inputs");

        { //conntected to past realization of current recurrent layer
            int WeightHL = nWeights;
            nWeights += g->recurrSize*g->recurrSize;
            int WeightIG = nWeights;
            nWeights += g->recurrSize*g->recurrSize;
            int WeightFG = nWeights;
            nWeights += g->recurrSize*g->recurrSize;
            int WeightOG = nWeights;
            nWeights += g->recurrSize*g->recurrSize;
            g->rl_recurrent->set(g->recurrSize, g->recurrPos, g->recurrSize, g->recurrPos,
                                 g->indState, WeightHL, WeightIG, WeightFG, WeightOG);
        }
        
        #ifndef _scaleR_
        const Response * fI = (last) ? new Response : new SoftSign2;
        const Response * fG = new SoftSigm;
        const Response * fO = (last) ? new Response : new SoftSign;
        if (last) printf("Linear output\n");
        #else
        const Response * fI = new SoftSign2;
        const Response * fG = new SoftSigm;
        const Response * fO = new SoftSign;
        if (last) printf("Logic output\n");
        #endif
        
        NormalLayer * l = new LSTMLayer(g->recurrSize, g->recurrPos, g->indState, g->wPeep, g->biasIN, g->biasIG, g->biasFG, g->biasOG, g->rl_inputs_vec, g->rl_recurrent, g->rl_outputs_vec, fI, fG, fO, last);
        //NormalLayer * l = new LSTMLayer(g->recurrSize, g->recurrPos, g->indState, g->wPeep, g->biasIN, g->biasIG, g->biasFG, g->biasOG, g->rl_inputs, g->rl_recurrent, g->rl_outputs, fI, fG, fO, last);
        layers.push_back(l);
    }
}

Network::Network(const vector<int>& layerSize, const bool bLSTM, const Settings & settings) : //, bool bSeparateOutputs=false) :
Pdrop(settings.nnPdrop), nInputs(layerSize.front()), nOutputs(layerSize.back()),
nLayers(0), nNeurons(0), nWeights(0), nBiases(0), nStates(0),
allocatedFrozenWeights(false), allocatedDroputWeights(false), backedUp(false),
gen(settings.gen), bDump(not settings.bTrain)
{
    const int nMixedLayers = layerSize.size();
    if(nMixedLayers<3) die("Put at least one hidden layer. \n");
    if(nInputs<1)      die("No inputs. \n");
    iOut.resize(nOutputs);
    {
        Graph * g = new Graph();
        g->first = true;
        g->normalSize = nInputs;
        nNeurons += g->normalSize;
        G.push_back(g);
    }
    
    if(not settings.bSeparateOutputs) //this creates just one output layer with one feed-forward neuron per output
    {
        for (int i=1; i<nMixedLayers; i++)
        { //layer 0 is the input layer
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
        for (int i=0; i<nOutputs; i++) {
            iOut[i] = G.back()->normalPos + i;
            //iOutputs = (bLSTM) ? G.back()->recurrPos : G.back()->normalPos;
        }
    }
    else //create one second-to-last layer per each output, each connected to one feed-forward neuron representing one output
    {
        for (int i=1; i<nMixedLayers-2; i++) //up to third-to-last
        { //layer 0 is the input layer
            Graph * g = new Graph();
            bool first = i==1; bool last = false;
            if (bLSTM) { //
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
        
        const int firstSeparate = nMixedLayers - 2;
        const int lastJointLayer = G.size() - 1;
        const bool first = 1 == firstSeparate;
        
        for (int i=0; i<nOutputs; i++) {
            Graph * g = new Graph();
            if (bLSTM) { //
                g->recurrSize = layerSize[firstSeparate];
                g->normalSize = 0;
                addLSTM(G[lastJointLayer],  g,first,false);
            } else {
                g->normalSize = layerSize[firstSeparate];
                g->recurrSize = 0;
                addNormal(G[lastJointLayer],g,first,false);
            }
            
            G.push_back(g);
            Graph * o = new Graph();
            o->normalSize = 1;
            o->recurrSize = 0;
            addNormal(G.back(),o,false,true);
            iOut[i] = o->normalPos;
            printf("iOut[%d] = %d\n",i,iOut[i]);
            G.push_back(o);
        }
    }
    
    nLayers = layers.size();
    printf("nNeurons= %d, nWeights= %d, nBiases= %d, nStates= %d iOut = %d, nInputs = %d, nOutputs = %d \n",
           nNeurons, nWeights, nBiases, nStates, iOut[0], nInputs, nOutputs);
    
    for (int i=0; i<settings.nAgents; ++i) {
        Mem * m = new Mem(nNeurons, nStates);
        mem.push_back(m);
    }
    dump_ID.resize(settings.nAgents);
    allocateSeries(3);
    
    grad = new Grads(nWeights,nBiases);
    _allocateClean(weights, nWeights)
    _allocateClean(biases,  nBiases)
    
    if (settings.nThreads>1) {
        Vgrad.resize(settings.nThreads);
        for (int i=0; i<settings.nThreads; ++i)
            Vgrad[i] = new Grads(nWeights, nBiases);
    } else {
            _grad    = new Grads(nWeights, nBiases);
    }
    
    for (int i=1; i<static_cast<int>(G.size()); i++) {
        initializeWeights(*G[i], weights, biases);
    }
    updateFrozenWeights();
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
    string nameBackup = fname;
    
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
    return true;
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output, const Activation* const _M, Activation* const _N, const Real* const _weights, const Real* const _biases) const
{
    for (int j=0; j<nInputs; j++)
        *(_N->outvals +j) = _input[j];
    
    for (int j=0; j<nLayers; j++)
        layers[j]->propagate(_M,_N,_weights,_biases);
    
    assert(static_cast<int>(_output.size())==nOutputs);
    
    /*
    int j(0);
    for (int i=iOutputs; i<nNeurons; i++) {
        *(_N->errvals+i) = 0.;
        _output[j] = *(_N->outvals+i);
        j++;
    }
     */
    for (int i=0; i<nOutputs; i++) {
        *(_N->errvals + iOut[i]) = 0.;
        _output[i] = *(_N->outvals + iOut[i]);
    }
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output, Activation* const _N, const Real* const _weights, const Real* const _biases) const
{
    for (int j=0; j<nInputs; j++)
    *(_N->outvals +j) = _input[j];
    
    for (int j=0; j<nLayers; j++)
        layers[j]->propagate(_N,_weights,_biases);
    
    assert(static_cast<int>(_output.size())==nOutputs);
    /*
    int j(0);
    for (int i=iOutputs; i<nNeurons; i++) {
        *(_N->errvals+i) = 0.;
        _output[j] = *(_N->outvals+i);
        j++;
    }
     */
    for (int i=0; i<nOutputs; i++) {
        *(_N->errvals + iOut[i]) = 0.;
        _output[i] = *(_N->outvals + iOut[i]);
    }
}

void Network::setOutputErrors(vector<Real>& _errors, Activation* const _N)
{
    assert(static_cast<int>(_errors.size())==nOutputs);
    for (int i=0; i<nOutputs; i++) {
        *(_N->errvals + iOut[i]) = _errors[i];
    }
}

//No time dependencies
void Network::computeDeltas(Activation* const _series, const Real* const _weights, const Real* const _biases) const
{
    for (int i=1; i<=nLayers; i++)
        layers[nLayers-i]->backPropagateDelta(_series,_weights,_biases);
}

void Network::computeGrads(const Activation* const lab, Grads* const _Grad) const
{
    for (int i=0; i<nLayers; i++)
    layers[i]->backPropagateGrads(lab,_Grad); //grad is zero-equal
}

void Network::computeAddGrads(const Activation* const lab, Grads* const _Grad) const
{
    for (int i=0; i<nLayers; i++)
    layers[i]->backPropagateAddGrads(lab,_Grad);  //grad is add-equal
}

//Back Prop Through Time:
//compute deltas: start from last activation, propagate deltas back to first
void Network::computeDeltasSeries(vector<Activation*>& _series, const int first, const int last, const Real* const _weights, const Real* const _biases) const
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

void Network::computeGradsSeries(const vector<Activation*>& _series, const int k, Grads* const _Grad) const
{
    for (int i=0; i<nLayers; i++)
        layers[i]->backPropagateGrads(series[k-1],series[k],_Grad);
}

void Network::computeAddGradsSeries(const vector<Activation*>& _series, const int first, const int last, Grads* const _Grad) const
{
    for (int i=0; i<nLayers; i++)
        layers[i]->backPropagateAddGrads(series[first],_Grad);
    
    for (int k=first+1; k<=last; k++)
        for (int i=0; i<nLayers; i++)
            layers[i]->backPropagateAddGrads(series[k-1],series[k],_Grad);
}

void Network::updateFrozenWeights()
{
    if (allocatedFrozenWeights==false) {
        _allocateQuick(tgt_weights, nWeights)
        _allocateQuick(tgt_biases,   nBiases)
        allocatedFrozenWeights = true;
    }
    
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int j=0; j<nWeights; j++)
            *(tgt_weights + j) = *(weights + j);
        
        #pragma omp for nowait
        for (int j=0; j<nBiases; j++)
            *(tgt_biases + j) = *(biases + j);
    }
}

void Network::moveFrozenWeights(const Real alpha)
{
    if (allocatedFrozenWeights==false) updateFrozenWeights();
    const Real _alpha = 1. - alpha;

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int j=0; j<nWeights; j++)
            *(tgt_weights + j) = *(tgt_weights + j)*_alpha + *(weights + j)*alpha;

        #pragma omp for nowait
        for (int j=0; j<nBiases; j++)
            *(tgt_biases + j) = *(tgt_biases + j)*_alpha + *(biases + j)*alpha;
    }
}

void Network::expandMemory(Mem * _M, Activation * _N) const
{
    std::swap(_N->outvals,_M->outvals);
    std::swap(_N->ostates,_M->ostates);
}

void Network::allocateSeries(int _k, vector<Activation*> & _series)
{
    for (int j=static_cast<int>(_series.size()); j<=_k; j++) {
        Activation * ns = new Activation(nNeurons,nStates);
        series.push_back(ns);
    }
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

        bernoulli_distribution dis(Pkeep);
        for (int j=0; j<nWeights; j++) //TODO: betterer, simder, paralleler
        {
            bool res = dis(*gen);
            *(weights + j) = (res) ? *(weights_DropoutBackup + j)*fac : 0.;
        }
    }
}

void Network::removeDropoutMask()
{
    if (allocatedDroputWeights && backedUp) {
        swap(weights_DropoutBackup,weights);
        backedUp = false;
    }
}

void Network::checkGrads(const vector<vector<Real>>& inputs, const int lastn)
{
    printf("Checking gradients\n");
    vector<int> errorPlacements(lastn);
    vector<Real> partialResults(lastn);

    int nseries = inputs.size();
    vector<Real> res(nOutputs);
    allocateSeries(nseries+1);
    
    const Real incr = 1e-6;
    
    uniform_real_distribution<Real> dis(0.,1.);
    for (int i=0; i<lastn; i++)
        errorPlacements[i] = nOutputs*dis(*gen);
    
    Grads * g = new Grads(nWeights,nBiases);
    Grads * G = new Grads(nWeights,nBiases);
    
    {
        predict(inputs[0], res, series[0]);
        
        vector<Real> errs(nOutputs,0);
        errs[errorPlacements[0]] = -1.;
        setOutputErrors(errs, series[0]);
    }
    
    for (int k=1; k<lastn; k++)
    {
        predict(inputs[k], res, series[k-1], series[k]);
        
        vector<Real> errs(nOutputs,0);
        errs[errorPlacements[k]] = -1.;
        setOutputErrors(errs, series[k]);
    }

    computeDeltasSeries(series, 0, lastn-1);
    computeAddGradsSeries(series, 0, lastn-1, G);
    
    
    for (int w=0; w<nWeights; w++) {
        //1
        *(weights+w) += incr;
        
        predict(inputs[0], res, series[0]);
        partialResults[0] =- res[errorPlacements[0]];
        
        for (int k=1; k<lastn; k++)
        {
            predict(inputs[k], res, series[k-1], series[k]);
            partialResults[k] =- res[errorPlacements[k]];
        }
        
        //2
        *(weights+w) -= 2*incr;
        
        predict(inputs[0], res, series[0]);
        partialResults[0] += res[errorPlacements[0]];
            
        for (int k=1; k<lastn; k++)
        {
            predict(inputs[k], res, series[k-1], series[k]);
            partialResults[k] += res[errorPlacements[k]];
        }
        
        //0
        *(weights+w) += incr;
        
        Real grad(0);
        for (int k=0; k<lastn; k++)
            grad += partialResults[k];
        *(g->_W+w) = grad/(2.*incr);
        
        //const Real scale = fabs(*(biases+w));
        const Real scale = max(fabs(*(G->_W+w)),fabs(*(g->_W+w)));
        const Real err = (*(G->_W+w)-*(g->_W+w))/scale;
        if (fabs(err)>1e-4) cout <<"W"<<w<<" "<<*(G->_W+w)<<" "<<*(g->_W+w)<<" "<<err<<endl;
    }
    
    for (int w=0; w<nBiases; w++) {
        //1
        *(biases+w) += incr;
        
        predict(inputs[0], res, series[0]);
        partialResults[0] =- res[errorPlacements[0]];
        
        for (int k=1; k<lastn; k++) {
            predict(inputs[k], res, series[k-1], series[k]);
            partialResults[k] =- res[errorPlacements[k]];
        }
        
        //2
        *(biases+w) -= 2*incr;
        
        predict(inputs[0], res, series[0]);
        partialResults[0] += res[errorPlacements[0]];
        
        for (int k=1; k<lastn; k++) {
            predict(inputs[k], res, series[k-1], series[k]);
            partialResults[k] += res[errorPlacements[k]];
        }
        
        //0
        *(biases+w) += incr;
        
        
        Real grad(0);
        for (int k=0; k<lastn; k++)
            grad += partialResults[k];
        *(g->_B+w) = grad/(2.*incr);
        
        //const Real scale = fabs(*(biases+w));
        const Real scale = max(fabs(*(G->_B+w)),fabs(*(g->_B+w)));
        const Real err = (*(G->_B+w)-*(g->_B+w))/scale;
        if (fabs(err)>1e-4) cout <<"B"<<w<<" "<<*(G->_B+w)<<" "<<*(g->_B+w)<<" "<<err<<endl;
    }
    printf("\n"); fflush(0);
}

/*
void Network::checkGrads(const vector<vector<Real>>& inputs, const int lastn)
{
    printf("Checking gradients\n");
    const int ierr = 0;
 
    int nseries = inputs.size();
    vector<Real> res(nOutputs), errs(nOutputs,0);
    allocateSeries(nseries+1);
    
    Grads * g = new Grads(nWeights,nBiases);
    Grads * G = new Grads(nWeights,nBiases);
    const Real eps = 1e-5;
    
    predict(inputs[0], res, series[0]);
    
    for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
    
    errs[ierr] = -1.;
    setOutputErrors(errs, series[lastn-1]);
    computeDeltasSeries(series, 0, lastn-1);
    computeAddGradsSeries(series, 0, lastn-1, G);
    
    for (int w=0; w<nWeights; w++) {
        *(weights+w) += eps;
        predict(inputs[0], res, series[0]);
        for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
        const Real out1 = - *(series[lastn-1]->outvals+iOut[ierr]);
        
        *(weights+w) -= 2*eps;
        predict(inputs[0], res, series[0]);
        for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
        const Real out2 = - *(series[lastn-1]->outvals+iOut[ierr]);
        
        *(weights+w) += eps;
        *(g->_W+w) = (out1-out2)/(2*eps);
        
        //const Real scale = fabs(*(biases+w));
        const Real scale = max(fabs(*(G->_W+w)),fabs(*(g->_W+w)));
        const Real err = (*(G->_W+w)-*(g->_W+w))/scale;
        if (fabs(err)>1e-4) cout <<"W"<<w<<" "<<*(G->_W+w)<<" "<<*(g->_W+w)<<" "<<err<<endl;
    }
    
    for (int w=0; w<nBiases; w++) {
        *(biases+w) += eps;
        predict(inputs[0], res, series[0]);
        for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
        const Real out1 = - *(series[lastn-1]->outvals+iOut[ierr]);
        
        *(biases+w) -= 2*eps;
        predict(inputs[0], res, series[0]);
        for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
        const Real out2 = - *(series[lastn-1]->outvals+iOut[ierr]);
        
        *(biases+w) += eps;
        *(g->_B+w) = (out1-out2)/(2*eps);
        
        //const Real scale = fabs(*(biases+w));
        const Real scale = max(fabs(*(G->_B+w)),fabs(*(g->_B+w)));
        const Real err = (*(G->_B+w)-*(g->_B+w))/scale;
        if (fabs(err)>1e-4) cout <<"B"<<w<<" "<<*(G->_B+w)<<" "<<*(g->_B+w)<<" "<<err<<endl;
    }
    printf("\n"); fflush(0);
 }
 */

void Network::computeDeltasInputs(vector<Real>& grad, const Activation* const _series, const Real* const _weights, const Real* const _biases) const
{//no weight grad to care about, no recurrent links
    assert(static_cast<int>(grad.size())==nInputs);
    for (int n=0; n<nInputs; n++) {
        Real err(0);
        for (const auto & link : *(G[0]->nl_outputs_vec)) {
            //loop over all layers to which this layer is connected to
            err += layers[0]->propagateErrors(link, _series, n, _weights);
            //the propagateErrors method does not have any layer specific info, so it's fine
        }
        grad[n] = err; //no activation function on inputs
    }
}