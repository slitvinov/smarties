/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "../Settings.h"
#include "Activations.h"
#include <iostream>
#include <cstring>

#define _allocateClean(name, size) { const int sizeSIMD=ceil(size/4.)*4.*sizeof(Real); posix_memalign((void **)& name, 32, sizeSIMD); memset(name, 0, sizeSIMD); }
#define _allocateQuick(name, size) { const int sizeSIMD=ceil(size/4.)*4.*sizeof(Real); posix_memalign((void **)& name, 32, sizeSIMD); }
#define _myfree( name ) free( name );

using namespace std;

struct Link
{
    bool LSTM;
    /*
     a link here is defined as link layer to layer:
     index iI along the network activation outvals representing the index of the first neuron of input layer
     the number nI of neurons of the input layer
     the index iO of the first neuron of the output layer
     the number of neurons in the output layer nO
     the index of the first weight iW along the weight vector
     the weights are all to all: so this link occupies space iW to (iW + nI*nO) along weight vector
     */
    int nI, iI, nO, iO, iW;
    
    /*
     if link is TO lstm, then the rules change a bit
     each LSTM block contains 4 neurons, one is the proper cell and then there are the 3 gates
     if a input signal is connected to one of the four, is also connected to the others 
     thus we just need the index of the first weight for the 3 gates (could have skipped this, iWi = iW + nO*nI and so forth)
     additionally the LSTM contains a memory, contained in Activation->ostate
     memory and gates are treated differently than normal neurons, therefore are contained in separate array, and i keep track of the position with iC
     */
    int iC, iWI, iWF, iWO;
    
    Link(int nI, int iI, int nO, int iO, int iW) : LSTM(false), nI(nI), iI(iI), nO(nO), iO(iO), iW(iW), iC(-1), iWI(-1), iWF(-1), iWO(-1)
    { }
    
    Link() : LSTM(false), nI(0), iI(0), nO(0), iO(0), iW(0), iC(0), iWI(0), iWF(0), iWO(0)
    { }
    
    Link(int nI, int iI, int nO, int iO, int iC, int iW, int iWI, int iWF, int iWO) : LSTM(true), nI(nI), iI(iI), nO(nO), iO(iO), iW(iW), iC(iC), iWI(iWI), iWF(iWF), iWO(iWO)
    { }
    
    void set(int _nI, int _iI, int _nO, int _iO, int _iW)
    {
        this->LSTM = false; this->nI = _nI; this->iI = _iI; this->nO = _nO; this->iO = _iO; this->iW = _iW; this->iC = 0; this->iWI = 0; this->iWF = 0; this->iWO = 0;
        print();
    }
    
    void set(int _nI, int _iI, int _nO, int _iO, int _iC, int _iW, int _iWI, int _iWF, int _iWO)
    {
        this->LSTM = true; this->nI = _nI; this->iI = _iI; this->nO = _nO; this->iO = _iO; this->iW = _iW; this->iC = _iC; this->iWI = _iWI; this->iWF = _iWF; this->iWO = _iWO;
        print();
    }
    
    void print() const
    {
        cout << nI << " " << iI << " " << nO << " " << iO << " " << iW << " " << iC << " " << iWI << " " << iWF << " " << iWO << " " << endl;
        fflush(0);
    }
};

struct Graph //misleading, this is just the graph for a single layer
{
    bool first, last;
    int recurrSize, normalSize, recurrSize_SIMD, normalSize_SIMD, recurrPos, normalPos;
    
    //Link *rl_inputs, *rl_outputs, *nl_inputs, , *nl_outputs;
    Link *rl_recurrent, *nl_recurrent;
    vector<Link*> *rl_inputs_vec, *rl_outputs_vec, *nl_inputs_vec, *nl_outputs_vec;
    
    int wPeep, indState;
    int biasHL, biasIN, biasIG, biasFG, biasOG;
    Graph() : first(false), last(false), recurrSize(0), normalSize(0), recurrSize_SIMD(0), normalSize_SIMD(0), recurrPos(0),  normalPos(0), wPeep(0), indState(0), biasHL(0), biasIN(0), biasIG(0), biasFG(0), biasOG(0)
    {
        //rl_inputs = new Link();
        rl_recurrent = new Link();
        //rl_outputs = new Link();
        //nl_inputs = new Link();
        nl_recurrent = new Link();
        //nl_outputs = new Link();
        rl_inputs_vec = new vector<Link*>(); rl_outputs_vec = new vector<Link*>(); nl_inputs_vec = new vector<Link*>(); nl_outputs_vec = new vector<Link*>();
    }
    
    ~Graph()
    {
        //delete rl_inputs;
        _dispose_object( rl_recurrent);
        //delete rl_outputs;
        //delete nl_inputs;
        _dispose_object( nl_recurrent);
        //delete nl_outputs;
        for (auto & link : *rl_inputs_vec) _dispose_object( link);
        for (auto & link : *rl_outputs_vec) _dispose_object ( link);
        for (auto & link : *nl_inputs_vec) _dispose_object ( link);
        for (auto & link : *nl_outputs_vec) _dispose_object( link);
        _dispose_object( rl_inputs_vec);
        _dispose_object( rl_outputs_vec);
        _dispose_object( nl_inputs_vec);
        _dispose_object( nl_outputs_vec);
    }
};

struct Activation //All the network signals
{
    Activation(int _nNeurons, int _nStates): nNeurons(_nNeurons), nStates(_nStates)
    {
        //contains all inputs to each neuron (inputs to network input layer is empty)
        _allocateQuick(in_vals, nNeurons)
        //contains all neuron outputs that will be the incoming signal to linked layers (outputs of input layer is network inputs)
        _allocateQuick(outvals, nNeurons)
        //deltas for each neuron
        _allocateQuick(errvals, nNeurons)
        //memory of LSTM
        _allocateQuick(ostates, nStates)
        //inputs to gates (cell into in_vals)
        _allocateQuick(iIGates, nStates)
        _allocateQuick(iFGates, nStates)
        _allocateQuick(iOGates, nStates)
        //output of gates and LSTM cell
        _allocateQuick(oMCell, nStates)
        _allocateQuick(oIGates, nStates)
        _allocateQuick(oFGates, nStates)
        _allocateQuick(oOGates, nStates)
        //errors of gates and LSTM cell
        _allocateQuick(eMCell, nStates)
        _allocateQuick(eIGates, nStates)
        _allocateQuick(eFGates, nStates)
        _allocateQuick(eOGates, nStates)
    }
    
    ~Activation()
    {
        _myfree(in_vals)
        _myfree(outvals)
        _myfree(errvals)
        _myfree(ostates)
        
        _myfree(iIGates)
        _myfree(iFGates)
        _myfree(iOGates)
        
        _myfree(oMCell)
        _myfree(oIGates)
        _myfree(oFGates)
        _myfree(oOGates)
        
        _myfree(eMCell)
        _myfree(eIGates)
        _myfree(eFGates)
        _myfree(eOGates)
        
    }
    
    void clearOutput()
    {
        for (int j=0; j<nNeurons; j++)
        *(outvals +j) = 0.;
        
        for (int j=0; j<nStates; j++)
        *(ostates +j) = 0.;
    }
    
    void clearErrors()
    {
        for (int j=0; j<nNeurons; j++)
            *(errvals +j) = 0.;
        
        for (int j=0; j<nStates; j++) {
            *(eOGates +j) = 0.;
            *(eIGates +j) = 0.;
            *(eFGates +j) = 0.;
            *(eMCell  +j) = 0.;
        }
    }

    void clearInputs()
    {
        for (int j=0; j<nNeurons; j++)
            *(in_vals +j) = 0.;

        for (int j=0; j<nStates; j++) {
            *(iIGates +j) = 0.;
            *(iFGates +j) = 0.;
            *(iOGates +j) = 0.;
        }
    }
    
    const int nNeurons, nStates;
    Real *in_vals, *outvals, *errvals, *ostates, *iIGates, *iFGates, *iOGates, *oMCell, *oIGates, *oFGates, *oOGates, *eMCell, *eIGates, *eFGates, *eOGates;
};

struct Grads
{
    Grads(int _nWeights, int _nBiases): nWeights(_nWeights), nBiases(_nBiases)
    {
        _allocateClean(_W, nWeights)
        _allocateClean(_B, nBiases)
    }
    
    ~Grads()
    {
        _myfree(_W)
        _myfree(_B)
    }
    
    const int nWeights, nBiases;
    Real *_W, *_B;
};

struct Mem //Memory light recipient for prediction on agents
{
    Mem(int _nNeurons, int _nStates): nNeurons(_nNeurons), nStates(_nStates)
    {
        _allocateClean(outvals, nNeurons)
        _allocateClean(ostates, nStates)
    }
    
    ~Mem()
    {
        _myfree(outvals);
        _myfree(ostates);
    }
    const int nNeurons, nStates;
    Real *outvals, *ostates;
};

class NormalLayer
{
public:
    const bool last;
    //n neurons and position along Activation->in_vals and outvals
    const int nNeurons, n1stNeuron, n1stBias;
    //function outvals = func(in_vals)
    const Response * func;
    //const Link *input_links, *output_links;
    const Link *recurrent_links;
    const vector<Link*> *input_links, *output_links;
    
    NormalLayer(int nNeurons, int n1stNeuron, int n1stBias,
                //const Link* const nl_il, const Link* const nl_rl, const Link* const nl_ol,
                const vector<Link*>* const nl_il, const Link* const nl_rl, const vector<Link*>* const nl_ol,
                const Response* f, bool last) :
    last(last), nNeurons(nNeurons), n1stNeuron(n1stNeuron), n1stBias(n1stBias), func(f),
    input_links(nl_il), recurrent_links(nl_rl), output_links(nl_ol)
    {
        //printf("nNeurons= %d, n1stNeuron= %d, n1stBias= %d\n",nNeurons, n1stNeuron, n1stBias);
    }
    
    ~NormalLayer()
    {
        _dispose_object(func);
        //links deleted by network
    }
    
    virtual void propagate(Activation* const N, const Real* const weights, const Real* const biases) const;
    virtual void propagate(const Activation* const M, Activation* const N, const Real* const weights, const Real* const biases) const;
    
    virtual void backPropagateDeltaFirst(Activation* const C, const Activation* const N, const Real* const weights, const Real* const biases) const;
    virtual void backPropagateDelta(Activation* const C, const Real* const weights, const Real* const biases) const;
    
    virtual void backPropagateDelta(const Activation* const P, Activation* const C, const Activation* const N, const Real* const weights, const Real* const biases) const
    {   backPropagateDeltaFirst(C, N, weights, biases); }
    virtual void backPropagateDeltaLast(const Activation* const P, Activation* const C, const Real* const weights, const Real* const biases) const
    {   backPropagateDelta(C, weights, biases); }
    
    virtual void backPropagateGrads(const Activation* const C, Grads* const grad) const;
    virtual void backPropagateGrads(const Activation* const P, const Activation* const C, Grads* const grad) const;
    virtual void backPropagateAddGrads(const Activation* const C, Grads* const grad) const;
    virtual void backPropagateAddGrads(const Activation* const P, const Activation* const C, Grads* const grad) const;
    
    Real propagateErrors(const Link* const l, const Activation* const lab, const int iNeuron, const Real* const weights) const;
};

class LSTMLayer: public NormalLayer
{
public:
    const int n1stCell, n1stPeep, n1stBiasIG, n1stBiasFG, n1stBiasOG;
    const Response *ifun, *sigm;
    
    LSTMLayer(int nNeurons, int n1stNeuron, int indState, int n1stPeep,
              int n1stBias, int n1stBiasIG, int n1stBiasFG, int n1stBiasOG,
              //const Link* const rl_il, const Link* const rl_rl, const Link* const rl_ol,
              const vector<Link*>* const rl_il, const Link* const rl_rl, const vector<Link*>* const rl_ol,
              const Response* fI, const Response* fG, const Response* fO, bool last) :
    NormalLayer(nNeurons, n1stNeuron, n1stBias, rl_il, rl_rl, rl_ol, fO, last),
    n1stCell(indState), n1stPeep(n1stPeep), n1stBiasIG(n1stBiasIG),
    n1stBiasFG(n1stBiasFG), n1stBiasOG(n1stBiasOG), ifun(fI), sigm(fG)
    {
        printf("n1stCell= %d, n1stPeep= %d, n1stBiasIG= %d, n1stBiasFG= %d, n1stBiasOG= %d\n", n1stCell, n1stPeep, n1stBiasIG, n1stBiasFG, n1stBiasOG);
    }
    
    ~LSTMLayer()
    {
        _dispose_object(ifun);
        _dispose_object(sigm);
        //links deleted by network
    }
    
    void propagate(Activation* const N, const Real* const weights, const Real* const biases) const override;
    void propagate(const Activation* const M, Activation* const N, const Real* const weights, const Real* const biases) const override;
    
    void backPropagateDeltaFirst(Activation* const C, const Activation* const N, const Real* const weights, const Real* const biases) const override;
    void backPropagateDelta(Activation* const C, const Real* const weights, const Real* const biases) const override;
    
    void backPropagateDelta(const Activation* const P, Activation* const C, const Activation* const N, const Real* const weights, const Real* const biases) const override;
    void backPropagateDeltaLast(const Activation* const P, Activation* const C, const Real* const weights, const Real* const biases) const override;
    
    void backPropagateGrads(const Activation* const C, Grads* const grad) const override;
    void backPropagateGrads(const Activation* const P, const Activation* const C, Grads* const grad) const override;
    void backPropagateAddGrads(const Activation* const C, Grads* const grad) const override;
    void backPropagateAddGrads(const Activation* const P, const Activation* const C, Grads* const grad) const override;
};
