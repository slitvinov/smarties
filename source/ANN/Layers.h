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
#define KER1  
#define KER2
//#define _myallocate( name, size ) name = (Real*) _mm_malloc(ceil(size/SIMD)*SIMD*sizeof(Real), ALLOC);
//#define _myfree( name ) _mm_free( name );
#define _allocateClean( name, size ) {const int sizeSIMD=ceil(size/(Real)SIMD)*SIMD*sizeof(Real); posix_memalign((void **)& name,ALLOC,sizeSIMD); memset(name,0,sizeSIMD);}
#define _allocateQuick( name, size ) {const int sizeSIMD=ceil(size/(Real)SIMD)*SIMD*sizeof(Real); posix_memalign((void **)& name,ALLOC,sizeSIMD);}
//#define _allocateQuick( name, size ) {const int sizeSIMD=ceil(size/(Real)SIMD)*SIMD*sizeof(Real); posix_memalign((void **)& name,ALLOC,sizeSIMD); memset(name,0,sizeSIMD);}
#define _myfree( name ) free( name );
#if SIMD != 1
//#define SIMDKERNELS
//#define SIMDKERNELSIN
//#define SIMDKERNELSG
#endif
using namespace std;

struct Link
{
    bool LSTM;
    int nI, iI, nO, iO, iW, iC, iWI, iWF, iWO;
    
    Link(int nI, int iI, int nO, int iO, int iW) : LSTM(false), nI(nI), iI(iI), nO(nO), iO(iO), iW(iW), iC(-1), iWI(-1), iWF(-1), iWO(-1)
    {
        
    }
    
    Link() : LSTM(false), nI(0), iI(-1), nO(0), iO(-1), iW(iW), iC(-1), iWI(-1), iWF(-1), iWO(-1)
    {
        
    }
    
    Link(int nI, int iI, int nO, int iO, int iC, int iW, int iWI, int iWF, int iWO) : LSTM(true), nI(nI), iI(iI), nO(nO), iO(iO), iW(iW), iC(iC), iWI(iWI), iWF(iWF), iWO(iWO)
    {
    }
    
    void set(int _nI, int _iI, int _nO, int _iO, int _iW)
    {
        LSTM = false; nI = _nI; iI = _iI; nO = _nO; iO = _iO; iW = _iW; iC = -1; iWI = -1; iWF = -1; iWO = -1;
        printf("nI %d, iI %d, nO %d, iO %d, iW %d, iC %d, iWI %d, iWF %d, iWO %d\n", nI, iI, nO, iO, iW, iC, iWI, iWF, iWO);
    }
    
    
    void set(int _nI, int _iI, int _nO, int _iO, int _iC, int _iW, int _iWI, int _iWF, int _iWO)
    {
        LSTM = true; nI = _nI; iI = _iI; nO = _nO; iO = _iO; iW = _iW; iC = _iC; iWI = _iWI; iWF = _iWF; iWO = _iWO;
        printf("nI %d, iI %d, nO %d, iO %d, iW %d, iC %d, iWI %d, iWF %d, iWO %d\n", nI, iI, nO, iO, iW, iC, iWI, iWF, iWO);
    }
    
    string print() const
    {
        printf("nI %d, iI %d, nO %d, iO %d, iW %d, iC %d, iWI %d, iWF %d, iWO %d\n", nI, iI, nO, iO, iW, iC, iWI, iWF, iWO);
    }
};

struct Graph //misleading, this is just the graph for a single layer
{
    bool first;
    int recurrSize, normalSize, recurrSize_SIMD, normalSize_SIMD, recurrPos, normalPos;
    
    // links INTO layer FROM curr and FROM past layer
    Link *rl_inputs, *rl_recurrent, *rl_outputs, *nl_inputs, *nl_recurrent, *nl_outputs;

    int wPeep, indState;
    int biasHL, biasIN, biasIG, biasFG, biasOG;
    Graph() : first(false), recurrSize(0), normalSize(0), recurrSize_SIMD(0), normalSize_SIMD(0), recurrPos(0),  normalPos(0), wPeep(0), indState(0), biasHL(0), biasIN(0), biasIG(0), biasFG(0), biasOG(0)
    {
        rl_inputs = new Link(); rl_recurrent = new Link(); rl_outputs = new Link();
        nl_inputs = new Link(); nl_recurrent = new Link(); nl_outputs = new Link();
    }
};

struct Lab //All the network signals
{
    Lab(int _nNeurons, int _nStates): nNeurons(_nNeurons), nStates(_nStates)
    {
        _allocateQuick(in_vals, nNeurons)
        _allocateQuick(outvals, nNeurons)
        _allocateClean(errvals, nNeurons)
        
        _allocateQuick(ostates, nStates)
        _allocateQuick(iIGates, nStates)
        _allocateQuick(iFGates, nStates)
        _allocateQuick(iOGates, nStates)
        
        _allocateQuick(oMCell, nStates)
        _allocateQuick(oIGates, nStates)
        _allocateQuick(oFGates, nStates)
        _allocateQuick(oOGates, nStates)
        
        _allocateQuick(eMCell, nStates)
        _allocateQuick(eIGates, nStates)
        _allocateQuick(eFGates, nStates)
        _allocateQuick(eOGates, nStates)
    }
    ~Lab()
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
    const int nNeurons, n1stNeuron, n1stBias;
    const Activation * func;
    const Link *input_links, *recurrent_links, *output_links;
    
    NormalLayer(int nNeurons, int n1stNeuron, int n1stBias, const Link* const nl_il,
                const Link* const nl_rl, const Link* const nl_ol, const Activation* f, bool last) :
    last(last), nNeurons(nNeurons), n1stNeuron(n1stNeuron), n1stBias(n1stBias), func(f),
    input_links(nl_il), recurrent_links(nl_rl), output_links(nl_ol)
    {
        printf("nNeurons= %d, n1stNeuron= %d, n1stBias= %d\n",nNeurons, n1stNeuron, n1stBias);
        cout << input_links <<endl;
        cout << recurrent_links <<endl;
        cout << output_links <<endl;
    }
    
    //virtual void backPropagate(const Lab* const P, Lab* const C, Grads* const grad, const Real* const weights, const Real* const biases) const;
    
    virtual void propagate(Lab* const N, const Real* const weights, const Real* const biases) const;
    virtual void propagate(const Mem* const M, Lab* const N, const Real* const weights, const Real* const biases) const;
    virtual void propagate(const Lab* const M, Lab* const N, const Real* const weights, const Real* const biases) const;
    
    virtual void backPropagateDeltaFirst(Lab* const C, const Lab* const N, const Real* const weights, const Real* const biases) const;
    virtual void backPropagateDelta(Lab* const C, const Real* const weights, const Real* const biases) const;
    
    virtual void backPropagateDelta(const Lab* const P, Lab* const C, const Lab* const N, const Real* const weights, const Real* const biases) const;
    virtual void backPropagateDeltaLast(const Lab* const P, Lab* const C, const Real* const weights, const Real* const biases) const;
    
    virtual void backPropagateGrads(const Lab* const C, Grads* const grad) const;
    virtual void backPropagateGrads(const Lab* const P, const Lab* const C, Grads* const grad) const;
    virtual void backPropagateAddGrads(const Lab* const C, Grads* const grad) const;
    virtual void backPropagateAddGrads(const Lab* const P, const Lab* const C, Grads* const grad) const;
};

class LSTMLayer: public NormalLayer
{
public:
    const int n1stCell, n1stPeep, n1stBiasIG, n1stBiasFG, n1stBiasOG;
    const Activation *ifun, *sigm;
    
    LSTMLayer(int nNeurons, int n1stNeuron, int indState, int n1stPeep,
              int n1stBias, int n1stBiasIG, int n1stBiasFG, int n1stBiasOG,
              const Link* const rl_il, const Link* const rl_rl, const Link* const rl_ol,
              const Activation* fI, const Activation* fG, const Activation* fO, bool last) :
    NormalLayer(nNeurons, n1stNeuron, n1stBias, rl_il, rl_rl, rl_ol, fO, last),
    n1stCell(indState), n1stPeep(n1stPeep), n1stBiasIG(n1stBiasIG),
    n1stBiasFG(n1stBiasFG), n1stBiasOG(n1stBiasOG), ifun(fI), sigm(fG)
    {
        printf("n1stCell= %d, n1stPeep= %d, n1stBiasIG= %d, n1stBiasFG= %d, n1stBiasOG= %d\n", n1stCell, n1stPeep, n1stBiasIG, n1stBiasFG, n1stBiasOG);
    }
    
    //void backPropagate(const Lab* const P, Lab* const C, Grads* const grad, const Real* const weights, const Real* const biases) const override;
    
    void propagate(Lab* const N, const Real* const weights, const Real* const biases) const override;
    void propagate(const Mem* const M, Lab* const N, const Real* const weights, const Real* const biases) const override;
    void propagate(const Lab* const M, Lab* const N, const Real* const weights, const Real* const biases) const override;
    
    void backPropagateDeltaFirst(Lab* const C, const Lab* const N, const Real* const weights, const Real* const biases) const override;
    void backPropagateDelta(Lab* const C, const Real* const weights, const Real* const biases) const override;
    
    void backPropagateDelta(const Lab* const P, Lab* const C, const Lab* const N, const Real* const weights, const Real* const biases) const override;
    void backPropagateDeltaLast(const Lab* const P, Lab* const C, const Real* const weights, const Real* const biases) const override;
    
    void backPropagateGrads(const Lab* const C, Grads* const grad) const override;
    void backPropagateGrads(const Lab* const P, const Lab* const C, Grads* const grad) const override;
    void backPropagateAddGrads(const Lab* const C, Grads* const grad) const override;
    void backPropagateAddGrads(const Lab* const P, const Lab* const C, Grads* const grad) const override;
};
