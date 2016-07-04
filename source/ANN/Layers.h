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
#include <cstring>
#define KER1  
#define KER2
//#define _myallocate( name, size ) name = (Real*) _mm_malloc(ceil(size/SIMD)*SIMD*sizeof(Real), ALLOC);
//#define _myfree( name ) _mm_free( name );
#define _allocateClean( name, size ) {const int sizeSIMD=ceil(size/(Real)SIMD)*SIMD*sizeof(Real); posix_memalign((void **)& name,ALLOC,sizeSIMD); memset(name,0,sizeSIMD);}
#define _allocateQuick( name, size ) {const int sizeSIMD=ceil(size/(Real)SIMD)*SIMD*sizeof(Real); posix_memalign((void **)& name,ALLOC,sizeSIMD);}
#define _myfree( name ) free( name );
#if SIMD != 1
//#define SIMDKERNELS
//#define SIMDKERNELSIN
//#define SIMDKERNELSG
#endif
using namespace std;

struct Link
{
    const bool LSTM;
    const int nI, iI, nO, iO, iW, iC, iWI, iWF, iWO;
    
    Link(int nI, int iI, int nO, int iO, int iW) :
    LSTM(false), nI(nI), iI(iI), nO(nO), iO(iO), iW(iW), iC(-1), iWI(-1), iWF(-1), iWO(-1)
    {
        printf("nI %d, iI %d, nO %d, iO %d, iW %d, iC %d, iWI %d, iWF %d, iWO %d\n", nI, iI, nO, iO, iW, iC, iWI, iWF, iWO);
    }
    
    Link(int nI, int iI, int nO, int iO, int iC, int iW, int iWI, int iWF, int iWO) :
    LSTM(true), nI(nI), iI(iI), nO(nO), iO(iO), iW(iW), iC(iC), iWI(iWI), iWF(iWF), iWO(iWO)
    {
        printf("nI %d, iI %d, nO %d, iO %d, iW %d, iC %d, iWI %d, iWF %d, iWO %d\n", nI, iI, nO, iO, iW, iC, iWI, iWF, iWO);
    }
};

struct Graph //misleading, this is just the graph for a single layer
{
    bool first;
    int recurrSize, normalSize, recurrSize_SIMD, normalSize_SIMD, recurrPos, normalPos;
    
    // links INTO layer FROM curr and FROM past layer
    vector<Link*> *rl_c_l, *rl_o_l, *nl_c_l, *nl_o_l;
    // links FROM layer INTO curr and INTO future layer
    vector<Link*> *rl_l_c, *rl_l_f, *nl_l_c, *nl_l_f;

    int wPeep, indState;
    int biasHL, biasIN, biasIG, biasFG, biasOG;
    Graph() : first(false), recurrSize(0), normalSize(0), recurrSize_SIMD(0), normalSize_SIMD(0), recurrPos(0),  normalPos(0), wPeep(0), indState(0), biasHL(0), biasIN(0), biasIG(0), biasFG(0), biasOG(0)
    {
        rl_c_l = new vector<Link*>;
        rl_c_l->reserve(4);
        rl_o_l = new vector<Link*>;
        rl_o_l->reserve(4);
        
        nl_c_l = new vector<Link*>;
        nl_c_l->reserve(4);
        nl_o_l = new vector<Link*>;
        nl_o_l->reserve(4);
        
        rl_l_c = new vector<Link*>;
        rl_l_c->reserve(4);
        rl_l_f = new vector<Link*>;
        rl_l_f->reserve(4);
        
        nl_l_c = new vector<Link*>;
        nl_l_c->reserve(4);
        nl_l_f = new vector<Link*>;
        nl_l_f->reserve(4);
    }
};

struct Lab //All the network signals
{
    Lab(int _nNeurons, int _nStates): nNeurons(_nNeurons), nStates(_nStates)
    {
        _allocateQuick(in_vals, nNeurons)
        _allocateQuick(outvals, nNeurons)
        _allocateQuick(errvals, nNeurons)
        
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
    const vector<Link*> *curr_input_links, *prev_input_links, *curr_output_links, *next_output_links;
    
    NormalLayer(int nNeurons, int n1stNeuron, int n1stBias,
                const vector<Link*>* const nl_c_l, const vector<Link*>* const nl_o_l,
                const vector<Link*>* const nl_l_c, const vector<Link*>* const nl_l_f,
                const Activation* f, bool last) :
    last(last), nNeurons(nNeurons), n1stNeuron(n1stNeuron), n1stBias(n1stBias), func(f),
    curr_input_links(nl_c_l), prev_input_links(nl_o_l), curr_output_links(nl_l_c), next_output_links(nl_l_f)
    {
        printf("nNeurons= %d, n1stNeuron= %d, n1stBias= %d\n",nNeurons, n1stNeuron, n1stBias);
    }
    
    virtual void backPropagate(const Lab* const P, Lab* const C, Grads* const grad, const Real* const weights, const Real* const biases) const;
    
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
              const vector<Link*>* const rl_c_l, const vector<Link*>* const rl_o_l,
              const vector<Link*>* const rl_l_c, const vector<Link*>* const rl_l_f,
              const Activation* fI, const Activation* fG, const Activation* fO, bool last) :
    NormalLayer(nNeurons, n1stNeuron, n1stBias, rl_c_l, rl_o_l, rl_l_c, rl_l_f, fO, last),
    n1stCell(indState), n1stPeep(n1stPeep), n1stBiasIG(n1stBiasIG),
    n1stBiasFG(n1stBiasFG), n1stBiasOG(n1stBiasOG), ifun(fI), sigm(fG)
    {
        printf("n1stCell= %d, n1stPeep= %d, n1stBiasIG= %d, n1stBiasFG= %d, n1stBiasOG= %d\n",
               n1stCell, n1stPeep, n1stBiasIG, n1stBiasFG, n1stBiasOG);
    }
    
    void backPropagate(const Lab* const P, Lab* const C, Grads* const grad, const Real* const weights, const Real* const biases) const override;
    
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
