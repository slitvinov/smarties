/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <random>
#include "Approximator.h"
#include "../Settings.h"

#define KER1  
#define KER2
//#define _myallocate( name, size ) name = (Real*) _mm_malloc(ceil(size/SIMD)*SIMD*sizeof(Real), ALLOC);
//#define _myfree( name ) _mm_free( name );
#define _myallocate( name, size ) posix_memalign((void **) & name, ALLOC, ceil(size/(Real)SIMD)*SIMD*sizeof(Real) );
#define _myfree( name ) free( name );
#if SIMD != 1
//#define SIMDKERNELS
//#define SIMDKERNELSIN
//#define SIMDKERNELSG
#endif
using namespace std;

struct Link
{
    const bool LSTM, first, last;
    const int nI, iI, nO, iO, iW, iC, iWI, iWF, iWO, idSdW;
    
    Link(int nI, int iI, int nO, int iO, int iW, bool first=false, bool last=false) : LSTM(false), first(first), last(last), nI(nI), iI(iI), nO(nO), iO(iO), iW(iW), iC(-1), iWI(-1), iWF(-1), iWO(-1), idSdW(-1)
    {
        printf("nI %d, iI %d, nO %d, iO %d, iW %d, iC %d, iWI %d, iWF %d, iWO %d, idSdW %d\n", nI, iI, nO, iO, iW, iC, iWI, iWF, iWO, idSdW);
    }
    
    Link(int nI, int iI, int nO, int iO, int iC, int iW, int iWI, int iWF, int iWO, int idSdW, bool first=false, bool last=false) : LSTM(true), first(first), last(last), nI(nI), iI(iI), nO(nO), iO(iO), iW(iW), iC(iC), iWI(iWI), iWF(iWF), iWO(iWO), idSdW(idSdW)
    {
        printf("nI %d, iI %d, nO %d, iO %d, iW %d, iC %d, iWI %d, iWF %d, iWO %d, idSdW %d\n", nI, iI, nO, iO, iW, iC, iWI, iWF, iWO, idSdW);
    }
};

struct Graph //misleading, this is just the graph for a single layer
{
    bool first;
    int recurrSize, normalSize;
    int recurrPos,  normalPos;
    
    // links INTO layer FROM curr and FROM past layer
    vector<Link*> *rl_c_l, *rl_o_l, *nl_c_l, *nl_o_l;
    // links FROM layer INTO curr and INTO future layer
    vector<Link*> *rl_l_c, *rl_l_f, *nl_l_c, *nl_l_f;

    int wPeep, dSdB, indState;
    int biasHL, biasIN, biasIG, biasFG, biasOG;
    Graph() : first(false), recurrSize(0), normalSize(0), recurrPos(0),  normalPos(0), wPeep(0), dSdB(0), indState(0), biasHL(0), biasIN(0), biasIG(0), biasFG(0), biasOG(0)
    {
        rl_c_l = new vector<Link*>;
        rl_c_l->reserve(5);
        rl_o_l = new vector<Link*>;
        rl_o_l->reserve(5);
        
        nl_c_l = new vector<Link*>;
        nl_c_l->reserve(5);
        nl_o_l = new vector<Link*>;
        nl_o_l->reserve(5);
        
        rl_l_c = new vector<Link*>;
        rl_l_c->reserve(5);
        rl_l_f = new vector<Link*>;
        rl_l_f->reserve(5);
        
        nl_l_c = new vector<Link*>;
        nl_l_c->reserve(5);
        nl_l_f = new vector<Link*>;
        nl_l_f->reserve(5);
    }
};

struct Lab //All the network signals
{
    Lab(int _nNeurons, int _nStates): nNeurons(_nNeurons), nStates(_nStates)
    {
        _myallocate(in_vals, nNeurons)
        _myallocate(outvals, nNeurons)
        _myallocate(errvals, nNeurons)
        
        _myallocate(ostates, nStates)
        _myallocate(iIGates, nStates)
        _myallocate(iFGates, nStates)
        _myallocate(iOGates, nStates)
        
        _myallocate(oMCell, nStates)
        _myallocate(oIGates, nStates)
        _myallocate(oFGates, nStates)
        _myallocate(oOGates, nStates)
        
        _myallocate(eMCell, nStates)
        _myallocate(eIGates, nStates)
        _myallocate(eFGates, nStates)
        _myallocate(eOGates, nStates)
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
        _myallocate(_W, nWeights)
        _myallocate(_B, nBiases)
    }
    
    ~Grads()
    {
        _myfree(_W)
        _myfree(_B)
    }
    
    const int  nWeights, nBiases;
    Real *_W, *_B;
};

struct Dsdw //Essential stuff needed for LSTM gradients
{
    Dsdw(int _ndScelldW, int _ndScelldB): ndScelldB(_ndScelldB), ndScelldW(_ndScelldW)
    {
        _myallocate(IN, ndScelldW)
        _myallocate(IG, ndScelldW)
        _myallocate(FG, ndScelldW)
        _myallocate(DB, ndScelldB)
    }
    
    ~Dsdw()
    {
        _myfree(IN)
        _myfree(IG)
        _myfree(FG)
        _myfree(DB)
    }
    
    const int  ndScelldB, ndScelldW;
    Real *IN, *IG, *FG, *DB;
};

struct Mem //Memory light recipient for prediction on agents
{
    Mem(int _nNeurons, int _nStates): nNeurons(_nNeurons), nStates(_nStates)
    {
        _myallocate(outvals, nNeurons)
        _myallocate(ostates, nStates)
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
    const vector<Link*> * curr_input_links, * curr_output_links, * next_output_links;
    ActivationFunction * func;
    const vector<Link*> * prev_input_links;
    
    NormalLayer(int nNeurons, int n1stNeuron, int n1stBias, vector<Link*> * nl_c_l, vector<Link*> * nl_l_c, vector<Link*> * nl_l_f, ActivationFunction* f, bool last) : last(last), nNeurons(nNeurons), n1stNeuron(n1stNeuron), n1stBias(n1stBias),  curr_input_links(nl_c_l), curr_output_links(nl_l_c), next_output_links(nl_l_f), func(f)
    {
        printf("nNeurons= %d, n1stNeuron= %d, n1stBias= %d\n",nNeurons, n1stNeuron, n1stBias);
    }
    // normal online forwrd / bckwrd prop
    virtual void propagate(Mem * M, Lab * N, Real* weights, Real* biases);
    virtual void backPropagate(Mem * M, Lab * N, Dsdw * dsdw, Grads * grad, Real* weights, Real* biases);
    virtual void backPropagate(Lab * M, Lab * N, Dsdw * dsdw, Grads * grad, Real* weights, Real* biases);
    virtual void backPropagate(Lab * M, Lab * N, Grads * grad, Real* weights, Real* biases);
    // this is just to learn on time series: :(
    virtual void propagate(Lab * prev, Lab * curr, Real* weights, Real* biases);
    //virtual void backPropagateGrads(Lab * prev, Lab * curr, Dsdw * dsdw, Grads * grad);
    virtual void backPropagateDelta(Lab * prev, Lab * curr, Lab * next, Real* weights, Real* biases);
    virtual void backPropagateDelta(Lab * prev, Lab * curr, Real* weights, Real* biases);
    virtual void backPropagateGrads(Lab * prev, Lab * curr, Grads * grad);
    
    // "kernels"
    virtual KER1 void updateInputs(const int n, Lab * N, Real * outvals, Real* weights);
    virtual KER1 void updateOutputs(const int n, Lab * N, Real * oldstates, Real* weights, Real* biases);
    virtual KER1 void updateGrads(const int n, Lab * N, Real * oldvals, Real * oldstates, Dsdw * dsdw, Grads * grad);
    // actual kernels
    KER2 void addInputs(Real* tI, const int n, Link *l, Real * outvals, Real* weights);
    virtual KER2 void updateGrads(const int n, Link *l, Lab * N, Real * outvals, Real* gradW, Dsdw * dsdw);
#ifdef SIMDKERNELSIN
    KER2 void addInputsSIMD(vec & IN, const int n, Link *l, Real * outvals, Real* weights);
#endif
#ifdef SIMDKERNELSG
    virtual KER2 void updateGradsSIMD(const int n, Link *l, Lab * N, Real * outvals, Real* gradW, Dsdw * dsdw);
#endif
    //shared among "all" layer types
    KER2 void addErrors(const int n, Link *l, Lab * N, Real * errvals, Real* weights);
};

class LSTMLayer: public NormalLayer
{
public:
    const int n1stCell, n1stPeep, n1stBiasIG, n1stBiasFG, n1stBiasOG, n1stdSdB;
    ActivationFunction *sigm, *ifun, *ofun;
    
    LSTMLayer(int nNeurons, int n1stNeuron, int indState, int n1stPeep, int n1stBias, int n1stBiasIG, int n1stBiasFG, int n1stBiasOG, int n1stdSdB, vector<Link*> * rl_c_l, vector<Link*> * rl_o_l, vector<Link*> * rl_l_c, vector<Link*> * rl_l_f, ActivationFunction* f, bool last) : NormalLayer(nNeurons, n1stNeuron, n1stBias, rl_c_l, rl_l_c, rl_l_f, f, last), n1stCell(indState), n1stPeep(n1stPeep), n1stBiasIG(n1stBiasIG), n1stBiasFG(n1stBiasFG), n1stBiasOG(n1stBiasOG), n1stdSdB(n1stdSdB),
    sigm(new SoftSigm), ifun(new SoftSign2), ofun(new HardSign(2.))//, prev_input_links(rl_o_l)
    //sigm(new SoftSigm), ifun(SoftSign), ofun(new Linear) //, prev_input_links(rl_o_l)
    {
        prev_input_links = rl_o_l;
        printf("n1stCell= %d, n1stPeep= %d, n1stBiasIG= %d, n1stBiasFG= %d, n1stBiasOG= %d, n1stdSdB= %d\n",n1stCell, n1stPeep, n1stBiasIG, n1stBiasFG, n1stBiasOG, n1stdSdB);
    }
    
    // normal online forwrd / bckwrd prop
    void propagate(Mem * M, Lab * N, Real* weights, Real* biases) override;
    void backPropagate(Mem * M, Lab * N, Dsdw * dsdw, Grads * grad, Real* weights, Real* biases) override;
    void backPropagate(Lab * M, Lab * N, Dsdw * dsdw, Grads * grad, Real* weights, Real* biases) override;
    void backPropagate(Lab * M, Lab * N, Grads * grad, Real* weights, Real* biases) override;
    
    // this is just to learn on time series: :(
    void propagate(Lab * prev, Lab * curr, Real* weights, Real* biases) override;
    //void backPropagateGrads(Lab * prev, Lab * curr, Dsdw * dsdw, Grads * grad) override;
    void backPropagateDelta(Lab * prev, Lab * curr, Lab * next, Real* weights, Real* biases) override;
    void backPropagateDelta(Lab * prev, Lab * curr, Real* weights, Real* biases) override;
    void backPropagateGrads(Lab * prev, Lab * curr, Grads * grad) override;
    
    // "kernels"
    KER1 void updateInputs(const int n, Lab * N, Real * outvals, Real* weights) override;
    KER1 void updateOutputs(const int n, Lab * N, Real * oldstates, Real* weights, Real* biases) override;
    KER1 void updateGrads(const int n, Lab * N, Real * oldvals, Real * oldstates, Dsdw * dsdw, Grads * grad) override;
    KER1 void updateGradsLight(const int n, Lab * N, Real * oldvals, Real * oldstates, Grads * grad);
    
    // "kernels"
    KER2 void addInputs(Real* tC, Real* tI, Real* tF, Real* tO, const int n, Link *l, Real * outvals, Real* weights);
    KER2 void updateGrads(const int n, Link *l, Lab * N, Real * outvals, Real* gradW, Dsdw * dsdw) override;
    KER2 void updateGradsLight( const int n, Link *l, Lab * N, Real * outvals, Real* gradW);
#ifdef SIMDKERNELSIN
    KER2 void addInputsSIMD(vec & IN, vec & IG, vec & FG, vec & OG, const int n, Link *l, Real * outvals, Real* weights);
#endif
#ifdef SIMDKERNELSG
    KER2 void updateGradsSIMD(const int n, Link *l, Lab * N, Real * outvals, Real* gradW, Dsdw * dsdw) override;
    KER2 void updateGradsSIMDLight(const int n, Link *l, Lab * N, Real * outvals, Real* gradW);
#endif
};