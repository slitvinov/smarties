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
#include "Links.h"
#include <iostream>

using namespace std;

class NormalLayer
{
public:
    const bool last;
    //n neurons and position along Activation->in_vals and outvals
    const int nNeurons, n1stNeuron, n1stBias;
    //function outvals = func(in_vals)
    const Response * func;
    const Link *recurrent_link;
    const vector<Link*> *input_links, *output_links;
    
    NormalLayer(int nNeurons, int n1stNeuron, int n1stBias,
                const vector<Link*>* const nl_il, const Link* const nl_rl, const vector<Link*>* const nl_ol,
                const Response* f, bool last) :
    last(last), nNeurons(nNeurons), n1stNeuron(n1stNeuron), n1stBias(n1stBias), func(f),
    input_links(nl_il), recurrent_link(nl_rl), output_links(nl_ol)
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
    const int n1stCell, n1stBiasIG, n1stBiasFG, n1stBiasOG;
    const Response *ifun, *sigm;
    
    LSTMLayer(int nNeurons, int n1stNeuron, int indState,
              int n1stBias, int n1stBiasIG, int n1stBiasFG, int n1stBiasOG,
              const vector<Link*>* const rl_il, const Link* const rl_rl, const vector<Link*>* const rl_ol,
              const Response* fI, const Response* fG, const Response* fO, bool last) :
    NormalLayer(nNeurons, n1stNeuron, n1stBias, rl_il, rl_rl, rl_ol, fO, last),
    n1stCell(indState), n1stBiasIG(n1stBiasIG),
    n1stBiasFG(n1stBiasFG), n1stBiasOG(n1stBiasOG), ifun(fI), sigm(fG)
    {
        printf("n1stCell= %d, n1stBiasIG= %d, n1stBiasFG= %d, n1stBiasOG= %d\n", n1stCell, n1stBiasIG, n1stBiasFG, n1stBiasOG);
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
