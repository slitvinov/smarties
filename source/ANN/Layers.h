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

template<typename Func>
class NormalLayer
{
protected:
    const Link* const recurrent_link;
    const vector<Link*>* const input_links;
    const bool last;
    const int nNeurons, n1stNeuron, n1stBias;
public:
    NormalLayer(int nNeurons, int n1stNeuron, int n1stBias, const vector<Link*>* const nl_il, const Link* const nl_rl, bool last) :
    last(last), nNeurons(nNeurons), n1stNeuron(n1stNeuron), n1stBias(n1stBias), input_links(nl_il), recurrent_link(nl_rl) {
	printf("nNeurons= %d, n1stNeuron= %d, n1stBias= %d\n",nNeurons, n1stNeuron, n1stBias);
    }
    
    virtual void propagate(const Activation* const prev, Activation* const curr, const Real* const weights, const Real* const biases) const;
    virtual void backPropagate(const Activation* const prev, const Activation* const curr, const Activation* const next, Grads* const grad, const Real* const weights) const;

    void propagate(Activation* const curr, const Real* const weights, const Real* const biases) const;
    void backPropagate(const Activation* const curr, Grads* const grad, const Real* const weights) const;
};

template<typename Func>
class Conv2DLayer
{
    const int outputWidth, outputHeight, outputDepth;
public:
    Conv2DLayer(int nNeurons, int n1stNeuron, int n1stBias, const vector<Link*>* const nl_il, bool last) :
	NormalLayer(nNeurons,n1stNeuron,n1stBias,nl_il,nullptr,last) {
	printf("outputWidth=%d, outputHeight=%d, outputDepth=%d\n",outputWidth, outputHeight, outputDepth);
    }

    void propagate(const Activation* const prev, Activation* const curr, const Real* const weights, const Real* const biases) const  override;
	void backPropagate(const Activation* const prev, const Activation* const curr, const Activation* const next, Grads* const grad, const Real* const weights) const  override;
};

template<typename Func, typename Sigm, typename Cell>
class LSTMLayer: public NormalLayer
{
    const int n1stCell, n1stBiasIG, n1stBiasFG, n1stBiasOG;
public:
    LSTMLayer(int nNeurons, int n1stNeuron, int indState,
              int n1stBias, int n1stBiasIG, int n1stBiasFG, int n1stBiasOG,
              const vector<Link*>* const rl_il, const Link* const rl_rl, bool last) :
    NormalLayer(nNeurons, n1stNeuron, n1stBias, rl_il, rl_rl, last),
    n1stCell(indState), n1stBiasIG(n1stBiasIG), n1stBiasFG(n1stBiasFG), n1stBiasOG(n1stBiasOG) {
	printf("n1stCell=%d, n1stBiasIG=%d, n1stBiasFG=%d, n1stBiasOG=%d\n", n1stCell, n1stBiasIG, n1stBiasFG, n1stBiasOG);
    }
    
    void propagate(const Activation* const prev, Activation* const curr, const Real* const weights, const Real* const biases) const  override;
	void backPropagate(const Activation* const prev, const Activation* const curr, const Activation* const next, Grads* const grad, const Real* const weights) const  override;
};

template<typename Func>
class WhiteningLayer: public NormalLayer
{
public:
	WhiteningLayer(int nNeurons, int n1stNeuron, const vector<Link*>* const nl_il) :
	NormalLayer(nNeurons,n1stNeuron,-1,nl_il,false)
    { }

    void propagate(const Activation* const prev, Activation* const curr, const Real* const weights, const Real* const biases) const  override;
	void backPropagate(const Activation* const prev, const Activation* const curr, const Activation* const next, Grads* const grad, const Real* const weights) const  override;
};
