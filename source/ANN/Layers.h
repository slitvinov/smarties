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

class Layer
{
    public:
    virtual void propagate(const Activation* const prev, Activation* const curr, 
                           const Real* const weights, const Real* const biases, const Real noise) const = 0;
    virtual void backPropagate( Activation* const prev,  Activation* const curr, const Activation* const next, Grads* const grad, const Real* const weights) const = 0;

    void propagate(Activation* const curr, const Real* const weights, const Real* const biases, const Real noise) const 
        { return propagate(nullptr, curr, weights, biases, noise); }
    void backPropagate( Activation* const curr, Grads* const grad, const Real* const weights) const
        { return backPropagate(nullptr, curr, nullptr, grad, weights); }
};

template<typename outFunc>
class NormalLayer: public Layer
{
    const int nNeurons, n1stNeuron, n1stBias;
    const vector<NormalLink*>* const input_links;
    const NormalLink* const recurrent_link;
    const bool last;
public:
    typedef outFunc Func;
    NormalLayer(int nNeurons, int n1stNeuron, int n1stBias, const vector<NormalLink*>* const nl_il, const NormalLink* const nl_rl, bool last) :
    nNeurons(nNeurons), n1stNeuron(n1stNeuron), n1stBias(n1stBias), input_links(nl_il), recurrent_link(nl_rl), last(last)
    {
	   printf("Normal Layer of size %d, with first ID %d and first bias ID %d\n",nNeurons, n1stNeuron, n1stBias);
    }
    
    void propagate(const Activation* const prev, Activation* const curr, 
                   const Real* const weights, const Real* const biases, const Real noise) const override
    {
        Real* const outputs = curr->outvals +n1stNeuron;
        Real* const inputs = curr->in_vals +n1stNeuron;
        for (int n=0; n<nNeurons; n++)
                inputs[n] = *(biases +n1stBias +n);

        for (const auto & link : *input_links)
                      link->propagate(curr,curr,weights);

        if(recurrent_link not_eq nullptr && prev not_eq nullptr)
            recurrent_link->propagate(prev,curr,weights);

        Func::eval(inputs, outputs, nNeurons);
    }
    void backPropagate( Activation* const prev,  Activation* const curr, const Activation* const next, 
                       Grads* const grad, const Real* const weights) const override
    {
        const Real* const inputs = curr->in_vals +n1stNeuron;
        Real* const deltas = curr->errvals +n1stNeuron;
        Func::mulDiff(inputs, deltas, nNeurons);

        for (const auto & link : *input_links)
                      link->backPropagate(curr,curr,weights,grad->_W);

        if(recurrent_link not_eq nullptr && prev not_eq nullptr)
            recurrent_link->backPropagate(prev,curr,weights,grad->_W);

        for (int n=0; n<nNeurons; n++) *(grad->_B +n1stBias +n) += deltas[n];
    }
};

template<typename outFunc>
class Conv2DLayer : public Layer
{
    const int nNeurons, n1stNeuron, n1stBias, outputWidth, outputHeight, outputDepth;
    const vector<LinkToConv2D*>* const input_links;
    const bool last;
public:
    typedef outFunc Func;
    Conv2DLayer(int nNeurons, int n1stNeuron, int n1stBias, const vector<LinkToConv2D*>* const nl_il, bool last) :
	nNeurons(nNeurons), n1stNeuron(n1stNeuron), n1stBias(n1stBias), 
    outputWidth(outputWidth), outputHeight(outputHeight), outputDepth(outputDepth), input_links(nl_il), last(last)
    {
	   printf("Conv2D Layer of size %d (%d x %d x %d), with first ID %d and first bias ID %d\n",
              nNeurons,outputWidth,outputHeight,outputDepth, n1stNeuron, n1stBias);
    }

    void propagate(const Activation* const prev, Activation* const curr, 
                   const Real* const weights, const Real* const biases, const Real noise) const  override
    {
        Real* const inputs  = curr->in_vals +n1stNeuron;
        Real* const outputs = curr->outvals +n1stNeuron;

        for(int o=0; o<nNeurons;  o++)
            inputs[o] = *(biases +n1stBias +o);

        for (const auto & link : *input_links)
                      link->propagate(curr,curr,weights);

        //recurrent con2D? I get them for free with this code, but let's hope it never comes to that
        //if(recurrent_link not_eq nullptr && prev not_eq nullptr)
        //    recurrent_link->propagate(prev,curr,weights);

        assert(nNeurons == outputWidth*outputHeight*outputDepth);
        Func::eval(inputs, outputs, nNeurons);
    }
	void backPropagate( Activation* const prev,  Activation* const curr, const Activation* const next, 
                       Grads* const grad, const Real* const weights) const  override
    {
        const Real* const inputs = curr->in_vals +n1stNeuron;
        Real* const errors = curr->errvals +n1stNeuron;
        Func::mulDiff(inputs, errors, nNeurons);

        for (const auto & link : *input_links)
            link->backPropagate(curr,curr,weights,grad->_W);

        //recurrent con2D? I get them for free with this code, but let's hope it never comes to that
        //if(recurrent_link not_eq nullptr && prev not_eq nullptr)
        //    recurrent_link->backPropagate(prev,curr,weights,grad->_W);

        for (int n=0; n<nNeurons; n++) *(grad->_B +n1stBias +n) += errors[n];
    }
};

template<typename outFunc, typename gateFunc, typename cellFunc>
class LSTMLayer: public Layer
{
    const int nNeurons, n1stNeuron, n1stBias, n1stCell, n1stBiasIG, n1stBiasFG, n1stBiasOG;
    const vector<LinkToLSTM*>* const input_links;
    const LinkToLSTM* const recurrent_link;
    const bool last;
public:
    typedef outFunc Func;
    typedef gateFunc Sigm;
    typedef cellFunc Cell;
    
    LSTMLayer(int nNeurons, int n1stNeuron, int indState,
              int n1stBias, int n1stBiasIG, int n1stBiasFG, int n1stBiasOG,
              const vector<LinkToLSTM*>* const rl_il, const LinkToLSTM* const rl_rl, bool last) :
    nNeurons(nNeurons), n1stNeuron(n1stNeuron), n1stBias(n1stBias), n1stCell(indState), 
    n1stBiasIG(n1stBiasIG), n1stBiasFG(n1stBiasFG), n1stBiasOG(n1stBiasOG), 
    input_links(rl_il), recurrent_link(rl_rl), last(last)
    {
	    printf("LSTM Layer of size %d, with first ID %d, first cell ID %d, and first bias ID %d\n",nNeurons, n1stNeuron, n1stCell, n1stBias);
        assert(n1stBiasIG==n1stBias  +nNeurons);
        assert(n1stBiasFG==n1stBiasIG+nNeurons);
        assert(n1stBiasOG==n1stBiasFG+nNeurons);
    }
    
    void propagate(const Activation* const prev, Activation* const curr, 
                   const Real* const weights, const Real* const biases, const Real noise) const  override
    {
        Real* const outputI = curr->oIGates +n1stCell;
        Real* const outputF = curr->oFGates +n1stCell;
        Real* const outputO = curr->oOGates +n1stCell;
        Real* const outputC = curr->oMCell +n1stCell;
        Real* const inputs = curr->in_vals +n1stNeuron;
        Real* const inputI = curr->iIGates +n1stCell;
        Real* const inputF = curr->iFGates +n1stCell;
        Real* const inputO = curr->iOGates +n1stCell;

        for (int n=0; n<nNeurons; n++) {
            inputs[n] = *(biases +n1stBias +n);
            inputI[n] = *(biases +n1stBiasIG +n);
            inputF[n] = *(biases +n1stBiasFG +n);
            inputO[n] = *(biases +n1stBiasOG +n);
        }

        for (const auto & link : *input_links)
                      link->propagate(curr,curr,weights);

        if(recurrent_link not_eq nullptr && prev not_eq nullptr)
            recurrent_link->propagate(prev,curr,weights);

        Cell::eval(inputs, outputC, nNeurons);
        Sigm::eval(inputI, outputI, nNeurons);
        Sigm::eval(inputF, outputF, nNeurons);
        Sigm::eval(inputO, outputO, nNeurons);
        for (int o=0; o<nNeurons; o++)
            *(curr->ostates +n1stCell +o) = outputC[o] * outputI[o] +
                    (prev==nullptr ?  0 : *(prev->ostates +n1stCell +o) * outputF[o]);

        Func::eval(curr->ostates +n1stCell, curr->outvals +n1stNeuron, nNeurons);
        for (int o=0; o<nNeurons; o++) *(curr->outvals +n1stNeuron +o) *= outputO[o];
    }
	void backPropagate( Activation* const prev,  Activation* const curr, const Activation* const next, 
                       Grads* const grad, const Real* const weights) const  override
    {
        const Real* const inputs = curr->in_vals +n1stNeuron;
        const Real* const inputI = curr->iIGates +n1stCell;
        const Real* const inputF = curr->iFGates +n1stCell;
        const Real* const inputO = curr->iOGates +n1stCell;
        const Real* const outputI = curr->oIGates +n1stCell;
        const Real* const outputF = curr->oFGates +n1stCell;
        const Real* const outputO = curr->oOGates +n1stCell;
        const Real* const outputC = curr->oMCell +n1stCell;
        Real* const deltas = curr->errvals +n1stNeuron;
        Real* const deltaI = curr->eIGates +n1stCell;
        Real* const deltaF = curr->eFGates +n1stCell;
        Real* const deltaO = curr->eOGates +n1stCell;
        Real* const deltaC = curr->eMCell +n1stCell;

        Real evalCurrState[nNeurons], diffCurrState[nNeurons];
        
        Cell::evalDiff(inputs, deltaC, outputI, nNeurons);
        Sigm::evalDiff(inputI, deltaI, outputC, nNeurons);
        Sigm::evalDiff(inputO, deltaO, deltas,  nNeurons);
        Func::eval(curr->ostates +n1stCell, evalCurrState, nNeurons);
        Func::evalDiff(curr->ostates +n1stCell, diffCurrState, nNeurons);

        for (int o=0; o<nNeurons; o++) 
            deltas[o]  = deltas[o] * outputO[o] * diffCurrState[o] +
                    (next==nullptr ?  0 : *(next->errvals+n1stNeuron+o)* *(next->oFGates+n1stCell+o));

        if (prev==nullptr) for (int o=0; o<nNeurons; o++) deltaF[o] = 0.;
        else Sigm::evalDiff(inputF, deltaF, prev->ostates+n1stCell, nNeurons);
		
		for (int o=0; o<nNeurons; o++) {
            deltaC[o] *= deltas[o];
            deltaI[o] *= deltas[o];
            deltaO[o] *= evalCurrState[o];
			deltaF[o] *= deltas[o];
		}

	for (const auto & link : *input_links)
				  link->backPropagate(curr,curr,weights,grad->_W);

	if(recurrent_link not_eq nullptr && prev not_eq nullptr)
		recurrent_link->backPropagate(prev,curr,weights,grad->_W);

	for (int n=0; n<nNeurons; n++)  { //grad bias == delta
		*(grad->_B +n1stBias   +n) += deltaC[n] ;
		*(grad->_B +n1stBiasIG +n) += deltaI[n];
		*(grad->_B +n1stBiasFG +n) += deltaF[n];
		*(grad->_B +n1stBiasOG +n) += deltaO[n];
	}
}
};

class WhiteningLayer: public Layer
{
    const WhiteningLink* const link;
    const int nNeurons, n1stNeuron;
    mt19937* const gen;
public:
	WhiteningLayer(int nNeurons, int n1stNeuron, const WhiteningLink* const nl_il, mt19937* const gen) :
    nNeurons(nNeurons), n1stNeuron(n1stNeuron), link(nl_il), gen(gen)
    {
        printf("Whitening layer of size %d starting from ID %d\n",nNeurons,n1stNeuron);
    }

    void propagate(const Activation* const prev, Activation* const curr, 
                   const Real* const weights, const Real* const biases, const Real noise) const  override
    {
        Real* const inputs = curr->in_vals + n1stNeuron;
        Real* const outputs = curr->outvals + n1stNeuron;
        const Real* const link_inputs = curr->outvals +link->iI;
        // 4 parameters per neuron:
        const Real* const link_means = weights + link->iW;
        const Real* const link_vars = weights + link->iW +nNeurons;
        const Real* const link_scales = weights + link->iW +2*nNeurons;
        const Real* const link_shifts = weights + link->iW +3*nNeurons;

        for (int n=0; n<nNeurons; n++) {
                const Real std = std::max(std::numeric_limits<Real>::epsilon(), link_vars[n]);
                const Real xhat = (link_inputs[n] - link_means[n])/std::sqrt(std);
                inputs[n] = xhat;
        }
        
        if (noise>0) {
            normal_distribution<Real> dis(0.,noise);
            for (int n=0; n<nNeurons; n++)
                inputs[n] += dis(*gen);
        }
        
        for (int n=0; n<nNeurons; n++)
            outputs[n] = link_scales[n]*inputs[n] + link_shifts[n]; 
    }
	void backPropagate( Activation* const prev,  Activation* const curr, const Activation* const next, 
                       Grads* const grad, const Real* const weights) const  override
    {
        Real* const link_errors = curr->errvals + link->iI;
        const Real* const errors = curr->errvals +n1stNeuron;
        const Real* const inputs = curr->in_vals + n1stNeuron;
        const Real* const link_inputs = curr->outvals +link->iI;
        const Real* const link_means = weights + link->iW;
        Real* const grad_means = grad->_W + link->iW;
        const Real* const link_vars = weights + link->iW +nNeurons;
        Real* const grad_vars = grad->_W + link->iW +nNeurons;
        const Real* const link_scales = weights + link->iW +2*nNeurons;
        Real* const grad_scales = grad->_W + link->iW +2*nNeurons;
        Real* const grad_shifts = grad->_W + link->iW +3*nNeurons;

        for (int n=0; n<nNeurons; n++)  {
            const Real std = std::max(std::numeric_limits<Real>::epsilon(), link_vars[n]);
            link_errors[n] = errors[n]*link_scales[n]/std::sqrt(std);
#ifndef _whitenTarget_
            const Real dEdMean = link_inputs[n]-link_means[n];
            grad_means[n] += 0.001*dEdMean;
            grad_vars[n] += 0.001*(dEdMean*dEdMean - link_vars[n]);
#endif
            grad_scales[n] += inputs[n]*errors[n];
            grad_shifts[n] += errors[n];
        }
    }
};
