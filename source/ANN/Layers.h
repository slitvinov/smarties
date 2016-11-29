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
//#include <cblas.h>
#include <iostream>
using namespace std;

class Layer
{
    public:
    virtual void propagate(const Activation* const prev, Activation* const curr, 
                           const Real* const weights, const Real* const biases, const Real noise) const = 0;
    virtual void backPropagate( Activation* const prev,  Activation* const curr, const Activation* const next, Grads* const grad, const Real* const weights, const Real* const biases) const = 0;

    void propagate(Activation* const curr, const Real* const weights, const Real* const biases, const Real noise) const 
        { return propagate(nullptr, curr, weights, biases, noise); }
    void backPropagate( Activation* const curr, Grads* const grad, const Real* const weights, const Real* const biases) const
        { return backPropagate(nullptr, curr, nullptr, grad, weights, biases); }
};

template<typename outFunc>
class NormalLayer: public Layer
{
    const int nNeurons, n1stNeuron, n1stBias, nNeurons_simd;
    const vector<NormalLink*>* const input_links;
    const NormalLink* const recurrent_link;
public:
    typedef outFunc Func;
    NormalLayer(int nNeurons, int n1stNeuron, int n1stBias, const vector<NormalLink*>* const nl_il, const NormalLink* const nl_rl, const int nn_simd) :
    nNeurons(nNeurons), n1stNeuron(n1stNeuron), n1stBias(n1stBias), nNeurons_simd(nn_simd), input_links(nl_il), recurrent_link(nl_rl)
    {
	   printf("Normal Layer of size %d, with first ID %d and first bias ID %d\n",nNeurons, n1stNeuron, n1stBias);
    }
    
    void propagate(const Activation* const prev, Activation* const curr, 
                   const Real* const weights, const Real* const biases, const Real noise) const override
    {
        Real* __restrict__ const outputs = curr->outvals +n1stNeuron;
        Real* __restrict__ const inputs = curr->in_vals +n1stNeuron;
        const Real* __restrict__ const bias = biases +n1stBias;
        __builtin_assume_aligned(outputs,  __vec_width__);
        __builtin_assume_aligned(inputs, __vec_width__);
        __builtin_assume_aligned(bias, __vec_width__);

        for (int n=0; n<nNeurons; n++) inputs[n] = bias[n];

        for (const auto & link : *input_links)
            link->propagate(curr,curr,weights);
/*
        	cblas_dgemv(CblasRowMajor, CblasTrans, link->nI, nNeurons_simd,
        				1.0, weights  + link->iW, nNeurons_simd,
						curr->outvals + link->iI, 1,
						1.0, inputs, 1);
*/
        if(recurrent_link not_eq nullptr && prev not_eq nullptr)
            recurrent_link->propagate(prev,curr,weights);
/*
        	cblas_dgemv(CblasRowMajor, CblasTrans, nNeurons, nNeurons_simd,
        				1.0, weights  +recurrent_link->iW, nNeurons_simd,
						prev->outvals +n1stNeuron, 1,
						1.0, inputs, 1);
*/
        Func::eval(inputs, outputs, nNeurons);
    }
    void backPropagate( Activation* const prev,  Activation* const curr, const Activation* const next, 
                       Grads* const grad, const Real* const weights, const Real* const biases) const override
    {
        const Real* __restrict__ const inputs = curr->in_vals +n1stNeuron;
        Real* __restrict__ const deltas = curr->errvals +n1stNeuron;
        Real* __restrict__ const gradbias = grad->_B +n1stBias;
        __builtin_assume_aligned(deltas,  __vec_width__);
        __builtin_assume_aligned(inputs, __vec_width__);
        __builtin_assume_aligned(gradbias, __vec_width__);

        Func::mulDiff(inputs, deltas, nNeurons);

        for (const auto & link : *input_links)
                      link->backPropagate(curr,curr,weights,grad->_W);

        if(recurrent_link not_eq nullptr && prev not_eq nullptr)
            recurrent_link->backPropagate(prev,curr,weights,grad->_W);

        for (int n=0; n<nNeurons; n++) gradbias[n] += deltas[n];
    }
};

template<typename outFunc>
class Conv2DLayer : public Layer
{
    const int nNeurons, n1stNeuron, n1stBias, outputWidth, outputHeight, outputDepth, nNeurons_simd;
    const vector<LinkToConv2D*>* const input_links;
public:
    typedef outFunc Func;
    Conv2DLayer(int nNeurons, int n1stNeuron, int n1stBias, const vector<LinkToConv2D*>* const nl_il, const int nn_simd) :
	nNeurons(nNeurons), n1stNeuron(n1stNeuron), n1stBias(n1stBias), 
    outputWidth(outputWidth), outputHeight(outputHeight), outputDepth(outputDepth), nNeurons_simd(nn_simd), input_links(nl_il)
    {
	   printf("Conv2D Layer of size %d (%d x %d x %d), with first ID %d and first bias ID %d\n",
              nNeurons,outputWidth,outputHeight,outputDepth, n1stNeuron, n1stBias);
    }

    void propagate(const Activation* const prev, Activation* const curr, 
                   const Real* const weights, const Real* const biases, const Real noise) const  override
    {
        Real* __restrict__ const inputs  = curr->in_vals +n1stNeuron;
        Real* __restrict__ const outputs = curr->outvals +n1stNeuron;
        const Real* __restrict__ const bias = biases +n1stBias;
        __builtin_assume_aligned(outputs,  __vec_width__);
        __builtin_assume_aligned(inputs, __vec_width__);
        __builtin_assume_aligned(bias, __vec_width__);

        for(int o=0; o<nNeurons;  o++) inputs[o] = bias[o];

        for (const auto & link : *input_links)
                      link->propagate(curr,curr,weights);

        //recurrent con2D? I get them for free with this code, but let's hope it never comes to that
        //if(recurrent_link not_eq nullptr && prev not_eq nullptr)
        //    recurrent_link->propagate(prev,curr,weights);

        Func::eval(inputs, outputs, nNeurons);
    }
	void backPropagate( Activation* const prev,  Activation* const curr, const Activation* const next, 
                       Grads* const grad, const Real* const weights, const Real* const biases) const  override
    {
        const Real* __restrict__ const inputs = curr->in_vals +n1stNeuron;
        Real* __restrict__ const errors = curr->errvals +n1stNeuron;
        Real* __restrict__ const gradbias = grad->_B +n1stBias;
        __builtin_assume_aligned(errors,  __vec_width__);
        __builtin_assume_aligned(inputs, __vec_width__);
        __builtin_assume_aligned(gradbias, __vec_width__);

        Func::mulDiff(inputs, errors, nNeurons);

        for (const auto & link : *input_links)
            link->backPropagate(curr,curr,weights,grad->_W);

        //recurrent con2D? I get them for free with this code, but let's hope it never comes to that
        //if(recurrent_link not_eq nullptr && prev not_eq nullptr)
        //    recurrent_link->backPropagate(prev,curr,weights,grad->_W);

        for (int n=0; n<nNeurons; n++) gradbias[n] += errors[n];
    }
};

template<typename outFunc, typename gateFunc, typename cellFunc>
class LSTMLayer: public Layer
{
    const int nNeurons, n1stNeuron, n1stBias, n1stCell, n1stBiasIG, n1stBiasFG, n1stBiasOG, nNeurons_simd;
    const vector<LinkToLSTM*>* const input_links;
    const LinkToLSTM* const recurrent_link;
public:
    typedef outFunc Func;
    typedef gateFunc Sigm;
    typedef cellFunc Cell;
    
    LSTMLayer(int nNeurons, int n1stNeuron, int indState,
              int n1stBias, int n1stBiasIG, int n1stBiasFG, int n1stBiasOG,
              const vector<LinkToLSTM*>* const rl_il, const LinkToLSTM* const rl_rl, const int nn_simd) :
    nNeurons(nNeurons), n1stNeuron(n1stNeuron), n1stBias(n1stBias), n1stCell(indState), 
    n1stBiasIG(n1stBiasIG), n1stBiasFG(n1stBiasFG), n1stBiasOG(n1stBiasOG), nNeurons_simd(nn_simd),
    input_links(rl_il), recurrent_link(rl_rl)
    {
	    printf("LSTM Layer of size %d, with first ID %d, first cell ID %d, and first bias ID %d\n",nNeurons, n1stNeuron, n1stCell, n1stBias);
        assert(n1stBiasIG==n1stBias  +nn_simd);
        assert(n1stBiasFG==n1stBiasIG+nn_simd);
        assert(n1stBiasOG==n1stBiasFG+nn_simd);
    }
    
    void propagate(const Activation* const prev, Activation* const curr, 
                   const Real* const weights, const Real* const biases, const Real noise) const  override
    {
        Real* __restrict__ const outputI = curr->oIGates +n1stCell;
        Real* __restrict__ const outputF = curr->oFGates +n1stCell;
        Real* __restrict__ const outputO = curr->oOGates +n1stCell;
        Real* __restrict__ const outputC = curr->oMCell +n1stCell;
        Real* __restrict__ const inputs = curr->in_vals +n1stNeuron;
        Real* __restrict__ const inputI = curr->iIGates +n1stCell;
        Real* __restrict__ const inputF = curr->iFGates +n1stCell;
        Real* __restrict__ const inputO = curr->iOGates +n1stCell;
        const Real* __restrict__ const biasC = biases +n1stBias;
        const Real* __restrict__ const biasI = biases +n1stBiasIG;
        const Real* __restrict__ const biasF = biases +n1stBiasFG;
        const Real* __restrict__ const biasO = biases +n1stBiasOG;
        __builtin_assume_aligned(outputI,  __vec_width__);
        __builtin_assume_aligned(outputF, __vec_width__);
        __builtin_assume_aligned(outputO, __vec_width__);
        __builtin_assume_aligned(outputC,  __vec_width__);
        __builtin_assume_aligned(inputs, __vec_width__);
        __builtin_assume_aligned(inputI, __vec_width__);
        __builtin_assume_aligned(inputF,  __vec_width__);
        __builtin_assume_aligned(inputO, __vec_width__);
        __builtin_assume_aligned(biasC, __vec_width__);
        __builtin_assume_aligned(biasI, __vec_width__);
        __builtin_assume_aligned(biasF, __vec_width__);
        __builtin_assume_aligned(biasO, __vec_width__);

        for (int n=0; n<nNeurons; n++) {
            inputs[n] = biasC[n];
            inputI[n] = biasI[n];
            inputF[n] = biasF[n];
            inputO[n] = biasO[n];
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
                       Grads* const grad, const Real* const weights, const Real* const biases) const  override
    {
        const Real* __restrict__ const inputs = curr->in_vals +n1stNeuron;
        const Real* __restrict__ const inputI = curr->iIGates +n1stCell;
        const Real* __restrict__ const inputF = curr->iFGates +n1stCell;
        const Real* __restrict__ const inputO = curr->iOGates +n1stCell;
        const Real* __restrict__ const outputI = curr->oIGates +n1stCell;
        const Real* __restrict__ const outputF = curr->oFGates +n1stCell;
        const Real* __restrict__ const outputO = curr->oOGates +n1stCell;
        const Real* __restrict__ const outputC = curr->oMCell +n1stCell;
        Real* __restrict__ const deltas = curr->errvals +n1stNeuron;
        Real* __restrict__ const deltaI = curr->eIGates +n1stCell;
        Real* __restrict__ const deltaF = curr->eFGates +n1stCell;
        Real* __restrict__ const deltaO = curr->eOGates +n1stCell;
        Real* __restrict__ const deltaC = curr->eMCell +n1stCell;
        Real* __restrict__ const gradbiasC = grad->_B +n1stBias;
        Real* __restrict__ const gradbiasI = grad->_B +n1stBiasIG;
        Real* __restrict__ const gradbiasF = grad->_B +n1stBiasFG;
        Real* __restrict__ const gradbiasO = grad->_B +n1stBiasOG;
        __builtin_assume_aligned(outputI,  __vec_width__);
        __builtin_assume_aligned(outputF, __vec_width__);
        __builtin_assume_aligned(outputO, __vec_width__);
        __builtin_assume_aligned(outputC,  __vec_width__);
        __builtin_assume_aligned(inputs, __vec_width__);
        __builtin_assume_aligned(inputI, __vec_width__);
        __builtin_assume_aligned(inputF,  __vec_width__);
        __builtin_assume_aligned(inputO, __vec_width__);
        __builtin_assume_aligned(deltas, __vec_width__);
        __builtin_assume_aligned(deltaI, __vec_width__);
        __builtin_assume_aligned(deltaF, __vec_width__);
        __builtin_assume_aligned(deltaO, __vec_width__);
        __builtin_assume_aligned(deltaC, __vec_width__);
        __builtin_assume_aligned(gradbiasC, __vec_width__);
        __builtin_assume_aligned(gradbiasI, __vec_width__);
        __builtin_assume_aligned(gradbiasF, __vec_width__);
        __builtin_assume_aligned(gradbiasO, __vec_width__);

        Real *evalCurrState, *diffCurrState;
        _allocateQuick(diffCurrState, nNeurons)
        _allocateQuick(evalCurrState, nNeurons)
        
        Cell::evalDiff(inputs, deltaC, outputI, nNeurons);
        Sigm::evalDiff(inputI, deltaI, outputC, nNeurons);
        Sigm::evalDiff(inputO, deltaO, deltas,  nNeurons);
        Func::eval(curr->ostates +n1stCell, evalCurrState, nNeurons);
        Func::evalDiff(curr->ostates +n1stCell, diffCurrState, nNeurons);

        for (int o=0; o<nNeurons; o++)
            deltas[o]  = deltas[o] * outputO[o] * diffCurrState[o] +
                    (next==nullptr ?  0 : *(next->errvals+n1stNeuron+o)* *(next->oFGates+n1stCell+o));

        if (prev==nullptr)
        	for (int o=0; o<nNeurons; o++) deltaF[o] = 0.;
        else
        	Sigm::evalDiff(inputF, deltaF, prev->ostates+n1stCell, nNeurons);
		
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
			gradbiasC[n] += deltaC[n];
			gradbiasI[n] += deltaI[n];
			gradbiasF[n] += deltaF[n];
			gradbiasO[n] += deltaO[n];
		}

		_myfree(evalCurrState)
		_myfree(diffCurrState)
	}
};

class WhiteningLayer: public Layer
{
    const WhiteningLink* const link;
    const int nNeurons, n1stNeuron, n1stBias, nNeurons_simd;
    mt19937* const gen;
public:
	WhiteningLayer(int nNeurons, int n1stNeuron, int n1stBias, const WhiteningLink* const nl_il, mt19937* const gen, const int nn_simd) :
    nNeurons(nNeurons), n1stNeuron(n1stNeuron), n1stBias(n1stBias), nNeurons_simd(nn_simd), link(nl_il), gen(gen)
    {
        printf("Whitening layer of size %d starting from ID %d. Means/vars start at bias %d\n",nNeurons,n1stNeuron,n1stBias);
    }

    void propagate(const Activation* const prev, Activation* const curr, 
                   const Real* const weights, const Real* const biases, const Real noise) const  override
    {
        Real* __restrict__ const inputs = curr->in_vals + n1stNeuron;
        Real* __restrict__ const outputs = curr->outvals + n1stNeuron;
        const Real* __restrict__ const link_inputs = curr->outvals +link->iI;
        const Real* __restrict__ const link_means = biases + n1stBias;
        const Real* __restrict__ const link_vars = biases + n1stBias +nNeurons_simd;
        const Real* __restrict__ const link_shifts = weights + link->iW;
        const Real* __restrict__ const link_scales = weights + link->iW +nNeurons_simd;
        __builtin_assume_aligned(outputs,  __vec_width__);
        __builtin_assume_aligned(inputs, __vec_width__);
        __builtin_assume_aligned(link_inputs, __vec_width__);
        __builtin_assume_aligned(link_means,  __vec_width__);
        __builtin_assume_aligned(link_vars, __vec_width__);
        __builtin_assume_aligned(link_shifts, __vec_width__);
        __builtin_assume_aligned(link_scales,  __vec_width__);
        const Real _eps = 1e-9;//std::numeric_limits<Real>::epsilon();

        for (int n=0; n<nNeurons; n++)
                inputs[n] = (link_inputs[n] - link_means[n])/std::sqrt(std::max(_eps,  link_vars[n]));
        //for (int n=0; n<nNeurons; n++) inputs[n] = link_inputs[n];
        
        if (noise>0) {
            normal_distribution<Real> dis(0.,noise);
            for (int n=0; n<nNeurons; n++)
                inputs[n] += dis(*gen);
        }
        
        for (int n=0; n<nNeurons; n++)
            outputs[n] = link_scales[n]*inputs[n] + link_shifts[n]; 
    }
	void backPropagate( Activation* const prev,  Activation* const curr, const Activation* const next, 
                       Grads* const grad, const Real* const weights, const Real* const biases) const  override
    {
        Real* __restrict__ const link_errors = curr->errvals + link->iI;
        const Real* __restrict__ const errors = curr->errvals +n1stNeuron;
        const Real* __restrict__ const inputs = curr->in_vals + n1stNeuron;
        const Real* __restrict__ const link_inputs = curr->outvals +link->iI;
        const Real* __restrict__ const link_means = biases + n1stBias;
        const Real* __restrict__ const link_vars = biases + n1stBias +nNeurons_simd;
        const Real* __restrict__ const link_shifts = weights + link->iW;
        const Real* __restrict__ const link_scales = weights + link->iW +nNeurons_simd;
        Real* __restrict__ const grad_means = grad->_B + n1stBias;
        Real* __restrict__ const grad_vars = grad->_B + n1stBias +nNeurons_simd;
        Real* __restrict__ const grad_shifts = grad->_W + link->iW;
        Real* __restrict__ const grad_scales = grad->_W + link->iW +nNeurons_simd;
        __builtin_assume_aligned(errors,  __vec_width__);
        __builtin_assume_aligned(inputs, __vec_width__);
        __builtin_assume_aligned(link_errors, __vec_width__);
        __builtin_assume_aligned(link_inputs, __vec_width__);
        __builtin_assume_aligned(link_means,  __vec_width__);
        __builtin_assume_aligned(link_vars, __vec_width__);
        __builtin_assume_aligned(link_shifts, __vec_width__);
        __builtin_assume_aligned(link_scales,  __vec_width__);
        __builtin_assume_aligned(grad_means,  __vec_width__);
        __builtin_assume_aligned(grad_vars, __vec_width__);
        __builtin_assume_aligned(grad_shifts, __vec_width__);
        __builtin_assume_aligned(grad_scales,  __vec_width__);
        const Real _eps = 1e-9; //std::numeric_limits<Real>::epsilon();

        for (int n=0; n<nNeurons; n++)  {
            const Real invstd = 1./std::sqrt(std::max(_eps, link_vars[n]));
            const Real dEdXhat = errors[n]*link_scales[n];

            //mean increases if input is greater than mean
            const Real dMudX = (link_inputs[n] - link_means[n]);
            //std increases if input is less than mean
            const Real dStddX = (dMudX*dMudX - link_vars[n]);

            //const Real dXhatdMu = -invstd;
            //const Real fac = std::max(_eps, std::pow(link_vars[n],1.5));
            //const Real dXhatdStd = -.5*(link_inputs[n]-link_means[n])*std::pow(invstd, 3);
            //const Real dXhatdX = invstd;
            
            grad_means[n] += dMudX;
            if (dStddX>0 || link_vars[n]>_eps)
            	grad_vars[n] += dStddX;

            const Real pid_avg =  dEdXhat*dMudX<0 ? -0.25*invstd : 0.25*invstd;
            const Real pid_std = link_inputs[n]-link_means[n] > 0  ? 
                                 ( dStddX*dEdXhat<0 ? -0.25*invstd : 0.25*invstd)
                                                                   :
                                 ( dStddX*dEdXhat>0 ? -0.25*invstd : 0.25*invstd);

            link_errors[n] = dEdXhat*(invstd + pid_avg + pid_std);
            
            grad_scales[n] += inputs[n]*errors[n];
            grad_shifts[n] += errors[n];
            
            //link_errors[n] = dEdXhat*invstd;
        }
    }
};
