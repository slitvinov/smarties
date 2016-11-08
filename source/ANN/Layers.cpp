/*
 *  Layers.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include "Layers.h"
#include <cassert>

using namespace ErrorHandling;


void NormalLayer::propagate(Activation* const curr, const Real* const weights, const Real* const biases) const {
	return propagate(nullptr, curr, weights, biases);
}

void NormalLayer::backPropagate(const Activation* const curr, Grads* const grad, const Real* const weights) const
{
	return backPropagate(nullptr, curr, nullptr, grad, weights);
}

void NormalLayer::propagate(
		const Activation* const prev, Activation* const curr,
		const Real* const weights, const Real* const biases) const
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

void LSTMLayer::propagate(
		const Activation* const prev, Activation* const curr,
		const Real* const weights, const Real* const biases) const
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
        		prev==nullptr ?  0 : *(prev->ostates +n1stCell +o) * outputF[o];

	Func::eval(curr->ostates +n1stCell, curr->outvals +n1stNeuron, nNeurons);
	for (int o=0; o<nNeurons; o++) *(curr->outvals +n1stNeuron +o) *= outputO[o];
}

void Conv2DLayer::propagate(
		const Activation* const prev, Activation* const curr,
		const Real* const weights, const Real* const biases) const
{
	Real* const inputs  = curr->in_vals +n1stNeuron;
	Real* const outputs = curr->outvals +n1stNeuron;

	for(int o=0; o<nNeurons;  o++)
		inputs[o] = *(biases +n1stBias +o);

	for (const auto & link : *input_links)
				  link->propagate(curr,curr,weights);

	//recurrent con2D? I get them for free with this code, but let's hope it never comes to that
	if(recurrent_link not_eq nullptr && prev not_eq nullptr)
		recurrent_link->propagate(prev,curr,weights);

	assert(nNeurons == outputWidth*outputHeight*outputDepth);
	Func::eval(inputs, outputs, nNeurons);
}

void NormalLayer::backPropagate(
		const Activation* const prev, const Activation* const curr, const Activation* const next,
		Grads* const grad, const Real* const weights) const
{
	const Real* const inputs = curr->in_vals +n1stNeuron;
	const Real* const deltas = curr->errvals +n1stNeuron;

	Real funcDiff[nNeurons];
	Func::evalDiff(inputs, funcDiff, nNeurons);
	for (int o=0; o<nNeurons; o++) deltas[o] *= funcDiff[o];

    for (const auto & link : *input_links)
    			  link->backPropagate(curr,curr,weights,grad->_W);

    if(recurrent_link not_eq nullptr && prev not_eq nullptr)
    	recurrent_link->backPropagate(prev,curr,weights,grad->_W);

    for (int n=0; n<nNeurons; n++) *(grad->_B +n1stBias +n) += deltas[n];
}

void LSTMLayer::backPropagate(
		const Activation* const prev, const Activation* const curr, const Activation* const next,
		Grads* const grad, const Real* const weights) const
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
	Cell::evalDiff(inputs, deltaC, nNeurons);
	Sigm::evalDiff(inputI, deltaI, nNeurons);
	Sigm::evalDiff(inputO, deltaO, nNeurons);
	Func::eval(curr->ostates +n1stCell, evalCurrState, nNeurons);
	Func::evalDiff(curr->ostates +n1stCell, diffCurrState, nNeurons);

	for (int o=0; o<nNeurons; o++) {
		deltaO[o] *= deltas[o] * evalCurrState[o];
		deltas[o]  = deltas[o] * outputO[o] * diffCurrState[o] +
				next==nullptr ?  0 : *(next->errvals+n1stNeuron+o)* *(next->oFGates+n1stCell+o);
		deltaC[o] *= deltas[o] * outputI[o];
		deltaI[o] *= deltas[o] * outputC[o];
	}

	if (prev==nullptr) {
		for (int o=0; o<nNeurons; o++) deltaF = 0.;
	} else {
		Sigm::evalDiff(inputF, deltaF, nNeurons);
		for (int o=0; o<nNeurons; o++) deltaF[o] *= deltas[o] * *(prev->ostates +n1stCell+o);
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

void Conv2DLayer::backPropagate(
		const Activation* const prev, const Activation* const curr, const Activation* const next,
		Grads* const grad, const Real* const weights) const
{
	const Real* const inputs = curr->in_vals +n1stNeuron;
	const Real* const errors = curr->errvals +n1stNeuron;
	Real funcDiff[nNeurons];
	Func::evalDiff(inputs, funcDiff, nNeurons);
	for (int o=0; o<nNeurons; o++) errors[o] *= funcDiff[o];

	for (const auto & link : *input_links)
    	link->backPropagate(curr,curr,weights,grad->_W);

	//recurrent con2D? I get them for free with this code, but let's hope it never comes to that
	if(recurrent_link not_eq nullptr && prev not_eq nullptr)
		recurrent_link->backPropagate(prev,curr,weights,grad->_W);

	for (int n=0; n<nNeurons; n++) *(grad->_B +n1stBias +n) += errors[n];
}

void WhiteningLayer::propagate(
		const Activation* const prev, Activation* const curr,
		const Real* const weights, const Real* const biases) const
{
	assert(input_links->size() == 1);
	const Link* const link = (*input_links)[0]; //only one as input
	Real* const inputs = curr->in_vals + n1stNeuron;
	Real* const outputs = curr->outvals + n1stNeuron;
    const Real* const link_inputs = curr->outvals +link->iI;
	// 4 parameters per neuron:
	const Real* const link_means = weights + link->iW;
	const Real* const link_vars = weights + link->iW +nNeurons;
	const Real* const link_scales = weights + link->iW +2*nNeurons;
	const Real* const link_shifts = weights + link->iW +3*nNeurons;

    for (int n=0; n<nNeurons; n++) {
    	assert(link_vars[n]>std::numeric_limits<Real>::epsilon());
    	const Real xhat = (link_inputs[n] - link_means[n])/std::sqrt(link_vars[n]);
    	inputs[n] = xhat;
    	outputs[n] = link_scales[n]*xhat + link_shifts[n]; //y
    }
}

void WhiteningLayer::backPropagate(
		const Activation* const prev, const Activation* const curr, const Activation* const next,
		Grads* const grad, const Real* const weights) const
{
	const Link* const link = (*input_links)[0];
	Real* const link_errors = curr->errvals + link->iI;
	const Real* const errors = curr->errvals +n1stNeuron;
	const Real* const inputs = curr->in_vals + n1stNeuron;
    const Real* const link_inputs = curr->outvals +link->iI;
	const Real* const link_means = weights + link->iW;
	const Real* const grad_means = grad->_W + link->iW;
	const Real* const link_vars = weights + link->iW +nNeurons;
	const Real* const grad_vars = grad->_W + link->iW +nNeurons;
	const Real* const link_scales = weights + link->iW +2*nNeurons;
	const Real* const grad_scales = grad->_W + link->iW +2*nNeurons;
	const Real* const grad_shifts = grad->_W + link->iW +3*nNeurons;

    for (int n=0; n<nNeurons; n++)  {
    	link_errors[n] = errors[n]*link_scales[n]/std::sqrt(link_vars[n])
        const Real dEdMean = link_inputs[n]-link_means[n];
        grad_means[n] += dEdMean;
        grad_vars[n] += dEdMean*dEdMean - link_vars[n];
        grad_scales[n] += inputs[n]*errors[n];
        grad_shifts[n] += errors[n];
    }
}
