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

void NormalLayer::backPropagateDelta(
		Activation* const prev, Activation* const curr, const Activation* const next,
		const Real* const weights, const Real* const biases) const
{
	const Real* const inputs = curr->in_vals +n1stNeuron;
	Real* const errors = curr->errvals +n1stNeuron;
	Real funcDiff[nNeurons];
	Func::evalDiff(inputs, funcDiff, nNeurons);
	for (int o=0; o<nNeurons; o++) errors[o] *= funcDiff[o];

	for (const auto & link : *input_links) {
		const Real* const link_weights = weights + link->iW;
		Real* const link_errors = curr->errvals + link->iI;
		const int n_inputs = link->nI;
		for (int i = 0; i < n_inputs; i++)
			for (int o = 0; o < nNeurons; o++)
				link_errors[i] += errors[o] * link_weights[n_inputs*i + o];
    }

	if(recurrent_link not_eq nullptr && prev not_eq nullptr) {
		const Real* const link_weights =  weights + recurrent_link->iW;
		Real* const link_errors = prev->errvals + recurrent_link->iI;
		const int n_inputs = recurrent_link->nI;
		for (int i = 0; i < n_inputs; i++)
			for (int o = 0; o < nNeurons; o++)
				link_errors[i] += errors[o] * link_weights[n_inputs*i + o];
	}
}

void LSTMLayer::backPropagateDelta(
		Activation* const prev, Activation* const curr, const Activation* const next,
		const Real* const weights, const Real* const biases) const
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

	for (const auto & link : *input_links) {
		const Real* const w_toOgate = weights +link->iWO;
		const Real* const w_toFgate = weights +link->iWF;
		const Real* const w_toIgate = weights +link->iWI;
		const Real* const w_toCell = weights +link->iW;
		Real* const link_errors = curr->errvals + link->iI;
		const int n_inputs = link->nI;

		for (int i = 0; i < n_inputs; i++)
		for (int o = 0; o < nNeurons; o++) {
			const int cc = n_inputs*i + o;
			link_errors[i] += deltaO[o] * w_toOgate[cc] + deltaC[o] * w_toCell[cc] +
								deltaI[o] * w_toIgate[cc] + deltaF[o] * w_toFgate[cc];
		}
    }

	if(recurrent_link not_eq nullptr && prev not_eq nullptr) {
		const Real* const w_toOgate = weights +recurrent_link->iWO;
		const Real* const w_toFgate = weights +recurrent_link->iWF;
		const Real* const w_toIgate = weights +recurrent_link->iWI;
		const Real* const w_toCell = weights +recurrent_link->iW;
		Real* const link_errors = prev->errvals +recurrent_link->iI;
		const int n_inputs = recurrent_link->nI;

		for (int i = 0; i < n_inputs; i++)
		for (int o = 0; o < nNeurons; o++) {
			const int cc = n_inputs*i + o;
			link_errors[i] += deltaO[o] * w_toOgate[cc] + deltaC[o] * w_toCell[cc] +
								deltaI[o] * w_toIgate[cc] + deltaF[o] * w_toFgate[cc];
		}
    }
}

void Conv2D::backPropagateDelta(
		Activation* const prev, Activation* const curr, const Activation* const next,
		const Real* const weights, const Real* const biases) const
{
	const Real* const inputs = curr->in_vals +n1stNeuron;
	const Real* const errors = curr->errvals +n1stNeuron;
	Real funcDiff[nNeurons];
	Func::evalDiff(inputs, funcDiff, nNeurons);
	for (int o=0; o<nNeurons; o++) errors[o] *= funcDiff[o];

	for (const auto & link : *input_links) {
		const Real* const link_weights = weights + link->iW;
		Real* const link_errors = curr->errvals + link->iI;

		const int iW(link->inputWidth), iH(link->inputHeight);
		const int iD(link->inputDepth), fW(link->width), fH(link->height);

		for(int ox=0; ox<outputWidth;  ox++)
		for(int oy=0; oy<outputHeight; oy++) {
			const int ix = ox*link->strideX - link->padX;
			const int iy = oy*link->strideY - link->padY;
			for(int fx=0; fx<fW; fx++)
			for(int fy=0; fy<fH; fy++) {
				const int cx(ix+fx), cy(iy+fy);
				//padding: skip addition if outside input boundaries
				if (cx < 0 || cy < 0 || cx >= iW | cy >= iH) continue;

				for(int iz=0; iz<iD; iz++)
				for(int fz=0; fz<outputDepth; fz++) {
					assert(iz +iD*(cy +iH*cx) < link->nI);
					const int pinp = iz +iD*(cy +iH*cx);
					const int pout = fz +outputDepth*(oy +outputHeight*ox);
					const int fid = fz +outputDepth*(iz +iD*(fy +fH*fx));
					link_errors[pinp] += link_weights[fid]*errors[pout];
				}
			}
		}
	}
}

void WhiteningLayer::backPropagateDelta(
		Activation* const prev, Activation* const curr, const Activation* const next,
		const Real* const weights, const Real* const biases) const
{
    // TODO: how can I easily implement multivariate version?
	const Link* const link = (*input_links)[0];
	Real* const errors = curr->errvals +n1stNeuron;
	Real* const link_errors = curr->errvals + link->iI;
	const Real* const link_vars = weights + link->iW +nNeurons;
	const Real* const link_scales = weights + link->iW +2*nNeurons;

    for (int n=0; n<nNeurons; n++)
    	link_errors[n] = errors[n]*link_scales[n]/std::sqrt(link_vars[n])
}

void NormalLayer::backPropagateGrads(
		const Activation* const prev, const Activation* const curr,
		Grads* const grad, const Real* const weights) const
{
	const Real* const deltas = curr->errvals +n1stNeuron;
    for (int n=0; n<nNeurons; n++)  //grad bias == delta
        *(grad->_B +n1stBias +n) += deltas[n];

    for (const auto & link : *input_links) {
    	const Real* const layer_input = curr->outvals +link->iI;
		Real* const link_dEdW = grad->_W +link->iW;
		const int n_inputs = link->nI;

		for (int i = 0; i < n_inputs; i++)
			for (int o = 0; o < nNeurons; o++)
				link_dEdW[n_inputs*i +o] += layer_input[i] * deltas[o];
    }

    if(recurrent_link not_eq nullptr && prev not_eq nullptr) {
    	const Real* const layer_input = prev->outvals +recurrent_link->iI;
		Real* const link_dEdW = grad->_W +recurrent_link->iW;
		const int n_inputs = recurrent_link->nI;

		for (int i = 0; i < n_inputs; i++)
			for (int o = 0; o < nNeurons; o++)
				link_dEdW[n_inputs*i +o] += layer_input[i] * deltas[o];
    }
}

void LSTMLayer::backPropagateGrads(
		const Activation* const prev, const Activation* const curr,
		Grads* const grad, const Real* const weights) const
{
	const Real* const dEdInputOGate = curr->eOGates +n1stCell;
	const Real* const dEdInputFGate = curr->eFGates +n1stCell;
	const Real* const dEdInputIGate = curr->eIGates +n1stCell;
	const Real* const dEdInputCell = curr->eMCell +n1stCell;

	for (int n=0; n<nNeurons; n++)  { //grad bias == delta
		*(grad->_B +n1stBias   +n) += dEdInputCell[n] ;
		*(grad->_B +n1stBiasIG +n) += dEdInputIGate[n];
		*(grad->_B +n1stBiasFG +n) += dEdInputFGate[n];
		*(grad->_B +n1stBiasOG +n) += dEdInputOGate[n];
	}

	for (const auto & link : *input_links) {
		const Real* const layer_input = curr->outvals +link->iI;
		Real* const dw_toOgate = grad->_W +link->iWO;
		Real* const dw_toFgate = grad->_W +link->iWF;
		Real* const dw_toIgate = grad->_W +link->iWI;
		Real* const dw_toCell = grad->_W +link->iW;
		const int n_inputs = link->nI;

		for (int i = 0; i < n_inputs; i++)
		for (int o = 0; o < nNeurons; o++) {
			const int cc = n_inputs*i + o;
			dw_toOgate[cc] += layer_input[i] * dEdInputOGate[o];
			dw_toCell[cc]  += layer_input[i] * dEdInputCell[o] ;
			dw_toIgate[cc] += layer_input[i] * dEdInputIGate[o];
			dw_toFgate[cc] += layer_input[i] * dEdInputFGate[o];
		}
	}

	if(recurrent_link not_eq nullptr && prev not_eq nullptr) {
		const Real* const layer_input = prev->outvals +recurrent_link->iI;
		Real* const dw_toOgate = grad->_W +recurrent_link->iWO;
		Real* const dw_toFgate = grad->_W +recurrent_link->iWF;
		Real* const dw_toIgate = grad->_W +recurrent_link->iWI;
		Real* const dw_toCell = grad->_W +recurrent_link->iW;
		const int n_inputs = recurrent_link->nI;

		for (int i = 0; i < n_inputs; i++)
		for (int o = 0; o < nNeurons; o++) {
			const int cc = n_inputs*i + o;
			dw_toOgate[cc] += layer_input[i] * dEdInputOGate[o];
			dw_toCell[cc]  += layer_input[i] * dEdInputCell[o] ;
			dw_toIgate[cc] += layer_input[i] * dEdInputIGate[o];
			dw_toFgate[cc] += layer_input[i] * dEdInputFGate[o];
		}
	}
}

void Conv2D::backPropagateGrads(
		const Activation* const prev, const Activation* const curr,
		Grads* const grad, const Real* const weights) const
{
	const Real* const deltas = curr->errvals +n1stNeuron;
	for (int n=0; n<nNeurons; n++)  //grad bias == delta
		*(grad->_B +n1stBias +n) += deltas[n];

	for (const auto & link : *input_links) {
		const Real* const link_inputs = curr->outvals + link->iI;
		Real* const link_dEdW = grad->_W +link->iW;

		const int iW(link->inputWidth), iH(link->inputHeight);
		const int iD(link->inputDepth), fW(link->width), fH(link->height);

		for(int ox=0; ox<outputWidth;  ox++)
		for(int oy=0; oy<outputHeight; oy++) {
			const int ix = ox*link->strideX - link->padX;
			const int iy = oy*link->strideY - link->padY;
			for(int fx=0; fx<fW; fx++)
			for(int fy=0; fy<fH; fy++) {
				const int cx(ix+fx), cy(iy+fy);
				//padding: skip addition if outside input boundaries
				if (cx < 0 || cy < 0 || cx >= iW | cy >= iH) continue;

				for(int iz=0; iz<iD; iz++)
				for(int fz=0; fz<outputDepth; fz++) {
					assert(iz +iD*(cy +iH*cx) < link->nI);
					const int pinp = iz +iD*(cy +iH*cx);
					const int pout = fz +outputDepth*(oy +outputHeight*ox);
					const int fid = fz +outputDepth*(iz +iD*(fy +fH*fx));
					link_dEdW[fid] += link_inputs[pinp]*deltas[pout];
				}
			}
		}
	}
}

void WhiteningLayer::backPropagateGrads(
		const Activation* const prev, const Activation* const curr,
		Grads* const grad, const Real* const weights) const
{
	const Link* const link = (*input_links)[0];
	const Real* const errors = curr->errvals +n1stNeuron;
	const Real* const inputs = curr->in_vals + n1stNeuron;
    const Real* const link_inputs = curr->outvals +link->iI;
	const Real* const link_means = weights + link->iW;
	const Real* const grad_means = grad->_W + link->iW;
	const Real* const link_vars = weights + link->iW +nNeurons;
	const Real* const grad_vars = grad->_W + link->iW +nNeurons;
	const Real* const grad_scales = grad->_W + link->iW +2*nNeurons;
	const Real* const grad_shifts = grad->_W + link->iW +3*nNeurons;

    for (int n=0; n<nNeurons; n++)  {
        const Real dEdMean = link_inputs[n]-link_means[n];
        grad_means[n] += dEdMean;
        grad_vars[n] += dEdMean*dEdMean - link_vars[n];
        grad_scales[n] += inputs[n]*errors[n];
        grad_shifts[n] += errors[n];
    }
}
