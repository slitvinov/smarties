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

void NormalLayer::propagate(
		const Activation* const prev, Activation* const curr,
		const Real* const weights, const Real* const biases) const
{
	Real* const outputs = curr->outvals +n1stNeuron;
	Real* const inputs = curr->in_vals +n1stNeuron;
	for (int n=0; n<nNeurons; n++)
			inputs[n] = *(biases +n1stBias +n);

	for (const auto & link : *input_links) {
		const Real* const link_input = curr->outvals + link->iI;
		const Real* const link_weights =  weights + link->iW;
		const int n_inputs = link->nI;

		for (int i = 0; i < n_inputs; i++)
			for (int o = 0; o < nNeurons; o++)
				inputs[o] += link_input[i] * link_weights[n_inputs*i + o];
	}

	if(recurrent_link not_eq nullptr && prev not_eq nullptr) {
		const Real* const link_input = prev->outvals + recurrent_link->iI;
		const Real* const link_weights =  weights + recurrent_link->iW;
		const int n_inputs = recurrent_link->nI;

		for (int i = 0; i < n_inputs; i++)
			for (int o = 0; o < nNeurons; o++)
				inputs[o] += link_input[i] * link_weights[n_inputs*i + o];
	}

	for (int o=0; o<nNeurons; o++)
		outputs[o] = func->eval(inputs[o]);
}

void LSTMLayer::propagate(
		const Activation* const prev, const Activation* const curr,
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

	for (const auto & link : *input_links) {
		const Real* const link_input = curr->outvals + link->iI;
		const Real* const weights_toIgate = weights +link->iWI;
		const Real* const weights_toFgate = weights +link->iWF;
		const Real* const weights_toOgate = weights +link->iWO;
		const Real* const weights_toCell = weights +link->iW;
		const int n_inputs = link->nI;

		for (int i = 0; i < n_inputs; i++)
		for (int o = 0; o < nNeurons; o++) {
			const int ID_link = n_inputs*i + o;
			inputs[o] += link_input[i] * weights_toCell[ID_link];
			inputI[o] += link_input[i] * weights_toIgate[ID_link];
			inputF[o] += link_input[i] * weights_toFgate[ID_link];
			inputO[o] += link_input[i] * weights_toOgate[ID_link];
		}
	}

	if(recurrent_link not_eq nullptr && prev not_eq nullptr) {
		const Real* const link_input = prev->outvals + recurrent_link->iI;
		const Real* const weights_toIgate = weights +recurrent_link->iWI;
		const Real* const weights_toFgate = weights +recurrent_link->iWF;
		const Real* const weights_toOgate = weights +recurrent_link->iWO;
		const Real* const weights_toCell = weights +recurrent_link->iW;
		const int n_inputs = recurrent_link->nI;

		for (int i = 0; i < n_inputs; i++)
		for (int o = 0; o < nNeurons; o++) {
			const int ID_link = n_inputs*i + o;
			inputs[o] += link_input[i] * weights_toCell[ID_link];
			inputI[o] += link_input[i] * weights_toIgate[ID_link];
			inputF[o] += link_input[i] * weights_toFgate[ID_link];
			inputO[o] += link_input[i] * weights_toOgate[ID_link];
		}
	}

    for (int o=0; o<nNeurons; o++) {
    	outputC[o] = ifun->eval(inputs[o]);
    	outputI[o] = sigm->eval(inputI[o]);
    	outputF[o] = sigm->eval(inputF[o]);
    	outputO[o] = sigm->eval(inputO[o]);

        const Real oS = outputC[o] * outputI[o] +
        		prev==nullptr ?  0 : *(prev->ostates +n1stCell +n) * outputF[o];
        *(curr->ostates +n1stCell   +o) = oS;
        *(curr->outvals +n1stNeuron +o) = func->eval(oS) * outputO[o];
    }
}

void Conv2D::propagate(
		const Activation* const prev, const Activation* const curr,
		const Real* const weights, const Real* const biases) const
{
	Real* const inputs  = curr->in_vals +n1stNeuron;
	Real* const outputs = curr->outvals +n1stNeuron;

	for(int ox=0; ox<outputWidth;  ox++)
	for(int oy=0; oy<outputHeight; oy++)
	for(int fz=0; fz<outputDepth; fz++) {
		const int idp = fz +outputDepth*(oy +outputHeight*ox);
		inputs[idp] = *(biases +n1stBias +idp);
	}

	for (const auto & link : *input_links) {
		const Real* const link_inputs = curr->outvals + link->iI;
		const Real* const link_weights = weights + link->iW;
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
					const int idp = fz +outputDepth*(oy +outputHeight*ox);
					assert(idp<nNeurons && iz+iD*(cy+iH*cx) < link->nI);
					inputs[idp] += link_inputs[iz +iD*(cy +iH*cx)] *
							link_weights[fz +outputDepth*(iz +iD*(fy +fH*fx))];
				}
			}
		}
	}

	for(int ox=0; ox<outputWidth;  ox++)
	for(int oy=0; oy<outputHeight; oy++)
	for(int fz=0; fz<outputDepth; fz++) {
		const int idp = fz +outputDepth*(oy +outputHeight*ox);
		outputs[idp] = func->eval(inputs[idp]);
	}
}

void NormalLayer::backPropagateDelta(
		Activation* const prev, Activation* const curr, Activation* const next,
		const Real* const weights, const Real* const biases) const
{
	const Real* const inputs = curr->in_vals +n1stNeuron;
	Real* const errors = curr->errvals +n1stNeuron;
	for (int o=0; o<nNeurons; o++)
		errors[o] *= func->evalDiff(inputs[o]);

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
		Activation* const prev, Activation* const curr, Activation* const next,
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

	for (int o=0; o<nNeurons; o++) {
		deltaO[o] = deltas[o] * sigm->evalDiff(inputO[o]) * func->eval(*(curr->ostates+n1stCell+o));

		deltas[o] = deltas[o] * outputO[o] * func->evalDiff(*(curr->ostates +n1stCell +o)) +
					next==nullptr ?  0 : *(next->errvals+n1stNeuron+o)* *(next->oFGates+n1stCell+o);

		deltaC[o] = ifun->evalDiff(inputs[o]) * outputI[o] * deltas[o];
		deltaI[o] = sigm->evalDiff(inputI[o]) * outputC[o] * deltas[o];
		deltaF[o] = prev==nullptr ?  0 : sigm->evalDiff(inputF[o])* *(prev->ostates+n1stCell+o)*deltas[o];
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
		Activation* const prev, Activation* const curr, Activation* const next, 
		const Real* const weights, const Real* const biases) const
{
	const Real* const inputs = curr->in_vals +n1stNeuron;
	const Real* const errors = curr->errvals +n1stNeuron;

	for(int ox=0; ox<outputWidth;  ox++)
	for(int oy=0; oy<outputHeight; oy++)
	for(int fz=0; fz<outputDepth; fz++) {
		const int idp = fz +outputDepth*(oy +outputHeight*ox);
		errors[idp] *= func->evalDiff(inputs[idp]);
	}

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

void NormalLayer::backPropagateGrads(
		const Activation* const prev, const Activation* const curr,
		Grads* const grad, const Real* const weights) const
{
	const Real* const deltas = curr->errvals +n1stNeuron;
    for (int n=0; n<nNeurons; n++)  //grad bias == delta
        *(grad->_B +n1stBias +n) = deltas[n];

    for (const auto & link : *input_links) {
    	const Real* const layer_input = curr->outvals +link->iI;
		Real* const link_dEdW = grad->_W +link->iW;
		const int n_inputs = link->nI;

		for (int i = 0; i < n_inputs; i++)
			for (int o = 0; o < nNeurons; o++)
				link_dEdW[n_inputs*i +o] = layer_input[i] * deltas[o];
    }

    if(recurrent_link not_eq nullptr && prev not_eq nullptr) {
    	const Real* const layer_input = prev->outvals +recurrent_link->iI;
		Real* const link_dEdW = grad->_W +recurrent_link->iW;
		const int n_inputs = recurrent_link->nI;

		for (int i = 0; i < n_inputs; i++)
			for (int o = 0; o < nNeurons; o++)
				link_dEdW[n_inputs*i +o] = layer_input[i] * deltas[o];
    }
}

void NormalLayer::backPropagateAddGrads(
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
	const Real* const dSdInputFGate = curr->eFGates +n1stCell;
	const Real* const dSdInputIGate = curr->eIGates +n1stCell;
	const Real* const dSdInputCell = curr->eMCell +n1stCell;

    for (int n=0; n<nNeurons; n++)  { //grad bias == delta
        *(grad->_B +n1stBias   +n) = dSdInputCell[n];
		*(grad->_B +n1stBiasIG +n) = dSdInputIGate[n];
		*(grad->_B +n1stBiasFG +n) = dSdInputFGate[n];
		*(grad->_B +n1stBiasOG +n) = dEdInputOGate[n];
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
			dw_toOgate[cc] = layer_input[i] * dEdInputOGate[o];
			dw_toCell[cc]  = layer_input[i] * dSdInputCell[o];
			dw_toIgate[cc] = layer_input[i] * dSdInputIGate[o];
			dw_toFgate[cc] = layer_input[i] * dSdInputFGate[o];
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
			dw_toOgate[cc] = layer_input[i] * dEdInputOGate[o];
			dw_toCell[cc]  = layer_input[i] * dSdInputCell[o];
			dw_toIgate[cc] = layer_input[i] * dSdInputIGate[o];
			dw_toFgate[cc] = layer_input[i] * dSdInputFGate[o];
		}
    }
}

void LSTMLayer::backPropagateAddGrads(
		const Activation* const prev, const Activation* const curr,
		Grads* const grad, const Real* const weights) const
{
	const Real* const dEdInputOGate = curr->eOGates +n1stCell;
	const Real* const dSdInputFGate = curr->eFGates +n1stCell;
	const Real* const dSdInputIGate = curr->eIGates +n1stCell;
	const Real* const dSdInputCell = curr->eMCell +n1stCell;

	for (int n=0; n<nNeurons; n++)  { //grad bias == delta
		*(grad->_B +n1stBias   +n) += dSdInputCell[n] ;
		*(grad->_B +n1stBiasIG +n) += dSdInputIGate[n];
		*(grad->_B +n1stBiasFG +n) += dSdInputFGate[n];
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
			dw_toCell[cc]  += layer_input[i] * dSdInputCell[o] ;
			dw_toIgate[cc] += layer_input[i] * dSdInputIGate[o];
			dw_toFgate[cc] += layer_input[i] * dSdInputFGate[o];
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
			dw_toCell[cc]  += layer_input[i] * dSdInputCell[o] ;
			dw_toIgate[cc] += layer_input[i] * dSdInputIGate[o];
			dw_toFgate[cc] += layer_input[i] * dSdInputFGate[o];
		}
	}
}

void WhiteningLayer::propagate(Activation* const N, const Real* const weights, const Real* const biases) const
{
	const Link* const iL = (*input_links)[0]; //only one as input
	assert(input_links->size() == 1);
    for (int n=0; n<nNeurons; n++) {
        Real input[2];
        const Real* const output_LayerFrom = lab->outvals +iI;
    	const Real* const my_weights =  weights +iW;
    	// 4 parameters per neuron:
    	const Real mean  = my_weights[ID_NeuronTo*4];
    	const Real var   = my_weights[ID_NeuronTo*4+1];
    	const Real scale = my_weights[ID_NeuronTo*4+2];
    	const Real shift = my_weights[ID_NeuronTo*4+3];
    	assert(var>std::numeric_limits<double>::epsilon());
    	//this is X hat
    	inputs[0] = (output_LayerFrom[ID_NeuronTo] - mean)/std::sqrt(var);
    	//this is y
    	inputs[1] =  scale*inputs[0] + shift;
        *(N->in_vals +n1stNeuron +n) = input[0]; //xhat
        *(N->outvals +n1stNeuron +n) = input[1]; //y
    }
}

void WhiteningLayer::backPropagateDelta(Activation* const C, const Real* const weights, const Real* const biases) const
{
    /*
     * function should prepare dE/dx to be read from non-scaled layer
     * TODO: how can I easily implement multivariate version?
     */
	const Link* const iL = (*input_links)[0]; //only one as input
    const Real* const my_weights = weights + iL->iW;

    for (int n=0; n<nNeurons; n++) {
        Real dEdy = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
        for (const auto & link : *output_links) {
        	const Real* const my_weights =  weights +iW;
			const Real* const dEdInput_LayerTo = lab->errvals +iO;
			const Real invvar= 1./std::sqrt(my_weights[ID_NeuronFrom*4+1]);
			const Real scale = my_weights[ID_NeuronFrom*4+2];
			return dEdInput_LayerTo[ID_NeuronFrom]*scale*invvar;
        }
        *(C->errvals +n1stNeuron +n) = dEdy;
    }
}

void WhiteningLayer::backPropagateGrads(const Activation* const C, Grads* const grad, const Real* const weights) const
{
	const Link* const iL = (*input_links)[0]; //only one as input
	const Real* const x = C->outvals + iL->iI;
    const Real* const my_weights = weights + iL->iW;

    for (int n=0; n<nNeurons; n++)  {
        const Real mean  = my_weights[n*4];
        const Real var   = my_weights[n*4+1];

        const Real dEdy = *(C->errvals +n1stNeuron +n);
        const Real dEdScale = *(C->in_vals +n1stNeuron +n)*dEdy;
        const Real dEdShift = dEdy;
        const Real dEdMean = x[n]-mean;
        const Real dEdVar = dEdMean*dEdMean - var;

        *(grad->_W + iL->iW +n*4   ) = dEdMean;
        *(grad->_W + iL->iW +n*4 +1) = dEdVar;
        *(grad->_W + iL->iW +n*4 +2) = dEdScale;
        *(grad->_W + iL->iW +n*4 +3) = dEdShift;
    }
}
