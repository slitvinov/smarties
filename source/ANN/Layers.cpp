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

void NormalLayer::propagate(Activation* const N, const Real* const weights, const Real* const biases) const
{
	// prop: weights SIMD over features to avoid simd reduce on popagate: speed
	// means broadcast one input, multiply with wghts, obtain a vector to be added to features
	// backprop then requires reduce: multiply w and dEdI_next and reduce vector to get dEdy
	Real* const inputs  = N->in_vals +n1stNeuron;
	const Real* const biases = biases +n1stBias;
	Real* const outputs = N->outvals +n1stNeuron;

	std::copy_n(inputs, nNeurons, biases);

	for (const auto & link : *input_links) {
		/*
		 each link connects one layer to an other
		 multiple links in input_links vector means that a layer
		 can be connected to other layers anywhere in the net
		 (eg. might want to have each layer linked to two previous layers)
		 */
		const Real* const output_LayerFrom = N->outvals + link->iI;
		const Real* const link_weights =  weights + link->iW;
		const int const link_n_inputs = link->nI;

		for (int i = 0; i < link_n_inputs; i++)
			for (int o = 0; o < nNeurons; o++)
				inputs[o] += output_LayerFrom[i] * link_weights[link_n_inputs*i + o];
	}

	for (int o = 0; o < nNeurons; o++)
		outputs[o] = func->eval(inputs[o]);
    }
}

void Conv2D::propagate(Activation* const N, const Real* const weights, const Real* const biases) const
{
	Real* const inputs  = N->in_vals +n1stNeuron;
	Real* const outputs = N->outvals +n1stNeuron;

	for (const auto & link : *input_links) {
		const Real* const link_inputs = N->outvals + link->iI;
		const Real* const link_weights = weights + link->iW;
		const Real* const link_biases = biases + link->iB;

		const int iW(link->inputWidth), iH(link->inputHeight);
		const int iD(link->inputDepth), fW(link->width), fH(link->height);

		for(int ox=0; ox<outputWidth;  ox++)
		for(int oy=0; oy<outputHeight; oy++) {
			Real tmp[fN];
			const int ix = ox*link->strideX - link->padX;
			const int iy = oy*link->strideY - link->padY;

			for(int fx=0; fx<fW; fx++)
			for(int fy=0; fy<fH; fy++) {
				const int cx = ix+fx;
				const int cy = iy+fy;
				//padding: skip addition if outside input boundaries
				if (cx < 0 || cy < 0 || cx >= iW | cy >= iH) continue;

				for(int iz=0; iz<iD; iz++)
				for(int fz=0; fz<outputDepth; fz++) {
					assert(iz +iD*(cy +iH*cx) < link->nI);
					tmp[fz] += link_weights[fz +outputDepth*(fy +fH*fx)]*
							    link_inputs[iz +iD*(cy +iH*cx)];
				}
			}

			for(int fz=0; fz<outputDepth; fz++) {
				assert(fz +outputDepth*(oy +inputWidth*ox) < nNeurons);
				inputs[fz +outputDepth*(oy +inputWidth*ox)] += tmp[fz];
			}
		}

		for(int ox=0; ox<outputWidth;  ox++)
		for(int oy=0; oy<outputHeight; oy++)
		for(int fz=0; fz<outputDepth; fz++) {
			const int idp = fz +fN*(oy +inputWidth*ox);
			inputs[idp] += link_biases[fz];
			outputs[idp] = func->eval(inputs[idp]);
		}
	}
}

void Conv2D::backPropagateDelta(Activation* const N, const Real* const weights, const Real* const biases) const
{
	const Real* const inputs  = N->in_vals +n1stNeuron;
	const Real* const outputs = N->outvals +n1stNeuron;
	const Real* const errors  = N->errvals +n1stNeuron;

	for(int ox=0; ox<outputWidth;  ox++)
	for(int oy=0; oy<outputHeight; oy++)
	for(int fz=0; fz<outputDepth; fz++) {
		const int idp = fz +outputDepth*(oy +inputWidth*ox);
		errors[idp] *= func->evalDiff(inputs[idp]);
	}

	for (const auto & link : *input_links) {
		const Real* const link_inputs = N->outvals + link->iI;
		const Real* const link_weights = weights + link->iW;
		const Real* const link_biases = biases + link->iB;
		Real* const link_errors  = N->errvals + link->iI;

		const int iW(link->inputWidth), iH(link->inputHeight);
		const int iD(link->inputDepth), fW(link->width), fH(link->height);

		for(int ox=0; ox<outputWidth;  ox++)
		for(int oy=0; oy<outputHeight; oy++) {
			const int ix = ox*link->strideX - link->padX;
			const int iy = oy*link->strideY - link->padY;

			for(int fx=0; fx<fW; fx++)
			for(int fy=0; fy<fH; fy++) {
				const int cx = ix+fx;
				const int cy = iy+fy;
				//padding: skip addition if outside input boundaries
				if (cx < 0 || cy < 0 || cx >= iW | cy >= iH) continue;

				for(int iz=0; iz<iD; iz++)
				for(int fz=0; fz<outputDepth; fz++) {
					assert(iz +iD*(cy +iH*cx) < link->nI);
					const int pinp = iz +iD*(cy +iH*cx);
					const int pout = fz +outputDepth*(oy +inputWidth*ox);
					const int fid = fz +outputDepth*(fy +fH*fx);
					link_errors[pinp] += link_weights[fid]*errors[pout];
				}
			}
		}
	}
}

void WhiteningLayer::propagate(Activation* const N, const Real* const weights, const Real* const biases) const
{
	const Link* const iL = (*input_links)[0]; //only one as input
	assert(input_links.size() == 1);
    for (int n=0; n<nNeurons; n++) {
        Real input[2];
        iL->propagate(input,N,n,weights);
        *(N->in_vals +n1stNeuron +n) = input[0]; //xhat
        *(N->outvals +n1stNeuron +n) = input[1]; //y
    }
}

void NormalLayer::backPropagateDelta_reverse(Activation* const C, const Real* const weights, const Real* const biases) const
{
    for (int n=0; n<nNeurons; n++) {
        //if this an output neuron, the error is written from outside in the corresponding errvals, else zero
        Real dEdy = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;

        for (const auto & link : *output_links) //loop over all layers to which this layer is connected to
        	dEdy += link->backPropagate(C, n, weights);

        //delta_i = f'(input_i) * sum_(neurons j) ( error_j * w_i_j)
        const Real input = *(C->in_vals +n1stNeuron+n);
        *(C->errvals +n1stNeuron +n) = dEdy * func->evalDiff(input);
    }
}

void NormalLayer::backPropagateDelta(Activation* const C, const Real* const weights, const Real* const biases) const
{
    for (int n=0; n<nNeurons; n++) {
        //if this an output neuron, the error is written from outside in the corresponding errvals, else zero
        Real dEdy = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;

        for (const auto & link : *output_links) //loop over all layers to which this layer is connected to
        	dEdy += link->backPropagate(C, n, weights);

        //delta_i = f'(input_i) * sum_(neurons j) ( error_j * w_i_j)
        const Real input = *(C->in_vals +n1stNeuron+n);
        *(C->errvals +n1stNeuron +n) = dEdy * func->evalDiff(input);
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
        for (const auto & link : *output_links)
        	dEdy += link->backPropagate(C, n, weights);
        *(C->errvals +n1stNeuron +n) = dEdy;
    }
}

void NormalLayer::backPropagateGrads(const Activation* const C, Grads* const grad, const Real* const weights) const
{
    for (int n=0; n<nNeurons; n++)  //grad bias == delta
        *(grad->_B +n1stBias +n) = *(C->errvals +n1stNeuron +n);

    for (const auto & link : *input_links)
    	link->computeGrad(C, C, grad->_W);
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

void NormalLayer::backPropagateAddGrads(const Activation* const C, Grads* const grad, const Real* const weights) const
{
	for (int n=0; n<nNeurons; n++)  //grad bias == delta
		*(grad->_B +n1stBias +n) += *(C->errvals +n1stNeuron +n);

	for (const auto & link : *input_links)
		link->addUpGrads(C, C, grad->_W);
}

void WhiteningLayer::backPropagateAddGrads(const Activation* const C, Grads* const grad, const Real* const weights) const
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

    	*(grad->_W + iL->iW +n*4   ) += dEdMean;
    	*(grad->_W + iL->iW +n*4 +1) += dEdVar;
    	*(grad->_W + iL->iW +n*4 +2) += dEdScale;
    	*(grad->_W + iL->iW +n*4 +3) += dEdShift;
    }
}

//all of the following is for recurrent neural networks

void NormalLayer::propagate(const Activation* const M, Activation* const N, const Real* const weights, const Real* const biases) const
{
    for (int n=0; n<nNeurons; n++) {
        Real input = 0.;
        
        for (const auto & link : *input_links)
            input += link->propagate(N,n,weights);

        if(recurrent_link not_eq nullptr)
        	input += recurrent_link->propagate(M,n,weights);

        input += *(biases +n1stBias +n);
        *(N->in_vals +n1stNeuron +n) = input;
        *(N->outvals +n1stNeuron +n) = func->eval( input );
    }
}

void WhiteningLayer::propagate(const Activation* const M, Activation* const N, const Real* const weights, const Real* const biases) const
{
	propagate(N, weights, biases);
}

void LSTMLayer::propagate(Activation* const N, const Real* const weights, const Real* const biases) const
{
    for (int n=0; n<nNeurons; n++) {
    	Real inputs[4] = {0,0,0,0};

        for (const auto & link : *input_links)
            link->propagate(inputs,N,n,weights);

        inputs[0] += *(biases +n1stBias +n);
        *(N->in_vals +n1stNeuron +n) = inputs[0];
        *(N->oMCell  +n1stCell +n) = ifun->eval(inputs[0]);

        inputs[1] += *(biases +n1stBiasIG +n);
        *(N->iIGates +n1stCell +n) = inputs[1];
        *(N->oIGates +n1stCell +n) = sigm->eval(inputs[1]);
        
        inputs[2] += *(biases +n1stBiasFG +n);
        *(N->iFGates +n1stCell +n) = inputs[2];
        *(N->oFGates +n1stCell +n) = sigm->eval(inputs[2]);
        
        inputs[3] += *(biases +n1stBiasOG +n);
        *(N->iOGates +n1stCell +n) = inputs[3];
        *(N->oOGates +n1stCell +n) = sigm->eval(inputs[3]);
        
        const Real oS = *(N->oMCell  +n1stCell +n) * *(N->oIGates +n1stCell +n);
        *(N->ostates +n1stCell +n) = oS;
        *(N->outvals +n1stNeuron +n) = func->eval(oS) * *(N->oOGates +n1stCell +n);
    }
}

void LSTMLayer::propagate(const Activation* const M, Activation* const N, const Real* const weights, const Real* const biases) const
{
    for (int n=0; n<nNeurons; n++) {
    	Real inputs[4] = {0,0,0,0};
        for (const auto & link : *input_links)
            link->propagate(inputs,N,n,weights);

        recurrent_link->propagate(inputs,M,n,weights);

        inputs[0] += *(biases +n1stBias +n);
        *(N->in_vals +n1stNeuron +n) = inputs[0];
        *(N->oMCell  +n1stCell +n) = ifun->eval(inputs[0]);
        
        inputs[1] += *(biases +n1stBiasIG +n);
        *(N->iIGates +n1stCell +n) = inputs[1];
        *(N->oIGates +n1stCell +n) = sigm->eval(inputs[1]);
        
        inputs[2] += *(biases +n1stBiasFG +n);
        *(N->iFGates +n1stCell +n) = inputs[2];
        *(N->oFGates +n1stCell +n) = sigm->eval(inputs[2]);
        
        inputs[3] += *(biases +n1stBiasOG +n);
        *(N->iOGates +n1stCell +n) = inputs[3];
        *(N->oOGates +n1stCell +n) = sigm->eval(inputs[3]);
        
        const Real oS = *(N->oMCell  +n1stCell +n) * *(N->oIGates +n1stCell +n)
        			  + *(M->ostates +n1stCell +n) * *(N->oFGates +n1stCell +n);
        *(N->ostates +n1stCell +n) = oS;
        *(N->outvals +n1stNeuron +n) = func->eval(oS) * *(N->oOGates +n1stCell +n);
    }
}

void NormalLayer::backPropagateDeltaFirst(Activation* const C, const Activation* const N, const Real* const weights, const Real* const biases) const
{
    for (int n=0; n<nNeurons; n++) {
        Real dEdy = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
        for (const auto & link : *output_links) //loop over all layers to which this layer is connected to
        	dEdy += link->backPropagate(C, n, weights);

        if(recurrent_link not_eq nullptr)
        	dEdy += recurrent_link->backPropagate(N, n, weights);
        
        *(C->errvals +n1stNeuron +n) = dEdy * func->evalDiff(*(C->in_vals +n1stNeuron+n));
    }
}

void WhiteningLayer::backPropagateDeltaFirst(Activation* const C, const Activation* const N, const Real* const weights, const Real* const biases) const
{
	backPropagateDelta(C, weights, biases);
}

void LSTMLayer::backPropagateDeltaFirst(Activation* const C, const Activation* const N, const Real* const weights, const Real* const biases) const
{
    for (int n=0; n<nNeurons; n++) {
        Real dEdy = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
        for (const auto & link : *output_links) //loop over all layers to which this layer is connected to
        	dEdy += link->backPropagate(C, n, weights);

        dEdy += recurrent_link->backPropagate(N, n, weights);

        *(C->eMCell +n1stCell+n) =       ifun->evalDiff(*(C->in_vals+n1stNeuron+n)) * *(C->oIGates+n1stCell+n);
        *(C->eIGates+n1stCell+n) =       sigm->evalDiff(*(C->iIGates+n1stCell  +n)) * *(C->oMCell +n1stCell+n);
        *(C->eFGates+n1stCell+n) = 0.0;
        *(C->eOGates+n1stCell+n) = dEdy * sigm->evalDiff(*(C->iOGates+n1stCell  +n)) * func->eval(*(C->ostates+n1stCell+n));
        *(C->errvals+n1stNeuron+n) = dEdy * *(C->oOGates+n1stCell+n) * func->evalDiff(*(C->ostates +n1stCell +n)) + *(N->errvals+n1stNeuron+n)* *(N->oFGates+n1stCell+n);
    }
}

void LSTMLayer::backPropagateDelta(Activation* const C, const Real* const weights, const Real* const biases) const
{
    for (int n=0; n<nNeurons; n++) {
    	Real dEdy = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
    	for (const auto & link : *output_links) //loop over all layers to which this layer is connected to
    		dEdy += link->backPropagate(C, n, weights);

        *(C->eMCell +n1stCell+n) = ifun->evalDiff(*(C->in_vals+n1stNeuron+n)) * *(C->oIGates+n1stCell+n);
        *(C->eIGates+n1stCell+n) = sigm->evalDiff(*(C->iIGates+n1stCell  +n)) * *(C->oMCell +n1stCell+n);
        *(C->eFGates+n1stCell+n) = 0.0;
        *(C->eOGates+n1stCell+n) = dEdy * sigm->evalDiff(*(C->iOGates+n1stCell  +n)) * func->eval(*(C->ostates+n1stCell+n));
        *(C->errvals+n1stNeuron+n) = dEdy * *(C->oOGates+n1stCell+n) * func->evalDiff(*(C->ostates +n1stCell +n));
    }
}

void LSTMLayer::backPropagateDelta(const Activation* const P, Activation* const C, const Activation* const N, const Real* const weights, const Real* const biases) const
{
    for (int n=0; n<nNeurons; n++) {
        Real dEdy = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
        for (const auto & link : *output_links) //loop over all layers to which this layer is connected to
        	dEdy += link->backPropagate(C, n, weights);

        dEdy += recurrent_link->backPropagate(N, n, weights);

        *(C->eMCell +n1stCell+n) = ifun->evalDiff(*(C->in_vals+n1stNeuron+n)) * *(C->oIGates+n1stCell+n);
        *(C->eIGates+n1stCell+n) = sigm->evalDiff(*(C->iIGates+n1stCell  +n)) * *(C->oMCell +n1stCell+n);
        *(C->eFGates+n1stCell+n) = sigm->evalDiff(*(C->iFGates+n1stCell  +n)) * *(P->ostates+n1stCell+n);
        *(C->eOGates+n1stCell+n) = dEdy * sigm->evalDiff(*(C->iOGates+n1stCell  +n)) * func->eval(*(C->ostates+n1stCell+n));
        *(C->errvals+n1stNeuron+n) = dEdy * *(C->oOGates+n1stCell+n) * func->evalDiff(*(C->ostates +n1stCell +n)) + *(N->errvals+n1stNeuron+n)* *(N->oFGates+n1stCell+n);
    }
}

void LSTMLayer::backPropagateDeltaLast(const Activation* const P, Activation* const C, const Real* const weights, const Real* const biases) const
{
    for (int n=0; n<nNeurons; n++) {
    	Real dEdy = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
    	for (const auto & link : *output_links) //loop over all layers to which this layer is connected to
    		dEdy += link->backPropagate(C, n, weights);

        *(C->eMCell +n1stCell+n) = ifun->evalDiff(*(C->in_vals+n1stNeuron+n)) * *(C->oIGates+n1stCell+n);
        *(C->eIGates+n1stCell+n) = sigm->evalDiff(*(C->iIGates+n1stCell  +n)) * *(C->oMCell +n1stCell+n);
        *(C->eFGates+n1stCell+n) = sigm->evalDiff(*(C->iFGates+n1stCell  +n)) * *(P->ostates+n1stCell+n);
        *(C->eOGates+n1stCell+n) = dEdy * sigm->evalDiff(*(C->iOGates+n1stCell  +n)) * func->eval(*(C->ostates+n1stCell+n));
        *(C->errvals+n1stNeuron+n) = dEdy * *(C->oOGates+n1stCell+n) * func->evalDiff(*(C->ostates +n1stCell +n));
    }
}

void NormalLayer::backPropagateGrads(const Activation* const P, const Activation* const C, Grads* const grad, const Real* const weights) const
{
	backPropagateGrads(C, grad, weights);

    if(recurrent_link not_eq nullptr)
    	recurrent_link->computeGrad(P, C, grad->_W);
}

void WhiteningLayer::backPropagateGrads(const Activation* const P, const Activation* const C, Grads* const grad, const Real* const weights) const
{
	backPropagateGrads(C, grad, weights);
}

void LSTMLayer::backPropagateGrads(const Activation* const P, const Activation* const C, Grads* const grad, const Real* const weights) const
{
	backPropagateGrads(C, grad, weights);

    if(recurrent_link not_eq nullptr)
    	recurrent_link->computeGrad(P, C, grad->_W);
}

void LSTMLayer::backPropagateGrads(const Activation* const C, Grads* const grad, const Real* const weights) const
{
    for (int n=0; n<nNeurons; n++) {
    	*(grad->_B +n1stBias   +n) = *(C->eMCell  +n1stCell +n) * *(C->errvals +n1stNeuron +n);
    	*(grad->_B +n1stBiasIG +n) = *(C->eIGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
    	*(grad->_B +n1stBiasFG +n) = *(C->eFGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
    	*(grad->_B +n1stBiasOG +n) = *(C->eOGates +n1stCell +n);
    }

	for (const auto & link : *input_links)
		link->computeGrad(C, C, grad->_W);
}

void NormalLayer::backPropagateAddGrads(const Activation* const P, const Activation* const C, Grads* const grad, const Real* const weights) const
{
	backPropagateAddGrads(C, grad, weights);

    if(recurrent_link not_eq nullptr)
    	recurrent_link->addUpGrads(P, C, grad->_W);
}

void WhiteningLayer::backPropagateAddGrads(const Activation* const P, const Activation* const C, Grads* const grad, const Real* const weights) const
{
	backPropagateAddGrads(C, grad, weights);
}

void LSTMLayer::backPropagateAddGrads(const Activation* const P, const Activation* const C, Grads* const grad, const Real* const weights) const
{
	backPropagateAddGrads(C, grad, weights);

    if(recurrent_link not_eq nullptr)
    	recurrent_link->addUpGrads(P, C, grad->_W);
}

void LSTMLayer::backPropagateAddGrads(const Activation* const C, Grads* const grad, const Real* const weights) const
{
    for (int n=0; n<nNeurons; n++) {
    	*(grad->_B +n1stBias   +n) += *(C->eMCell  +n1stCell +n) * *(C->errvals +n1stNeuron +n);
    	*(grad->_B +n1stBiasIG +n) += *(C->eIGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
    	*(grad->_B +n1stBiasFG +n) += *(C->eFGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
    	*(grad->_B +n1stBiasOG +n) += *(C->eOGates +n1stCell +n);
    }

	for (const auto & link : *input_links)
		link->addUpGrads(C, C, grad->_W);
}
