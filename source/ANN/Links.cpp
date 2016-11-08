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

void WhiteningLink::propagate(const Activation* const netFrom, const Activation* const netTo,
																	const Real* const weights) const
{
	die("You really should not be able to get here.\n");
}

void WhiteningLink::backPropagate(const Activation* const netFrom, const Activation* const netTo,
											const Real* const weights, const Real* const gradW) const
{
	die("You really should not be able to get here as well.\n");
}

void Link::propagate(const Activation* const netFrom, const Activation* const netTo,
																	const Real* const weights) const
{
	const Real* const link_input = netFrom->outvals + iI;
	Real* const link_outputs = netTo->in_vals + iO;
	const Real* const link_weights =  weights + iW;

	for (int i = 0; i < nI; i++)
	for (int o = 0; o < nO; o++) {
		assert(nI*i+o>=0 && nI*i+o<nW);
		link_outputs[o] += link_input[i] * link_weights[nO*i + o];
	}
}

void Link::backPropagate(const Activation* const netFrom, const Activation* const netTo,
											const Real* const weights, const Real* const gradW) const
{
	const Real* const deltas = netTo->errvals + iO;
	Real* const link_errors = netFrom->errvals + iI;
	const Real* const layer_input = netFrom->outvals + iI;
	const Real* const link_weights = weights + iW;
	Real* const link_dEdW = gradW + iW;

	for (int i = 0; i < nI; i++)
	for (int o = 0; o < nO; o++) {
		assert(nO*i+o>=0 && nO*i+o<nW);
		link_dEdW[nO*i+o] += layer_input[i] * deltas[o];
		link_errors[i] += deltas[o] * link_weights[nO*i+o];
	}
}

void LinkToLSTM::propagate(const Activation* const netFrom, const Activation* const netTo,
																	const Real* const weights) const
{
	Real* const inputs = netTo->in_vals + iO;
	Real* const inputI = netTo->iIGates + iC;
	Real* const inputF = netTo->iFGates + iC;
	Real* const inputO = netTo->iOGates + iC;
	const Real* const weights_toCell = weights + iW;
	const Real* const weights_toIgate = weights + iWI;
	const Real* const weights_toFgate = weights + iWF;
	const Real* const weights_toOgate = weights + iWO;
	const Real* const link_input = netFrom->outvals + iI;

	for (int i = 0; i < nI; i++)
	for (int o = 0; o < nO; o++) {
		assert(nO*i+o>=0 && nO*i+o<nW);
		inputs[o] += link_input[i] * weights_toCell[nO*i + o];
		inputI[o] += link_input[i] * weights_toIgate[nO*i + o];
		inputF[o] += link_input[i] * weights_toFgate[nO*i + o];
		inputO[o] += link_input[i] * weights_toOgate[nO*i + o];
	}
}

void LinkToLSTM::backPropagate(const Activation* const netFrom, const Activation* const netTo,
											const Real* const weights, const Real* const gradW) const
{
	const Real* const deltaI = netTo->eIGates +iC;
	const Real* const deltaF = netTo->eFGates +iC;
	const Real* const deltaO = netTo->eOGates +iC;
	const Real* const deltaC = netTo->eMCell +iC;
	const Real* const layer_input = netFrom->outvals + iI;
	Real* const link_errors = netFrom->errvals + iI;
	const Real* const w_toOgate = weights + iWO;
	const Real* const w_toFgate = weights + iWF;
	const Real* const w_toIgate = weights + iWI;
	const Real* const w_toCell = weights + iW;
	Real* const dw_toOgate = gradW + iWO;
	Real* const dw_toFgate = gradW + iWF;
	Real* const dw_toIgate = gradW + iWI;
	Real* const dw_toCell = gradW + iW;

	for (int i = 0; i < nI; i++)
	for (int o = 0; o < nO; o++) {
		const int cc = nO*i + o;
		assert(cc>=0 && cc<nW);
		dw_toOgate[cc] += layer_input[i] * deltaO[o];
		dw_toCell[cc]  += layer_input[i] * deltaC[o];
		dw_toIgate[cc] += layer_input[i] * deltaI[o];
		dw_toFgate[cc] += layer_input[i] * deltaF[o];
		link_errors[i] += deltaO[o] * w_toOgate[cc] + deltaC[o] * w_toCell[cc] +
							deltaI[o] * w_toIgate[cc] + deltaF[o] * w_toFgate[cc];
	}
}

void LinkToConv2D::propagate(const Activation* const netFrom, const Activation* const netTo,
																	const Real* const weights) const
{
	Real* const link_outputs = netTo->in_vals + iO;
	const Real* const link_inputs = netFrom->outvals + iI;
	const Real* const link_weights = weights + iW;

	for(int ox=0; ox<outputWidth;  ox++)
	for(int oy=0; oy<outputHeight; oy++) {
		const int ix = ox*strideX - padX;
		const int iy = oy*strideY - padY;
		for(int fx=0; fx<filterWidth; fx++)
		for(int fy=0; fy<filterHeight; fy++) {
			const int cx(ix+fx), cy(iy+fy);
			//padding: skip addition if outside input boundaries
			if (cx < 0 || cy < 0 || cx >= inputWidth | cy >= inputHeight) continue;

			for(int iz=0; iz<inputDepth; iz++)
			for(int fz=0; fz<outputDepth; fz++) {
				const int pinp = iz +inputDepth*(cy +inputHeight*cx);
				const int pout = fz +outputDepth*(oy +outputHeight*ox);
				const int fid = fz +outputDepth*(iz +inputDepth*(fy +filterHeight*fx));
				assert(pout>=0 && pout<nO && pinp>=0 && pinp<nI && fid>=0 && fid<nW);
				link_outputs[pout] += link_inputs[pinp] * link_weights[fid];
			}
		}
	}
}

void LinkToConv2D::backPropagate(const Activation* const netFrom, const Activation* const netTo,
											const Real* const weights, const Real* const gradW) const
{
	const Real* const link_inputs = netFrom->outvals + iI;
	const Real* const deltas = netTo->errvals + iO;
	const Real* const link_weights = weights + iW;
	Real* const link_errors = netFrom->errvals + iI;
	Real* const link_dEdW = gradW + iW;

	for(int ox=0; ox<outputWidth;  ox++)
	for(int oy=0; oy<outputHeight; oy++) {
		const int ix = ox*strideX - padX;
		const int iy = oy*strideY - padY;
		for(int fx=0; fx<filterWidth; fx++)
		for(int fy=0; fy<filterHeight; fy++) {
			const int cx(ix+fx), cy(iy+fy);
			//padding: skip addition if outside input boundaries
			if (cx < 0 || cy < 0 || cx >= inputWidth | cy >= inputHeight) continue;

			for(int iz=0; iz<inputDepth; iz++)
			for(int fz=0; fz<outputDepth; fz++) {
				const int pinp = iz+inputDepth*(cy+inputHeight*cx);
				const int pout = fz+outputDepth*(oy+outputHeight*ox);
				const int fid = fz+outputDepth*(iz+inputDepth*(fy+filterHeight*fx));
				assert(pout>=0 && pout<nO && pinp>=0 && pinp<nI && fid>=0 && fid<nW);
				link_errors[pinp] += link_weights[fid]*deltas[pout];
				link_dEdW[fid] += link_inputs[pinp]*deltas[pout];
			}
		}
	}
}

void Link::orthogonalize(const int n0, Real* const _weights, const int nOut, const int nIn) const
{
	if (nIn<nOut) return;
	for (int i=1; i<nOut; i++) {
		Real v_d_v_pre = 0.;
		for (int k=0; k<nIn; k++)
			v_d_v_pre += *(_weights+n0+k*nOut+i)* *(_weights+n0+k*nOut+i);
		assert(v_d_v_pre>std::numeric_limits<Real>::epsilon());

		for (int j=0; j<i;  j++) {
			Real u_d_u = 0.0;
			Real v_d_u = 0.0;
			for (int k=0; k<nIn; k++) {
				u_d_u += *(_weights+n0+k*nOut+j)* *(_weights+n0+k*nOut+j);
				v_d_u += *(_weights+n0+k*nOut+j)* *(_weights+n0+k*nOut+i);
			}
			assert(u_d_u>std::numeric_limits<Real>::epsilon());
			for (int k=0; k<nIn; k++)
				*(_weights+n0+k*nOut+i) -= v_d_u/u_d_u * *(_weights+n0+k*nOut+j);
		}

		Real v_d_v_post = 0.0;
		for (int k=0; k<nIn; k++)
			v_d_v_post += *(_weights+n0+k*nOut+i)* *(_weights+n0+k*nOut+i);
		assert(v_d_v_post>std::numeric_limits<Real>::epsilon());
		for (int k=0; k<nIn; k++)
			*(_weights+n0+k*nOut+i) *= std::sqrt(v_d_v_pre/v_d_v_post);
	}
}

void Link::print() const
{
	cout << nI << " " << iI << " " << nO << " " << iO << " " << iW << endl;
	fflush(0);
}

void LinkToLSTM::print() const
{
	cout << nI << " " << iI << " " << nO << " " << iO << " " << iW << " " << iC << " " << iWI << " " << iWF << " " << iWO << " " << endl;
	fflush(0);
}

void Link::initialize(mt19937* const gen, Real* const _weights) const
{
	const Real range = std::sqrt(6./(nO + nI));
	//uniform_real_distribution<Real> dis(-range,range);
	normal_distribution<Real> dis(0.,range);

	for (int w=iW ; w<(iW + nO*nI); w++)
		*(_weights +w) = dis(*gen);

	orthogonalize(iW, _weights);
}

void LinkToLSTM::initialize(mt19937* const gen, Real* const _weights) const
{
	const Real range = std::sqrt(6./(nO + nI));
	//uniform_real_distribution<Real> dis(-range,range);
	normal_distribution<Real> dis(0.,range);

	for (int w=iW ; w<(iW + nO*nI); w++)
		*(_weights +w) = dis(*gen);
	orthogonalize(iW, _weights);

	for (int w=iWI; w<(iWI+ nO*nI); w++)
		*(_weights +w) = dis(*gen);
	orthogonalize(iWI, _weights);

	for (int w=iWF; w<(iWF+ nO*nI); w++)
		*(_weights +w) = dis(*gen);
	orthogonalize(iWF, _weights);

	for (int w=iWO; w<(iWO+ nO*nI); w++)
		*(_weights +w) = dis(*gen);
	orthogonalize(iWO, _weights);
}

void LinkToConv2D::initialize(mt19937* const gen, Real* const _weights) const
{
	const Real range = std::sqrt(6./(nAdded+outputDepth));
	//uniform_real_distribution<Real> dis(-range,range);
	normal_distribution<Real> dis(0.,range);

	const int nAdded = filterWidth*filterHeight*inputDepth;
	assert(outputDepth*nAdded == nW);
	for (int w=iW ; w<(iW + outputDepth*nAdded); w++)
		*(_weights +w) = dis(*gen);

	orthogonalize(iW, _weights, outputDepth, nAdded);
}

void WhiteningLink::initialize(mt19937* const gen, Real* const _weights) const
{
	//set to 1 the temporary variance and scaling factor
	Real* const my_weights = _weights +iW;
	for (int p=0 ; p<4; p++)
	for (int o=0 ; o<nO; o++)
		my_weights[p*nO+o] = Real(1==p || 2==p);
}

void Link::save(std::ostringstream & o, Real* const _weights) const
{
	o << std::setprecision(10);

	for (int w=iW ; w<(iW + nO*nI); w++)
		o << *(_weights +w);
}

void LinkToLSTM::save(std::ostringstream & o, Real* const _weights) const
{
	o << std::setprecision(10);

	for (int w=iW ; w<(iW + nO*nI); w++)
		o << *(_weights +w);

	for (int w=iWI; w<(iWI + nO*nI); w++)
		o << *(_weights +w);

	for (int w=iWF; w<(iWF + nO*nI); w++)
		o << *(_weights +w);

	for (int w=iWO; w<(iWO + nO*nI); w++)
		o << *(_weights +w);
}

void LinkToConv2D::save(std::ostringstream & o, Real* const _weights) const
{
	o << std::setprecision(10);

	const int nAdded = filterWidth*filterHeight*inputDepth;
	for (int w=iW ; w<(iW + outputDepth*nAdded); w++)
		o << *(_weights +w);
}

void WhiteningLink::save(std::ostringstream & o, Real* const _weights) const
{
	o << std::setprecision(10);

	for (int w=iW ; w<(iW + nO*4); w++)
		o << *(_weights +w);
}

void Link::restart(std::istringstream & buf, Real* const _weights) const
{
	for (int w=iW ; w<(iW + nO*nI); w++) {
		Real tmp;
		buf >> tmp;
		assert(not std::isnan(tmp) & not std::isinf(tmp));
		*(_weights +w) = tmp;
	}
}

void LinkToLSTM::restart(std::istringstream & buf, Real* const _weights) const
{
	Real tmp;
	for (int w=iW ; w<(iW + nO*nI); w++) {
		buf >> tmp;
		assert(not std::isnan(tmp) & not std::isinf(tmp));
		*(_weights +w) = tmp;
	}
	for (int w=iWI; w<(iWI + nO*nI); w++) {
		buf >> tmp;
		assert(not std::isnan(tmp) & not std::isinf(tmp));
		*(_weights +w) = tmp;
	}
	for (int w=iWF; w<(iWF + nO*nI); w++) {
		buf >> tmp;
		assert(not std::isnan(tmp) & not std::isinf(tmp));
		*(_weights +w) = tmp;
	}
	for (int w=iWO; w<(iWO + nO*nI); w++) {
		buf >> tmp;
		assert(not std::isnan(tmp) & not std::isinf(tmp));
		*(_weights +w) = tmp;
	}
}

void LinkToConv2D::restart(std::istringstream & buf, Real* const _weights) const
{
	const int nAdded = filterWidth*filterHeight*inputDepth;
	for (int w=iW ; w<(iW + outputDepth*nAdded); w++) {
		Real tmp;
		buf >> tmp;
		assert(not std::isnan(tmp) & not std::isinf(tmp));
		*(_weights +w) = tmp;
	}
}

void WhiteningLink::restart(std::istringstream & buf, Real* const _weights) const
{
	for (int w=iW ; w<(iW + nO*4); w++) {
		Real tmp;
		buf >> tmp;
		assert(not std::isnan(tmp) & not std::isinf(tmp));
		*(_weights +w) = tmp;
	}
}

void Graph::initializeWeights(mt19937* const gen, Real* const _weights, Real* const _biases) const
{
	uniform_real_distribution<Real> dis(-sqrt(6./layerSize),sqrt(6./layerSize));

	for (const auto & l : *(input_links_vec))
		l->initialize(gen, _weights);

	if(recurrent_link not_eq nullptr)
		recurrent_link->initialize(gen, _weights);

	if (not output) //let's try not having bias on output layer
		for (int w=firstBias_ID; w<firstBias_ID+layerSize; w++)
			*(_biases +w) = dis(*gen);

	if (LSTM) { //let all gates be biased towards open: better backprop
		for (int w=firstBiasIG_ID; w<firstBiasIG_ID+layerSize; w++)
			*(_biases +w) = dis(*gen) + 0.5;

		for (int w=firstBiasFG_ID; w<firstBiasFG_ID+layerSize; w++)
			*(_biases +w) = dis(*gen) + 0.5;

		for (int w=firstBiasOG_ID; w<firstBiasOG_ID+layerSize; w++)
			*(_biases +w) = dis(*gen) + 0.5;
	}
}
