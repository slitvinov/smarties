/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Activations.h"
#include <iostream>

class Link
{
 public:
	const Uint iW, nI, iI, nO, iO, nO_simd, nW;
	Link(Uint _nI, Uint _iI, Uint _nO, Uint _iO, Uint _iW, Uint _nO_simd, Uint _nW)
	: iW(_iW), nI(_nI), iI(_iI), nO(_nO), iO(_iO), nO_simd(_nO_simd), nW(_nW) {}

	virtual ~Link() {}
	virtual void print() const = 0;

	void _initialize(mt19937* const gen, Real* const _weights, const Real scale,
		Uint n0, Uint nOut, Uint nIn, Uint n_simd) const
	{
		uniform_real_distribution<Real> dis(-scale,scale);
		//normal_distribution<Real> dis(0.,range);
		for (Uint i = 0; i < nIn; i++)
			for (Uint o = 0; o < nOut; o++)
				_weights[n0 + n_simd*i + o] = dis(*gen);
		//orthogonalize(n0, _weights, nOut, nAdded, n_simd);
	}

	virtual void save(vector<Real>& out, Real* const _weights) const = 0;
	void _save(vector<Real>& out, Real*const _weights, Uint n0, Uint nOut, Uint nIn, Uint n_simd) const
	{
		for (Uint i = 0; i < nIn; i++) for (Uint o = 0; o < nOut; o++) {
				const Uint w = n0 + n_simd*i + o;
				out.push_back(_weights[w]);
				assert(!std::isnan(_weights[w]) && !std::isinf(_weights[w]));
			}
	}

	virtual void restart(vector<Real>& buf, Real* const _weights) const = 0;
	void _restart(vector<Real>& buf, Real*const _weights, Uint n0, Uint nOut, Uint nIn, Uint n_simd) const
	{
		for (Uint i = 0; i < nIn; i++) for (Uint o = 0; o < nOut; o++) {
				const Uint w = n0 + n_simd*i + o;
        _weights[w] = buf.front();
        buf.erase(buf.begin(),buf.begin()+1);
				assert(!std::isnan(_weights[w]) && !std::isinf(_weights[w]));
			}
	}

	void orthogonalize(const Uint n0, Real* const _weights, Uint nOut, Uint nIn, Uint n_simd) const
	{
		if (nIn<nOut) return;

		for (Uint i=1; i<nOut; i++) {
			Real v_d_v_pre = 0.;
			for (Uint k=0; k<nIn; k++)
				v_d_v_pre += *(_weights+n0+k*n_simd+i)* *(_weights+n0+k*n_simd+i);
			if(v_d_v_pre<std::numeric_limits<Real>::epsilon())
        die("Initialization problem\n");

			for (Uint j=0; j<i;  j++) {
				Real u_d_u = 0.0;
				Real v_d_u = 0.0;
				for (Uint k=0; k<nIn; k++) {
					u_d_u += *(_weights+n0+k*n_simd+j)* *(_weights+n0+k*n_simd+j);
					v_d_u += *(_weights+n0+k*n_simd+j)* *(_weights+n0+k*n_simd+i);
				}
				if(u_d_u<std::numeric_limits<Real>::epsilon())
          die("Initialization problem\n");

				for (Uint k=0; k<nIn; k++)
					*(_weights+n0+k*n_simd+i) -= v_d_u/u_d_u * *(_weights+n0+k*n_simd+j);
			}

			Real v_d_v_post = 0.0;
			for (Uint k=0; k<nIn; k++)
				v_d_v_post += *(_weights+n0+k*n_simd+i)* *(_weights+n0+k*n_simd+i);

			if(v_d_v_post<std::numeric_limits<Real>::epsilon())
        die("Initialization problem\n");

			for (Uint k=0; k<nIn; k++)
				*(_weights+n0+k*n_simd+i) *= std::sqrt(v_d_v_pre/v_d_v_post);
		}
	}

	inline void regularize(Real* const weights, const Real lambda) const
	{
		//not sure:
		Lpenalization(weights, iW, nW, lambda);
	}
};

class NormalLink: public Link
{
 public:
	/*
     a link here is defined as link layer to layer:
     index iI along the network activation outvals representing the index of the first neuron of input layer
     the number nI of neurons of the input layer
     the index iO of the first neuron of the output layer
     the number of neurons in the output layer nO
     the index of the first weight iW along the weight vector
     the weights are all to all: so this link occupies space iW to (iW + nI*nO) along weight vector
	 */
	NormalLink(Uint _nI, Uint _iI, Uint _nO, Uint _iO, Uint _iW, Uint _nO_simd) :
	Link(_nI, _iI, _nO, _iO, _iW, _nO_simd, _nI*_nO_simd)
	{
		assert(iW % (__vec_width__/sizeof(Real)) == 0);
		assert(iI % (__vec_width__/sizeof(Real)) == 0);
		assert(iO % (__vec_width__/sizeof(Real)) == 0);
		assert(nO_simd % (__vec_width__/sizeof(Real)) == 0);
		print();
		assert(nI>0 && nO>0 && iI>=0 && iO>=0 && iW>=0);
	}

	void print() const
	{
		cout << "Normal link: nInputs="<< nI << " IDinput=" << iI
			<< " nOutputs=" << nO << " IDoutput" << iO << " IDweight" << iW
			<< " nWeights" << nW << " nO_simd"<<nO_simd << endl;
		fflush(0);
	}
	void save(vector<Real> & out, Real* const _weights) const override
	{
		_save(out, _weights, iW, nO, nI, nO_simd);
	}
	void restart(vector<Real> & buf, Real* const _weights) const override
	{
		_restart(buf, _weights, iW, nO, nI, nO_simd);
	}
	void initialize(mt19937*const gen, Real*const _weights, const Function*const func) const
	{
		_initialize(gen, _weights, func->initFactor(nI,nO), iW, nO, nI, nO_simd);
	}

	inline void propagate(const Activation* const netFrom, Activation* const netTo,
		const Real* const weights) const
	{
		const Real* __restrict__ const link_input = netFrom->outvals +iI;
		Real* __restrict__ const link_outputs = netTo->in_vals +iO;
		//__builtin_assume_aligned(link_outputs, __vec_width__);
		//__builtin_assume_aligned(link_input, __vec_width__);
		for (Uint i = 0; i < nI; i++) {
			const Real* __restrict__ const link_weights = weights +iW +nO_simd*i;
			//__builtin_assume_aligned(link_weights, __vec_width__);
			for (Uint o = 0; o < nO; o++) {
				link_outputs[o] += link_input[i] * link_weights[o];
			}
		}
	}

	inline void backPropagate(Activation* const netFrom, const Activation* const netTo,
		const Real* const weights, Real* const gradW) const
	{
		const Real* __restrict__ const layer_input = netFrom->outvals + iI;
		const Real* __restrict__ const deltas = netTo->errvals + iO;
		Real* __restrict__ const link_errors = netFrom->errvals + iI;
		//__builtin_assume_aligned(link_errors, __vec_width__);
		//__builtin_assume_aligned(layer_input, __vec_width__);
		//__builtin_assume_aligned(deltas, __vec_width__);

		for (Uint i = 0; i < nI; i++) {
			const Real* __restrict__ const link_weights = weights +iW +nO_simd*i;
			Real* __restrict__ const link_dEdW = gradW +iW +nO_simd*i;
			//__builtin_assume_aligned(link_weights, __vec_width__);
			//__builtin_assume_aligned(link_dEdW, __vec_width__);
			for (Uint o = 0; o < nO; o++) {
				link_dEdW[o] += layer_input[i] * deltas[o];
				link_errors[i] += deltas[o] * link_weights[o];
			}
		}
	}
};

class LinkToLSTM : public Link
{
 public:
	/*
     if link is TO lstm, then the rules change a bit
     each LSTM block contains 4 neurons, one is the proper cell and then there are the 3 gates
     if a input signal is connected to one of the four, is also connected to the others
     thus we just need the index of the first weight for the 3 gates (could have skipped this, iWi = iW + nO*nI and so forth)
     additionally the LSTM contains a memory, contained in Activation->ostate
     memory and gates are treated differently than normal neurons, therefore are contained in separate array, and i keep track of the position with iC
	 */
	const Uint iC, iWI, iWF, iWO;

	LinkToLSTM(Uint _nI, Uint _iI, Uint _nO, Uint _iO, Uint _iC, Uint _iW,
		Uint _iWI, Uint _iWF, Uint _iWO, Uint _nO_simd) :
		Link(_nI, _iI, _nO, _iO, _iW, _nO_simd, _nI*_nO_simd), iC(_iC),
		iWI(_iWI), iWF(_iWF), iWO(_iWO) //i care nW per neuron, just for the asserts
	{
		assert(iW  % (__vec_width__/sizeof(Real)) == 0);
		assert(iWI % (__vec_width__/sizeof(Real)) == 0);
		assert(iWF % (__vec_width__/sizeof(Real)) == 0);
		assert(iWO % (__vec_width__/sizeof(Real)) == 0);
		assert(iI  % (__vec_width__/sizeof(Real)) == 0);
		assert(iC  % (__vec_width__/sizeof(Real)) == 0);
		assert(iO  % (__vec_width__/sizeof(Real)) == 0);
		assert(nO_simd % (__vec_width__/sizeof(Real)) == 0);
		print();
		assert(iWI==iW +nW);
		assert(iWF==iWI+nW);
		assert(iWO==iWF+nW);
		assert(iC>=0 && iWI>0 && iWF>0 && iWO>0);
	}

	void print() const override
			{
		cout << "LSTM link: nInputs="<< nI << " IDinput=" << iI
		<< " nOutputs=" << nO << " IDoutput" << iO << " IDcell" << iC
		<< " IDweight" << iW << " nWeights" << nW << " nO_simd"<<nO_simd << endl;
		fflush(0);
			}

  void initialize(mt19937*const gen, Real*const _weights, const Function*const func) const
	{
    const Real width = 2*std::max(nO,nI); //stupid workaround...
    const Real gatesFac = std::sqrt(6./(width + nO));
		_initialize(gen, _weights, func->initFactor(width,nO),iW,  nO, nI, nO_simd);
		_initialize(gen, _weights, gatesFac,                  iWI, nO, nI, nO_simd);
		_initialize(gen, _weights, gatesFac,                  iWF, nO, nI, nO_simd);
		_initialize(gen, _weights, gatesFac,                  iWO, nO, nI, nO_simd);
	}
	void save(vector<Real> & out, Real* const _weights) const override
	{
		_save(out, _weights, iW,  nO, nI, nO_simd);
		_save(out, _weights, iWI, nO, nI, nO_simd);
		_save(out, _weights, iWF, nO, nI, nO_simd);
		_save(out, _weights, iWO, nO, nI, nO_simd);
	}
	void restart(vector<Real> & buf, Real* const _weights) const override
	{
		_restart(buf, _weights, iW,  nO, nI, nO_simd);
		_restart(buf, _weights, iWI, nO, nI, nO_simd);
		_restart(buf, _weights, iWF, nO, nI, nO_simd);
		_restart(buf, _weights, iWO, nO, nI, nO_simd);
	}

	void propagate(const Activation* const netFrom, Activation* const netTo,
		const Real* const weights) const
	{
		const Real* __restrict__ const link_input = netFrom->outvals + iI;
		Real* __restrict__ const inputs = netTo->in_vals + iO;
		Real* __restrict__ const inputI = netTo->iIGates + iC;
		Real* __restrict__ const inputF = netTo->iFGates + iC;
		Real* __restrict__ const inputO = netTo->iOGates + iC;
		//__builtin_assume_aligned(inputs, __vec_width__);
		//__builtin_assume_aligned(inputI, __vec_width__);
		//__builtin_assume_aligned(inputF, __vec_width__);
		//__builtin_assume_aligned(inputO, __vec_width__);
		//__builtin_assume_aligned(link_input, __vec_width__);

		for (Uint i = 0; i < nI; i++) {
			const Real* __restrict__ const weights_toCell  = weights + iW  + nO_simd*i;
			const Real* __restrict__ const weights_toIgate = weights + iWI + nO_simd*i;
			const Real* __restrict__ const weights_toFgate = weights + iWF + nO_simd*i;
			const Real* __restrict__ const weights_toOgate = weights + iWO + nO_simd*i;
			//__builtin_assume_aligned(weights_toCell, __vec_width__);
			//__builtin_assume_aligned(weights_toIgate, __vec_width__);
			//__builtin_assume_aligned(weights_toFgate, __vec_width__);
			//__builtin_assume_aligned(weights_toOgate, __vec_width__);

			for (Uint o = 0; o < nO; o++) {
				inputs[o] += link_input[i] * weights_toCell[o];
				inputI[o] += link_input[i] * weights_toIgate[o];
				inputF[o] += link_input[i] * weights_toFgate[o];
				inputO[o] += link_input[i] * weights_toOgate[o];
			}
		}
	}

	void backPropagate(Activation* const netFrom, const Activation* const netTo,
		const Real* const weights, Real* const gradW) const
	{
		const Real* __restrict__ const layer_input = netFrom->outvals + iI;
		Real* __restrict__ const link_errors = netFrom->errvals + iI;
		const Real* __restrict__ const deltaI = netTo->eIGates +iC;
		const Real* __restrict__ const deltaF = netTo->eFGates +iC;
		const Real* __restrict__ const deltaO = netTo->eOGates +iC;
		const Real* __restrict__ const deltaC = netTo->eMCell +iC;
		//__builtin_assume_aligned(layer_input, __vec_width__);
		//__builtin_assume_aligned(link_errors, __vec_width__);
		//__builtin_assume_aligned(deltaI, __vec_width__);
		//__builtin_assume_aligned(deltaF, __vec_width__);
		//__builtin_assume_aligned(deltaO, __vec_width__);
		//__builtin_assume_aligned(deltaC, __vec_width__);

		for (Uint i = 0; i < nI; i++) {
			const Real* __restrict__ const w_toOgate = weights +iWO +nO_simd*i;
			const Real* __restrict__ const w_toFgate = weights +iWF +nO_simd*i;
			const Real* __restrict__ const w_toIgate = weights +iWI +nO_simd*i;
			const Real* __restrict__ const w_toCell  = weights +iW  +nO_simd*i;
			Real* __restrict__ const dw_toOgate = gradW +iWO +nO_simd*i;
			Real* __restrict__ const dw_toFgate = gradW +iWF +nO_simd*i;
			Real* __restrict__ const dw_toIgate = gradW +iWI +nO_simd*i;
			Real* __restrict__ const dw_toCell  = gradW +iW  +nO_simd*i;
			//__builtin_assume_aligned(dw_toOgate, __vec_width__);
			//__builtin_assume_aligned(dw_toFgate, __vec_width__);
			//__builtin_assume_aligned(dw_toIgate, __vec_width__);
			//__builtin_assume_aligned(dw_toCell,  __vec_width__);
			//__builtin_assume_aligned(w_toOgate, __vec_width__);
			//__builtin_assume_aligned(w_toFgate, __vec_width__);
			//__builtin_assume_aligned(w_toIgate, __vec_width__);
			//__builtin_assume_aligned(w_toCell,  __vec_width__);

			for (Uint o = 0; o < nO; o++) {
				dw_toOgate[o] += layer_input[i] * deltaO[o];
				dw_toCell[o]  += layer_input[i] * deltaC[o];
				dw_toIgate[o] += layer_input[i] * deltaI[o];
				dw_toFgate[o] += layer_input[i] * deltaF[o];
				link_errors[i] += deltaO[o]*w_toOgate[o] + deltaC[o]*w_toCell[o] +
						deltaI[o]*w_toIgate[o] + deltaF[o]*w_toFgate[o];
			}
		}
	}
};

class LinkToConv2D : public Link
{
 public:
	const Uint inputWidth, inputHeight, inputDepth;
	const Uint filterWidth, filterHeight, outputDepth_simd;
	const Uint outputWidth, outputHeight, outputDepth;
	const Uint strideX, strideY, padX, padY;

	LinkToConv2D(Uint _nI, Uint _iI, Uint _nO, Uint _iO, Uint _iW, Uint _nO_simd,
		Uint _inW, Uint _inH, Uint _inD, Uint _fW, Uint _fH, Uint _fN, Uint _outW,
		Uint _outH, Uint _sX=1, Uint _sY=1, Uint _pX=0, Uint _pY=0) :
		Link(_nI, _iI, _nO, _iO, _iW, _nO_simd, _fW*_fH*_nO_simd*_inD),
		inputWidth(_inW), inputHeight(_inH), inputDepth(_inD),
		filterWidth(_fW), filterHeight(_fH), outputDepth_simd(_nO_simd),
		outputWidth(_outW), outputHeight(_outH), outputDepth(_fN),
		strideX(_sX), strideY(_sY), padX(_pX), padY(_pY)
	{
		assert(iW % (__vec_width__/sizeof(Real)) == 0);
		assert(iI % (__vec_width__/sizeof(Real)) == 0);
		assert(iO % (__vec_width__/sizeof(Real)) == 0);
		assert(outputDepth_simd % (__vec_width__/sizeof(Real)) == 0);
		assert(nW>0);
		assert(inputWidth*inputHeight*inputDepth == nI);
		assert(outputWidth*outputHeight*outputDepth == nO);
		//this class prescribes the bottom padding, let's figure out if the top one makes sense
		// inW_withPadding = inputWidth + bottomPad + topPad (where bottomPad = padX,padY)
		//first: All pixels of input are covered. topPad must be >=0, and stride leq than filter size
		assert((outputWidth -1)*strideX + filterWidth  - (inputWidth+padX)  >= 0);
		assert((outputHeight-1)*strideY + filterHeight - (inputHeight+padY) >= 0);
		assert(filterWidth >= strideX && filterHeight >= strideY);
		//second condition: do not feed an output pixel only with padding
		assert((outputWidth -1)*strideX + filterWidth  - (inputWidth+padX)  < filterWidth);
		assert((outputHeight-1)*strideY + filterHeight - (inputHeight+padY) < filterHeight);
		assert(padX < filterWidth && padY < filterHeight);
		print();
	}

	void print() const override
	{
		printf("iW=%d, nI=%d, iI=%d, nO=%d, iO=%d, nW=%d\n",
		iW,nI,iI,nO,iO,nW);
		printf("inputWidth=%d, inputHeight=%d, inputDepth=%d\n",
		inputWidth, inputHeight, inputDepth);
		printf("outputWidth=%d, outputHeight=%d, outputDepth=%d (%d)\n",
		outputWidth, outputHeight, outputDepth, outputDepth_simd);
		printf("filterWidth=%d, filterHeight=%d, strideX=%d, strideY=%d, padX=%d, padY=%d\n",
				filterWidth, filterHeight, strideX, strideY, padX, padY);
		fflush(0);
	}
  void initialize(mt19937*const gen, Real*const _weights, const Function*const func) const
	{
    const Uint nAdded = filterWidth*filterHeight*inputDepth;
		assert(outputDepth_simd*nAdded == nW);
		_initialize(gen, _weights, func->initFactor(nAdded,outputDepth), iW, nO, nAdded, outputDepth_simd);
	}
	void save(vector<Real> & out, Real* const _weights) const override
	{
		const Uint nAdded = filterWidth*filterHeight*inputDepth;
		_save(out, _weights, iW, nO, nAdded, outputDepth_simd);
	}
	void restart(vector<Real> & buf, Real* const _weights) const override
	{
		const Uint nAdded = filterWidth*filterHeight*inputDepth;
		_restart(buf, _weights, iW, nO, nAdded, outputDepth_simd);
	}

	void propagate(const Activation* const netFrom, Activation* const netTo, const Real* const weights) const
	{
		for(Uint ox=0; ox<outputWidth;  ox++)
			for(Uint oy=0; oy<outputHeight; oy++) {
				const int ix = static_cast<int>(ox*strideX) -  static_cast<int>(padX);
				const int iy = static_cast<int>(oy*strideY) -  static_cast<int>(padY);
				for(Uint fx=0; fx<filterWidth; fx++)
					for(Uint fy=0; fy<filterHeight; fy++) {
						const int cx=ix+static_cast<int>(fx);
						const int cy=iy+static_cast<int>(fy);
						//padding: skip addition if outside input boundaries
            if ( cx < 0 || static_cast<Uint>(cx) >= inputWidth
              || cy < 0 || static_cast<Uint>(cy) >= inputHeight)
							continue;

						const Real* __restrict__ const link_inputs =
								netFrom->outvals +iI +inputDepth*(cy +inputHeight*cx);
						Real* __restrict__ const link_outputs =
								netTo->in_vals +iO+outputDepth*(oy+outputHeight*ox);
						//__builtin_assume_aligned(link_outputs, __vec_width__);
						//__builtin_assume_aligned(link_inputs, __vec_width__);

						for(Uint iz=0; iz<inputDepth; iz++) {
							const Real* __restrict__ const link_weights =
									weights +iW +outputDepth*(iz +inputDepth*(fy +filterHeight*fx));
							//__builtin_assume_aligned(link_weights, __vec_width__);

							for(Uint fz=0; fz<outputDepth; fz++) {
								link_outputs[fz] += link_inputs[iz] * link_weights[fz];
							}
						}
					}
			}
	}

	void backPropagate(Activation* const netFrom, const Activation* const netTo,
		const Real* const weights, Real* const gradW) const
	{
		for(Uint ox=0; ox<outputWidth;  ox++)
			for(Uint oy=0; oy<outputHeight; oy++) {
				const int ix = static_cast<int>(ox*strideX) -  static_cast<int>(padX);
				const int iy = static_cast<int>(oy*strideY) -  static_cast<int>(padY);
				for(Uint fx=0; fx<filterWidth; fx++)
					for(Uint fy=0; fy<filterHeight; fy++) {
						const int cx=ix+static_cast<int>(fx);
						const int cy=iy+static_cast<int>(fy);
						//padding: skip addition if outside input boundaries
						if ( cx < 0 || static_cast<Uint>(cx) >= inputWidth
              || cy < 0 || static_cast<Uint>(cy) >= inputHeight)
							continue;

						const Real* __restrict__ const link_inputs =
								netFrom->outvals +iI +inputDepth*(cy +inputHeight*cx);
						Real* __restrict__ const link_errors =
								netFrom->errvals +iI +inputDepth*(cy +inputHeight*cx);
						const Real* __restrict__ const deltas =
								netTo->errvals +iO+outputDepth*(oy+outputHeight*ox);
						//__builtin_assume_aligned(link_inputs, __vec_width__);
						//__builtin_assume_aligned(link_errors, __vec_width__);
						//__builtin_assume_aligned(deltas, __vec_width__);

						for(Uint iz=0; iz<inputDepth; iz++) {
							const Real* __restrict__ const link_weights =
									weights +iW +outputDepth*(iz+inputDepth*(fy+filterHeight*fx));
							Real* __restrict__ const link_dEdW    =
									gradW +iW +outputDepth*(iz+inputDepth*(fy+filterHeight*fx));
							//__builtin_assume_aligned(link_weights, __vec_width__);
							//__builtin_assume_aligned(link_dEdW, __vec_width__);

							for(Uint fz=0; fz<outputDepth; fz++) {
								link_errors[iz] += link_weights[fz]*deltas[fz];
								link_dEdW[fz] += link_inputs[iz]*deltas[fz];
							}
						}
					}
			}
	}
};
