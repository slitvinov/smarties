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
#include "Activations.h"
#include <iostream>
#include <cassert>
#include <sstream>
#include <iomanip>

// #define _whitenTarget_

using namespace std;

class Link
{
public:
	virtual ~Link() {}
	virtual void print() const = 0;
	virtual void initialize(mt19937* const gen, Real* const _weights) const = 0;
	virtual void restart(std::istringstream & buf, Real* const _weights) const = 0;
	virtual void save(std::ostringstream & buf, Real* const _weights) const = 0;
	//virtual void propagate(const Activation* const netFrom, Activation* const netTo, const Real* const weights) const = 0;
	//virtual void backPropagate(Activation* const netFrom, const Activation* const netTo, const Real* const weights, Real* const gradW) const =0;
	void orthogonalize(const int n0, Real* const _weights, int nOut, int nIn, int n_simd) const
	{
		if (nIn<nOut) return;

		for (int i=1; i<nOut; i++) {
			Real v_d_v_pre = 0.;
			for (int k=0; k<nIn; k++)
				v_d_v_pre += *(_weights+n0+k*n_simd+i)* *(_weights+n0+k*n_simd+i);
			if(v_d_v_pre<std::numeric_limits<Real>::epsilon()) die("Initialization problem\n");

			for (int j=0; j<i;  j++) {
				Real u_d_u = 0.0;
				Real v_d_u = 0.0;
				for (int k=0; k<nIn; k++) {
					u_d_u += *(_weights+n0+k*n_simd+j)* *(_weights+n0+k*n_simd+j);
					v_d_u += *(_weights+n0+k*n_simd+j)* *(_weights+n0+k*n_simd+i);
				}
				if(u_d_u<std::numeric_limits<Real>::epsilon()) die("Initialization problem\n");

				for (int k=0; k<nIn; k++)
					*(_weights+n0+k*n_simd+i) -= v_d_u/u_d_u * *(_weights+n0+k*n_simd+j);
			}

			Real v_d_v_post = 0.0;
			for (int k=0; k<nIn; k++)
				v_d_v_post += *(_weights+n0+k*n_simd+i)* *(_weights+n0+k*n_simd+i);
			if(v_d_v_post<std::numeric_limits<Real>::epsilon()) die("Initialization problem\n");

			for (int k=0; k<nIn; k++)
				*(_weights+n0+k*n_simd+i) *= std::sqrt(v_d_v_pre/v_d_v_post);
		}
	}
	virtual void resetRunning() {};
	virtual void updateRunning(Activation* const act, const int counter) {};
	virtual void printRunning(int counter, std::ostringstream & oa, std::ostringstream & os) {};
};

class NormalLink: public Link
{
 public:
	const int iW, nI, iI, nO, iO, nO_simd, nW;
    /*
     a link here is defined as link layer to layer:
     index iI along the network activation outvals representing the index of the first neuron of input layer
     the number nI of neurons of the input layer
     the index iO of the first neuron of the output layer
     the number of neurons in the output layer nO
     the index of the first weight iW along the weight vector
     the weights are all to all: so this link occupies space iW to (iW + nI*nO) along weight vector
     */
	 	NormalLink(int _nI, int _iI, int _nO, int _iO, int _iW, int _nO_simd) :
		iW(_iW), nI(_nI), iI(_iI), nO(_nO), iO(_iO), nO_simd(_nO_simd), nW(_nI*_nO_simd)
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
        cout << "Normal link: nInputs="<< nI << " IDinput=" << iI << " nOutputs=" << nO << " IDoutput" << iO << " IDweight" << iW << " nWeights" << nW << " nO_simd"<<nO_simd << endl;
        fflush(0);
    }

    void initialize(mt19937* const gen, Real* const _weights) const override
    {
      //printf("Initializing normal\n");
        const Real range = std::sqrt(6./(nO + nI));
        uniform_real_distribution<Real> dis(-range,range);
        //normal_distribution<Real> dis(0.,range);

        for (int i = 0; i < nI; i++)
				for (int o = 0; o < nO; o++)
					_weights[iW + nO_simd*i + o] = dis(*gen);

        //orthogonalize(iW, _weights, nO, nI, nO_simd);
    }

    void restart(std::istringstream & buf, Real* const _weights) const override
    {
    	for (int i = 0; i < nI; i++)
        for (int o = 0; o < nO; o++) {
        	const int w = iW + nO_simd*i + o;
            Real tmp;
            buf >> tmp;
            assert(not std::isnan(tmp) & not std::isinf(tmp));
            _weights[w] = tmp;
        }
    }

    void save(std::ostringstream & out, Real* const _weights) const override
    {
        out << std::setprecision(10);
    	for (int i=0; i<nI; i++) for (int o=0; o<nO; o++)
        	out << _weights[iW + nO_simd*i + o]  << "\n";
    }

    inline void propagate(const Activation* const netFrom, Activation* const netTo, const Real* const weights) const
    {
        const Real* __restrict__ const link_input = netFrom->outvals +iI;
        Real* __restrict__ const link_outputs = netTo->in_vals +iO;
        __builtin_assume_aligned(link_outputs, __vec_width__);
        __builtin_assume_aligned(link_input, __vec_width__);
				#if 1
        for (int i = 0; i < nI; i++) {
            const Real* __restrict__ const link_weights = weights +iW +nO_simd*i;
            __builtin_assume_aligned(link_weights, __vec_width__);
        for (int o = 0; o < nO; o++) {
            link_outputs[o] += link_input[i] * link_weights[o];
        }
        }
				#else
        for (int o = 0; o < nO; o++) {
            const Real* __restrict__ const link_weights = weights +iW +nI*o;
        for (int i = 0; i < nI; i++) {
            link_outputs[o] += link_input[i] * link_weights[i];
        }
        }
				#endif
    }

    inline void backPropagate(Activation* const netFrom, const Activation* const netTo, const Real* const weights, Real* const gradW) const
    {
        const Real* __restrict__ const layer_input = netFrom->outvals + iI;
        const Real* __restrict__ const deltas = netTo->errvals + iO;
        Real* __restrict__ const link_errors = netFrom->errvals + iI;
        __builtin_assume_aligned(link_errors, __vec_width__);
        __builtin_assume_aligned(layer_input, __vec_width__);
        __builtin_assume_aligned(deltas, __vec_width__);

        for (int i = 0; i < nI; i++) {
            const Real* __restrict__ const link_weights = weights +iW +nO_simd*i;
            Real* __restrict__ const link_dEdW = gradW +iW +nO_simd*i;
            __builtin_assume_aligned(link_weights, __vec_width__);
            __builtin_assume_aligned(link_dEdW, __vec_width__);
        for (int o = 0; o < nO; o++) {
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
	const int iW, nI, iI, nO, iO, iC, iWI, iWF, iWO, nO_simd, nW;

	LinkToLSTM(int _nI, int _iI, int _nO, int _iO, int _iC, int _iW, int _iWI, int _iWF, int _iWO, int _nO_simd) :
	iW(_iW), nI(_nI), iI(_iI), nO(_nO), iO(_iO), iC(_iC), iWI(_iWI), iWF(_iWF), iWO(_iWO), nO_simd(_nO_simd), nW(_nI*_nO_simd) //i care nW per neuron, just for the asserts
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
		assert(iC>=0 && iWI>=0 && iWF>=0 && iWO>=0);
	}

    void print() const override
    {
        cout << "LSTM link: nInputs="<< nI << " IDinput=" << iI << " nOutputs=" << nO << " IDoutput" << iO << " IDcell" << iC << " IDweight" << iW << " nWeights" << nW << " nO_simd"<<nO_simd << endl;
        fflush(0);
    }

    void initialize(mt19937* const gen, Real* const _weights) const override
    {
      	printf("Initializing LSTM\n");
				#ifndef __posDef_layers_
        const Real range = std::sqrt(6./(nO + nI));
        #else
				const Real range = 2./nI;
        #endif
        uniform_real_distribution<Real> dis(-range,range);
        //normal_distribution<Real> dis(0.,range);

	      for (int i = 0; i < nI; i++)
				for (int o = 0; o < nO; o++) {
					_weights[iW  + nO_simd*i + o] = dis(*gen);
					_weights[iWI + nO_simd*i + o] = dis(*gen);
					_weights[iWF + nO_simd*i + o] = dis(*gen);
					_weights[iWO + nO_simd*i + o] = dis(*gen);
				}
        //orthogonalize(iW,  _weights, nO, nI, nO_simd);
        //orthogonalize(iWI, _weights, nO, nI, nO_simd);
        //orthogonalize(iWF, _weights, nO, nI, nO_simd);
        //orthogonalize(iWO, _weights, nO, nI, nO_simd);
    }

    void restart(std::istringstream & buf, Real* const _weights) const override
    {
        Real tmp;
    	for (int i=0; i<nI; i++) for (int o=0; o<nO; o++) {
            buf >> tmp;
            assert(not std::isnan(tmp) & not std::isinf(tmp));
            _weights[iW + nO_simd*i + o] = tmp;
        }
    	for (int i=0; i<nI; i++) for (int o=0; o<nO; o++) {
            buf >> tmp;
            assert(not std::isnan(tmp) & not std::isinf(tmp));
            _weights[iWI + nO_simd*i + o] = tmp;
        }
    	for (int i=0; i<nI; i++) for (int o=0; o<nO; o++) {
            buf >> tmp;
            assert(not std::isnan(tmp) & not std::isinf(tmp));
            _weights[iWF + nO_simd*i + o] = tmp;
        }
    	for (int i=0; i<nI; i++) for (int o=0; o<nO; o++) {
            buf >> tmp;
            assert(not std::isnan(tmp) & not std::isinf(tmp));
            _weights[iWO + nO_simd*i + o] = tmp;
        }
    }

    void save(std::ostringstream & out, Real* const _weights) const override
    {
        out << std::setprecision(10);
        for (int i=0; i<nI; i++) for (int o=0; o<nO; o++)
        	out << _weights[iW  + nO_simd*i + o] << "\n";

        for (int i=0; i<nI; i++) for (int o=0; o<nO; o++)
        	out << _weights[iWI + nO_simd*i + o] << "\n";

        for (int i=0; i<nI; i++) for (int o=0; o<nO; o++)
        	out << _weights[iWF + nO_simd*i + o] << "\n";

        for (int i=0; i<nI; i++) for (int o=0; o<nO; o++)
        	out << _weights[iWO + nO_simd*i + o] << "\n";
    }

    void propagate(const Activation* const netFrom, Activation* const netTo, const Real* const weights) const
    {
        const Real* __restrict__ const link_input = netFrom->outvals + iI;
        Real* __restrict__ const inputs = netTo->in_vals + iO;
        Real* __restrict__ const inputI = netTo->iIGates + iC;
        Real* __restrict__ const inputF = netTo->iFGates + iC;
        Real* __restrict__ const inputO = netTo->iOGates + iC;
        __builtin_assume_aligned(inputs, __vec_width__);
        __builtin_assume_aligned(inputI, __vec_width__);
        __builtin_assume_aligned(inputF, __vec_width__);
        __builtin_assume_aligned(inputO, __vec_width__);
        __builtin_assume_aligned(link_input, __vec_width__);

        for (int i = 0; i < nI; i++) {
            const Real* __restrict__ const weights_toCell  = weights + iW  + nO_simd*i;
            const Real* __restrict__ const weights_toIgate = weights + iWI + nO_simd*i;
            const Real* __restrict__ const weights_toFgate = weights + iWF + nO_simd*i;
            const Real* __restrict__ const weights_toOgate = weights + iWO + nO_simd*i;
            __builtin_assume_aligned(weights_toCell, __vec_width__);
            __builtin_assume_aligned(weights_toIgate, __vec_width__);
            __builtin_assume_aligned(weights_toFgate, __vec_width__);
            __builtin_assume_aligned(weights_toOgate, __vec_width__);

        for (int o = 0; o < nO; o++) {
            inputs[o] += link_input[i] * weights_toCell[o];
            inputI[o] += link_input[i] * weights_toIgate[o];
            inputF[o] += link_input[i] * weights_toFgate[o];
            inputO[o] += link_input[i] * weights_toOgate[o];
        }
        }
    }

    void backPropagate(Activation* const netFrom, const Activation* const netTo, const Real* const weights, Real* const gradW) const
    {
        const Real* __restrict__ const layer_input = netFrom->outvals + iI;
        Real* __restrict__ const link_errors = netFrom->errvals + iI;
        const Real* __restrict__ const deltaI = netTo->eIGates +iC;
        const Real* __restrict__ const deltaF = netTo->eFGates +iC;
        const Real* __restrict__ const deltaO = netTo->eOGates +iC;
        const Real* __restrict__ const deltaC = netTo->eMCell +iC;
        __builtin_assume_aligned(layer_input, __vec_width__);
        __builtin_assume_aligned(link_errors, __vec_width__);
        __builtin_assume_aligned(deltaI, __vec_width__);
        __builtin_assume_aligned(deltaF, __vec_width__);
        __builtin_assume_aligned(deltaO, __vec_width__);
        __builtin_assume_aligned(deltaC, __vec_width__);

        for (int i = 0; i < nI; i++) {
            const Real* __restrict__ const w_toOgate = weights +iWO +nO_simd*i;
            const Real* __restrict__ const w_toFgate = weights +iWF +nO_simd*i;
            const Real* __restrict__ const w_toIgate = weights +iWI +nO_simd*i;
            const Real* __restrict__ const w_toCell  = weights +iW  +nO_simd*i;
            Real* __restrict__ const dw_toOgate = gradW +iWO +nO_simd*i;
            Real* __restrict__ const dw_toFgate = gradW +iWF +nO_simd*i;
            Real* __restrict__ const dw_toIgate = gradW +iWI +nO_simd*i;
            Real* __restrict__ const dw_toCell  = gradW +iW  +nO_simd*i;
            __builtin_assume_aligned(dw_toOgate, __vec_width__);
            __builtin_assume_aligned(dw_toFgate, __vec_width__);
            __builtin_assume_aligned(dw_toIgate, __vec_width__);
            __builtin_assume_aligned(dw_toCell,  __vec_width__);
            __builtin_assume_aligned(w_toOgate, __vec_width__);
            __builtin_assume_aligned(w_toFgate, __vec_width__);
            __builtin_assume_aligned(w_toIgate, __vec_width__);
            __builtin_assume_aligned(w_toCell,  __vec_width__);

        for (int o = 0; o < nO; o++) {
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
	const int iW, nI, iI, nO, iO, outputDepth_simd;
    const int inputWidth, inputHeight, inputDepth;
    const int filterWidth, filterHeight;
    const int outputWidth, outputHeight, outputDepth;
	const int strideX, strideY, padX, padY;
	const int nW;

	LinkToConv2D(int _nI, int _iI, int _nO, int _iO, int _iW, int _nO_simd,
				int _inW, int _inH, int _inD,
    			int _fW, int _fH, int _fN, int _outW, int _outH,
				int _sX=1, int _sY=1, int _pX=0, int _pY=0) :
	iW(_iW), nI(_nI), iI(_iI), nO(_nO), iO(_iO), outputDepth_simd(_nO_simd), inputWidth(_inW), inputHeight(_inH), inputDepth(_inD), filterWidth(_fW), filterHeight(_fH),
	outputWidth(_outW), outputHeight(_outH), outputDepth(_fN), strideX(_sX), strideY(_sY), padX(_pX), padY(_pY), nW(_fW*_fH*_nO_simd*_inD)
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
			printf("iW=%d, nI=%d, iI=%d, nO=%d, iO=%d, nW=%d\n",iW,nI,iI,nO,iO,nW);
			printf("inputWidth=%d, inputHeight=%d, inputDepth=%d\n",inputWidth, inputHeight, inputDepth);
			printf("outputWidth=%d, outputHeight=%d, outputDepth=%d (%d)\n",outputWidth, outputHeight, outputDepth, outputDepth_simd);
			printf("filterWidth=%d, filterHeight=%d, strideX=%d, strideY=%d, padX=%d, padY=%d\n",
			filterWidth, filterHeight, strideX, strideY, padX, padY);
        fflush(0);
    }

    void initialize(mt19937* const gen, Real* const _weights) const override
    {
      printf("Initializing conv\n");
        const int nAdded = filterWidth*filterHeight*inputDepth;
        const Real range = std::sqrt(6./(nAdded+outputDepth));
        uniform_real_distribution<Real> dis(-range,range);
        //normal_distribution<Real> dis(0.,range);
        assert(outputDepth_simd*nAdded == nW);
        for (int i = 0; i < nAdded; i++)
				for (int o = 0; o < nO; o++)
					_weights[iW + outputDepth_simd*i + o] = dis(*gen);

        orthogonalize(iW, _weights, outputDepth, nAdded, outputDepth_simd);
    }

    void restart(std::istringstream & buf, Real* const _weights) const override
    {
        const int nAdded = filterWidth*filterHeight*inputDepth;
        for (int i = 0; i < nAdded; i++)
		for (int o = 0; o < nO; o++) {
            Real tmp;
            buf >> tmp;
            assert(not std::isnan(tmp) & not std::isinf(tmp));
            *(_weights +iW + outputDepth_simd*i + o) = tmp;
        }
    }

    void save(std::ostringstream & out, Real* const _weights) const override
    {
        out << std::setprecision(10);
        const int nAdded = filterWidth*filterHeight*inputDepth;
        for (int i = 0; i < nAdded; i++)
				for (int o = 0; o < nO; o++)
					out << _weights[iW + outputDepth_simd*i + o] << "\n";
    }

    void propagate(const Activation* const netFrom, Activation* const netTo, const Real* const weights) const
    {
        for(int ox=0; ox<outputWidth;  ox++)
        for(int oy=0; oy<outputHeight; oy++) {
						const int ix = ox*strideX - padX;
						const int iy = oy*strideY - padY;
            for(int fx=0; fx<filterWidth; fx++)
            for(int fy=0; fy<filterHeight; fy++) {
                const int cx=ix+fx, cy=iy+fy;
                //padding: skip addition if outside input boundaries
                if (cx < 0 || cy < 0 || cx >= inputWidth || cy >= inputHeight) continue;

                const Real* __restrict__ const link_inputs =
													netFrom->outvals +iI +inputDepth*(cy +inputHeight*cx);
                			Real* __restrict__ const link_outputs =
														netTo->in_vals +iO+outputDepth*(oy+outputHeight*ox);
                __builtin_assume_aligned(link_outputs, __vec_width__);
								__builtin_assume_aligned(link_inputs, __vec_width__);

                for(int iz=0; iz<inputDepth; iz++) {
                    const Real* __restrict__ const link_weights =
										weights +iW +outputDepth*(iz +inputDepth*(fy +filterHeight*fx));
                    __builtin_assume_aligned(link_weights, __vec_width__);

                for(int fz=0; fz<outputDepth; fz++) {
									//printf("oz %d \n",iO +outputDepth*(oy +outputHeight*ox)+fz);
										//printf("iz %d \n",iI +inputDepth*(cy +inputHeight*cx)+iz);
											//printf("wz %d \n",iW +outputDepth*(iz +inputDepth*(fy +filterHeight*fx))+fz);
                    link_outputs[fz] += link_inputs[iz] * link_weights[fz];
								}
                }
            }
        }
    }

    void backPropagate(Activation* const netFrom, const Activation* const netTo, const Real* const weights, Real* const gradW) const
    {
        for(int ox=0; ox<outputWidth;  ox++)
        for(int oy=0; oy<outputHeight; oy++) {
            const int ix = ox*strideX - padX;
            const int iy = oy*strideY - padY;
            for(int fx=0; fx<filterWidth; fx++)
            for(int fy=0; fy<filterHeight; fy++) {
                const int cx = ix+fx;
								const int cy = iy+fy;
                //padding: skip addition if outside input boundaries
                if (cx < 0 || cy < 0 || cx >= inputWidth || cy >= inputHeight) continue;

                const Real* __restrict__ const link_inputs =
													netFrom->outvals +iI +inputDepth*(cy +inputHeight*cx);
                      Real* __restrict__ const link_errors =
													netFrom->errvals +iI +inputDepth*(cy +inputHeight*cx);
                const Real* __restrict__ const deltas =
														netTo->errvals +iO+outputDepth*(oy+outputHeight*ox);
                __builtin_assume_aligned(link_inputs, __vec_width__);
                __builtin_assume_aligned(link_errors, __vec_width__);
                __builtin_assume_aligned(deltas, __vec_width__);

                for(int iz=0; iz<inputDepth; iz++) {
                    const Real* __restrict__ const link_weights =
										weights +iW +outputDepth*(iz+inputDepth*(fy+filterHeight*fx));
                          Real* __restrict__ const link_dEdW    =
											gradW +iW +outputDepth*(iz+inputDepth*(fy+filterHeight*fx));
                    __builtin_assume_aligned(link_weights, __vec_width__);
                    __builtin_assume_aligned(link_dEdW, __vec_width__);

                for(int fz=0; fz<outputDepth; fz++) {
                    link_errors[iz] += link_weights[fz]*deltas[fz];
                    link_dEdW[fz] += link_inputs[iz]*deltas[fz];
                }
                }
            }
        }
    }
};

class WhiteningLink : public Link
{
    vector<Real> runningAvg, runningStd;
 public:
	const int iW, nI, iI, nO, iO, nW, nO_simd;
	WhiteningLink(int _nI, int _iI, int _nO, int _iO, int _iW, int _nO_simd) :
		iW(_iW), nI(_nI), iI(_iI), nO(_nO), iO(_iO), nW(2*_nI), nO_simd(_nO_simd)
    {
        print();
        assert(nI==nO && iI+nO_simd==iO);
    }

	void initialize(mt19937* const gen, Real* const _weights) const override
    {
        for (int p=0 ; p<2; p++)
        for (int o=0 ; o<nO; o++)
        	*(_weights +iW +p*nO_simd +o) = 1==p ? 1. : 0.; //set to 1 the scaling factor
    }

    void restart(std::istringstream & buf, Real* const _weights) const override
    {
        for (int p=0 ; p<2; p++)
        for (int o=0 ; o<nO; o++) {
            Real tmp;
            buf >> tmp;
            assert(not std::isnan(tmp) & not std::isinf(tmp));
            *(_weights +iW +p*nO_simd +o) = tmp;
        }
    }

    void save(std::ostringstream & out, Real* const _weights) const override
    {
        out << std::setprecision(10);
        for (int p=0; p<2; p++) for (int o=0; o<nO; o++)
            out << *(_weights +iW +p*nO_simd +o) << "\n";
    }

    void print() const override
    {
        cout << "Whitening link: nInputs="<< nI << " IDinput=" << iI << " nOutputs=" << nO << " IDoutput" << iO << " IDweight" << iW << " nO_simd"<<nO_simd<<endl;
        fflush(0);
    }

    void resetRunning() override {
    	if(runningAvg.size() != nO) runningAvg.resize(nO);
    	if(runningStd.size() != nO) runningStd.resize(nO);
    	for (int k=0; k<nO; k++) {
    		runningAvg[k] = 0;
    		runningStd[k] = 0;
    	}
    }

    void printRunning(int counter, std::ostringstream & oa, std::ostringstream & os) override {
    	counter = std::max(counter,2);
    	const Real invNm1 = 1./(counter-1);
		for (int i=0; i<nO; i++)  oa << runningAvg[i] << " ";
		for (int i=0; i<nO; i++)  os << runningStd[i]*invNm1 << " ";
    }

    void updateRunning(Activation* const act, const int counter) override {
    	assert(runningAvg.size() == nO && runningStd.size() == nO && counter>0);
    	const Real invN = 1./counter;

    	for (int k=0; k<nO; k++) {
    		const Real delta = act->in_vals[k+iO] - runningAvg[k];
    		runningAvg[k] += delta*invN;
    		runningStd[k] += delta*(act->in_vals[k+iO] - runningAvg[k]);
    	}
    }
};

struct Graph //misleading, this is just the graph for a single layer
{
	//TODO SIMD safety
    bool input, output, RNN, LSTM, Conv2D, normalize;
    int layerSize, layerSize_simd;
	int firstNeuron_ID; //recurrPos, normalPos;
    int firstState_ID;
    int firstBias_ID;
    int firstBiasWhiten;
    int firstBiasIG_ID, firstBiasFG_ID, firstBiasOG_ID;
    int layerWidth, layerHeight, layerDepth, layerDepth_simd;
    int padWidth, padHeight, featsWidth, featsHeight, featsNumber, strideWidth, strideHeight;
    vector<int> linkedTo;
    vector<Link*> * links;

    Graph() :
	input(false), output(false), RNN(false), LSTM(false), Conv2D(false), normalize(false),
	layerSize(0), layerSize_simd(0), firstNeuron_ID(0), firstState_ID(0), firstBias_ID(0), firstBiasWhiten(-1),
	firstBiasIG_ID(-1), firstBiasFG_ID(-1), firstBiasOG_ID(-1), //LSTM
	layerWidth(-1), layerHeight(-1), layerDepth(-1), layerDepth_simd(-1), padWidth(-1), padHeight(-1), //Conv2D
	featsWidth(-1), featsHeight(-1), featsNumber(-1), strideWidth(-1), strideHeight(-1)
    {
    	links = new vector<Link*>();
    }

    ~Graph()
    {
        for (auto& link : *links)
        	_dispose_object(link);
        _dispose_object(links);
    }

    void restart(std::istringstream & bufWeights,
    			 std::istringstream & bufBiases,
				 Real* const _weights,
				 Real* const _biases) const
    {
    	for (const auto & l : *(links))
			if(l not_eq nullptr) l->restart(bufWeights, _weights);

    	Real tmp;

			for (int w=firstBias_ID; w<firstBias_ID+layerSize; w++) {
				bufBiases >> tmp;
				assert(not std::isnan(tmp) & not std::isinf(tmp));
				*(_biases +w) = tmp;
			}

			if (LSTM) { //let all gates be biased towards open: better backprop
				for (int w=firstBiasIG_ID; w<firstBiasIG_ID+layerSize; w++){
					bufBiases >> tmp;
					assert(not std::isnan(tmp) & not std::isinf(tmp));
					*(_biases +w) = tmp;
				}

				for (int w=firstBiasFG_ID; w<firstBiasFG_ID+layerSize; w++){
					bufBiases >> tmp;
					assert(not std::isnan(tmp) & not std::isinf(tmp));
					*(_biases +w) = tmp;
				}

				for (int w=firstBiasOG_ID; w<firstBiasOG_ID+layerSize; w++){
					bufBiases >> tmp;
					assert(not std::isnan(tmp) & not std::isinf(tmp));
					*(_biases +w) = tmp;
				}
			}

			if (firstBiasWhiten>=0)
			for (int p=0 ; p<2; p++) for (int o=0 ; o<layerSize; o++){
				bufBiases >> tmp;
				assert(not std::isnan(tmp) & not std::isinf(tmp));
				_biases[firstBiasWhiten + p*layerSize_simd+o] = tmp;
			}
    }

    void save(std::ostringstream & outWeights,
			  std::ostringstream & outBiases,
			  Real* const _weights,
			  Real* const _biases) const
    {
			for (const auto & l : *(links))
				if(l not_eq nullptr)
					l->save(outWeights, _weights);

			for (int w=firstBias_ID; w<firstBias_ID+layerSize; w++)
				outBiases << *(_biases +w) << "\n";

			if (LSTM) { //let all gates be biased towards open: better backprop
				for (int w=firstBiasIG_ID; w<firstBiasIG_ID+layerSize; w++)
					outBiases << *(_biases +w) << "\n";

				for (int w=firstBiasFG_ID; w<firstBiasFG_ID+layerSize; w++)
					outBiases << *(_biases +w) << "\n";

				for (int w=firstBiasOG_ID; w<firstBiasOG_ID+layerSize; w++)
					outBiases << *(_biases +w) << "\n";
			}

			if (firstBiasWhiten>=0)
			for (int p=0 ; p<2; p++) for (int o=0 ; o<layerSize; o++)
				outBiases << _biases[firstBiasWhiten + p*layerSize_simd+o] << "\n";
    }

    void initializeWeights(mt19937* const gen, Real* const _weights, Real* const _biases) const
    {
				uniform_real_distribution<Real> dis(-sqrt(6./layerSize),sqrt(6./layerSize));

				if (firstBiasWhiten>=0) {
					for (int p=0 ; p<2; p++)
					for (int o=0 ; o<layerSize; o++)
					_biases[firstBiasWhiten + p*layerSize_simd+o] = Real(1==p);
				}

				if (input) return;

				printf("Initializing biases 1stBias %d (whitening %d) lS %d simd %d \n",
				firstBias_ID, firstBiasWhiten, layerSize, layerSize_simd);
				assert(layerSize>0 && layerSize_simd>0 && firstNeuron_ID>=0);

				for (const auto & l : *(links))
				    if(l not_eq nullptr) l->initialize(gen, _weights);

				if (not output) //let's try not having bias on output layer
				    for (int w=firstBias_ID; w<firstBias_ID+layerSize_simd; w++)
				        *(_biases +w) = dis(*gen);

				if (LSTM) { //let all gates be biased towards open: better backprop
				    assert(firstState_ID>=0 && firstBiasIG_ID>0 && firstBiasFG_ID>0 && firstBiasOG_ID>0);

				    for (int w=firstBiasIG_ID; w<firstBiasIG_ID+layerSize_simd; w++)
				        *(_biases +w) = dis(*gen) + 0.5;

				    for (int w=firstBiasFG_ID; w<firstBiasFG_ID+layerSize_simd; w++)
				        *(_biases +w) = dis(*gen) - 0.5;

				    for (int w=firstBiasOG_ID; w<firstBiasOG_ID+layerSize_simd; w++)
				        *(_biases +w) = dis(*gen) + 0.5;
				}
    }
};
