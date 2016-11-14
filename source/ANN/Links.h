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
	virtual void print() const = 0;
	virtual void initialize(mt19937* const gen, Real* const _weights) const = 0;
	virtual void restart(std::istringstream & buf, Real* const _weights) const = 0;
	virtual void save(std::ostringstream & buf, Real* const _weights) const = 0;
	virtual void propagate(const Activation* const netFrom, Activation* const netTo, const Real* const weights) const = 0;
	virtual void backPropagate(Activation* const netFrom, const Activation* const netTo, const Real* const weights, Real* const gradW) const =0;
	void orthogonalize(const int n0, Real* const _weights, int nOut, int nIn) const
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
	virtual void updateBatchStatistics(Real* const stds, Real* const avgs, const Activation* const act, const Real invN) {};
	virtual void applyBatchStatistics(Real* const stds, Real* const avgs, Real* const _weights, const Real invNm1) {};
	virtual void resetRunning() {};
	virtual void updateRunning(Activation* const act, const int counter) {};
	virtual void printRunning(int counter, std::ostringstream & oa, std::ostringstream & os) {};
};

class NormalLink: public Link
{
public:
	const int iW, nI, iI, nO, iO, nW;
    /*
     a link here is defined as link layer to layer:
     index iI along the network activation outvals representing the index of the first neuron of input layer
     the number nI of neurons of the input layer
     the index iO of the first neuron of the output layer
     the number of neurons in the output layer nO
     the index of the first weight iW along the weight vector
     the weights are all to all: so this link occupies space iW to (iW + nI*nO) along weight vector
     */
	NormalLink(int nI, int iI, int nO, int iO, int iW) : iW(iW), nI(nI), iI(iI), nO(nO), iO(iO), nW(nI*nO)
    {
		print();
		assert(nI>0 && nO>0 && iI>=0 && iO>=0 && iW>=0);
    }

    void print() const
    {
        cout << "Normal link: nInputs="<< nI << " IDinput=" << iI << " nOutputs=" << nO << " IDoutput" << iO << " IDweight" << iW << " nWeights" << nW << endl;
        fflush(0);
    }
    
    void initialize(mt19937* const gen, Real* const _weights) const override
    {
        const Real range = std::sqrt(6./(nO + nI));
        uniform_real_distribution<Real> dis(-range,range);
        //normal_distribution<Real> dis(0.,range);

        for (int w=iW ; w<(iW + nO*nI); w++) _weights[w] = dis(*gen);

        //orthogonalize(iW, _weights, nO, nI);
    }
    
    virtual void restart(std::istringstream & buf, Real* const _weights) const override
    {
        for (int w=iW ; w<(iW + nO*nI); w++) {
            Real tmp;
            buf >> tmp;
            assert(not std::isnan(tmp) & not std::isinf(tmp));
            _weights[w] = tmp;
        }
    }
    
    virtual void save(std::ostringstream & o, Real* const _weights) const override
    {
        o << std::setprecision(10);

        for (int w=iW ; w<(iW + nO*nI); w++) o << _weights[w];
    }

    virtual void propagate(const Activation* const netFrom, Activation* const netTo, const Real* const weights) const
    {
        const Real* __restrict__ const link_input = netFrom->outvals + iI;
        const Real* __restrict__ const link_weights = weights + iW;
        Real* __restrict__ const link_outputs = netTo->in_vals + iO;

        for (int i = 0; i < nI; i++)
        for (int o = 0; o < nO; o++) {
            assert(nO*i+o>=0 && nO*i+o<nW);
            link_outputs[o] += link_input[i] * link_weights[nO*i+o];
        }
    }
    
    virtual void backPropagate(Activation* const netFrom, const Activation* const netTo, const Real* const weights, Real* const gradW) const
    {
        const Real* __restrict__ const layer_input = netFrom->outvals + iI;
        const Real* __restrict__ const deltas = netTo->errvals + iO;
        const Real* __restrict__ const link_weights = weights + iW;
        Real* __restrict__ const link_errors = netFrom->errvals + iI;
        Real* __restrict__ const link_dEdW = gradW + iW;

        for (int i = 0; i < nI; i++)
        for (int o = 0; o < nO; o++) {
            assert(nO*i+o>=0 && nO*i+o<nW);
            link_dEdW[nO*i+o] += layer_input[i] * deltas[o];
            link_errors[i] += deltas[o] * link_weights[nO*i+o];
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
	const int iW, nI, iI, nO, iO, iC, iWI, iWF, iWO, nW;

	LinkToLSTM(int nI, int iI, int nO, int iO, int iC, int iW, int iWI, int iWF, int iWO) :
	iW(iW), nI(nI), iI(iI), nO(nO), iO(iO), iC(iC), iWI(iWI), iWF(iWF), iWO(iWO), nW(nI*nO) //i care nW per neuron, just for the asserts
	{
		print();
        assert(iWI==iW +nW);
        assert(iWF==iWI+nW);
        assert(iWO==iWF+nW);
		assert(iC>=0 && iWI>=0 && iWF>=0 && iWO>=0);
	}

    void print() const override
    {
        cout << "LSTM link: nInputs="<< nI << " IDinput=" << iI << " nOutputs=" << nO << " IDoutput" << iO << " IDcell" << iC << " IDweight" << iW << " nWeights" << nW << endl;
        fflush(0);
    }
    
    void initialize(mt19937* const gen, Real* const _weights) const override
    {
        const Real range = std::sqrt(6./(nO + nI));
        uniform_real_distribution<Real> dis(-range,range);
        //normal_distribution<Real> dis(0.,range);

        for (int w=iW ; w<(iW + nO*nI); w++)
            *(_weights +w) = dis(*gen);
        orthogonalize(iW, _weights, nO, nI);

        for (int w=iWI; w<(iWI+ nO*nI); w++)
            *(_weights +w) = dis(*gen);
        orthogonalize(iWI, _weights, nO, nI);

        for (int w=iWF; w<(iWF+ nO*nI); w++)
            *(_weights +w) = dis(*gen);
        orthogonalize(iWF, _weights, nO, nI);

        for (int w=iWO; w<(iWO+ nO*nI); w++)
            *(_weights +w) = dis(*gen);
        orthogonalize(iWO, _weights, nO, nI);
    }
    
    void restart(std::istringstream & buf, Real* const _weights) const override
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
    
    void save(std::ostringstream & o, Real* const _weights) const override
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
    
    void propagate(const Activation* const netFrom, Activation* const netTo, const Real* const weights) const override
    {
        Real* __restrict__ const inputs = netTo->in_vals + iO;
        Real* __restrict__ const inputI = netTo->iIGates + iC;
        Real* __restrict__ const inputF = netTo->iFGates + iC;
        Real* __restrict__ const inputO = netTo->iOGates + iC;
        const Real* __restrict__ const weights_toCell = weights + iW;
        const Real* __restrict__ const weights_toIgate = weights + iWI;
        const Real* __restrict__ const weights_toFgate = weights + iWF;
        const Real* __restrict__ const weights_toOgate = weights + iWO;
        const Real* __restrict__ const link_input = netFrom->outvals + iI;

        for (int i = 0; i < nI; i++)
        for (int o = 0; o < nO; o++) {
            assert(nO*i+o>=0 && nO*i+o<nW);
            inputs[o] += link_input[i] * weights_toCell[nO*i + o];
            inputI[o] += link_input[i] * weights_toIgate[nO*i + o];
            inputF[o] += link_input[i] * weights_toFgate[nO*i + o];
            inputO[o] += link_input[i] * weights_toOgate[nO*i + o];
        }
    }
    
    void backPropagate(Activation* const netFrom, const Activation* const netTo, const Real* const weights, Real* const gradW) const override
    {
        const Real* __restrict__ const deltaI = netTo->eIGates +iC;
        const Real* __restrict__ const deltaF = netTo->eFGates +iC;
        const Real* __restrict__ const deltaO = netTo->eOGates +iC;
        const Real* __restrict__ const deltaC = netTo->eMCell +iC;
        const Real* __restrict__ const layer_input = netFrom->outvals + iI;
        Real* __restrict__ const link_errors = netFrom->errvals + iI;
        const Real* __restrict__ const w_toOgate = weights + iWO;
        const Real* __restrict__ const w_toFgate = weights + iWF;
        const Real* __restrict__ const w_toIgate = weights + iWI;
        const Real* __restrict__ const w_toCell = weights + iW;
        Real* __restrict__ const dw_toOgate = gradW + iWO;
        Real* __restrict__ const dw_toFgate = gradW + iWF;
        Real* __restrict__ const dw_toIgate = gradW + iWI;
        Real* __restrict__ const dw_toCell = gradW + iW;

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
};


class LinkToConv2D : public Link
{
public:
	const int iW, nI, iI, nO, iO;
    const int inputWidth, inputHeight, inputDepth;
    const int filterWidth, filterHeight;
    const int outputWidth, outputHeight, outputDepth;
	const int strideX, strideY, padX, padY;
	const int nW;

	LinkToConv2D(int nI, int iI, int nO, int iO, int iW,
				int inW, int inH, int inD,
    			int fW, int fH, int fN, int outW, int outH,
				int sX=1, int sY=1, int pX=0, int pY=0) :
	iW(iW), nI(nI), iI(iI), nO(nO), iO(iO), inputWidth(inW), inputHeight(inH), inputDepth(inD), filterWidth(fW), filterHeight(fH),
	outputDepth(fN), outputWidth(outW), outputHeight(outH), strideX(sX), strideY(sY), padX(pX), padY(pY), nW(fW*fH*fN*inD)
	{
		assert(nW>0);
		assert(inputWidth*inputHeight*inputDepth == nI);
		assert(outputWidth*outputHeight*outputDepth == nO);
		const int inW_withPadding = (outputWidth-1)*strideX + filterWidth;
		const int inH_withPadding = (outputHeight-1)*strideY + filterHeight;
		//this class prescribes the bottom padding, let's figure out if the top one makes sense
		// inW_withPadding = inputWidth + bottomPad + topPad (where bottomPad = padX,padY)
		//first: All pixels of input are covered. topPad must be >=0, and stride leq than filter size
		assert(inW_withPadding-(inputWidth+padX) >= 0);
		assert(inH_withPadding-(inputHeight+padY) >= 0);
		assert(filterWidth >= strideX && filterHeight >= strideY);
		//second condition: do not feed an output pixel only with padding
		assert(inW_withPadding-(inputWidth+padX) < filterWidth);
		assert(inH_withPadding-(inputHeight+padY) < filterHeight);
		assert(padX < filterWidth && padY < filterHeight);
	}

    void print() const override
    {
        cout << nI << " " << iI << " " << nO << " " << iO << " " << iW << endl;
        fflush(0);
    }
    
    void initialize(mt19937* const gen, Real* const _weights) const override
    {
        const int nAdded = filterWidth*filterHeight*inputDepth;
        const Real range = std::sqrt(6./(nAdded+outputDepth));
        uniform_real_distribution<Real> dis(-range,range);
        //normal_distribution<Real> dis(0.,range);
        assert(outputDepth*nAdded == nW);
        for (int w=iW ; w<(iW + outputDepth*nAdded); w++)
            *(_weights +w) = dis(*gen);

        orthogonalize(iW, _weights, outputDepth, nAdded);
    }
    
    void restart(std::istringstream & buf, Real* const _weights) const override
    {
        const int nAdded = filterWidth*filterHeight*inputDepth;
        for (int w=iW ; w<(iW + outputDepth*nAdded); w++) {
            Real tmp;
            buf >> tmp;
            assert(not std::isnan(tmp) & not std::isinf(tmp));
            *(_weights +w) = tmp;
        }
    }
    
    void save(std::ostringstream & o, Real* const _weights) const override
    {
        o << std::setprecision(10);

        const int nAdded = filterWidth*filterHeight*inputDepth;
        for (int w=iW ; w<(iW + outputDepth*nAdded); w++)
            o << *(_weights +w);
    }
    
    void propagate(const Activation* const netFrom, Activation* const netTo, const Real* const weights) const override
    {
        Real* __restrict__ const link_outputs = netTo->in_vals + iO;
        const Real* __restrict__ const link_inputs = netFrom->outvals + iI;
        const Real* __restrict__ const link_weights = weights + iW;

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
    
    void backPropagate(Activation* const netFrom, const Activation* const netTo, const Real* const weights, Real* const gradW) const override
    {
        const Real* __restrict__ const link_inputs = netFrom->outvals + iI;
        const Real* __restrict__ const deltas = netTo->errvals + iO;
        const Real* __restrict__ const link_weights = weights + iW;
        Real* __restrict__ const link_errors = netFrom->errvals + iI;
        Real* __restrict__ const link_dEdW = gradW + iW;

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
};

class WhiteningLink : public Link
{
    vector<Real> runningAvg, runningStd;
public:
	const int iW, nI, iI, nO, iO, nW;
	WhiteningLink(int nI, int iI, int nO, int iO, int iW) : iW(iW), nI(nI), iI(iI), nO(nO), iO(iO), nW(2*nI)
    { 
        print();
        assert(nI==nO && iI+nI==iO);
    }
	void initialize(mt19937* const gen, Real* const _weights) const override
    {
        //set to 1 the scaling factor
        Real* const my_weights = _weights +iW;
        for (int p=0 ; p<2; p++)
        for (int o=0 ; o<nO; o++)
            my_weights[p*nO+o] = Real(1==p);
    }

    void print() const override
    {
        cout << "Whitening link: nInputs="<< nI << " IDinput=" << iI << " nOutputs=" << nO << " IDoutput" << iO << " IDweight" << iW << endl;
        fflush(0);
    }
    
    void updateBatchStatistics(Real* const stds, Real* const avgs, const Activation* const act, const Real invN) override
	{
    	die("WRONG\n");
		for (int k=0; k<nO; k++) {
			const Real delta = act->outvals[k+iI] - avgs[k+iO];
			avgs[k+iO] += delta*invN;
			stds[k+iO] += delta*(act->outvals[k+iI] - avgs[k+iO]);
		}
	}
    void applyBatchStatistics(Real* const stds, Real* const avgs, Real* const _weights, const Real invNm1)
    {
    	die("WRONG\n");
        Real* const link_means = _weights +iW;
        Real* const link_vars = _weights +iW +nO;
        const Real eta = 0.01;
        const Real _eta = 1. - eta;
        for (int k=0; k<nO; k++) {
			link_means[k] = _eta*link_means[k] + eta*avgs[k+iO];
			const Real tmp = std::max(stds[k+iO]*invNm1, std::numeric_limits<Real>::epsilon());
			link_vars[k] = _eta*link_vars[k] + eta*tmp;
			avgs[k+iO] = 0;
			stds[k+iO] = 0;
		}
    }

    void restart(std::istringstream & buf, Real* const _weights) const override
    {
        for (int w=iW ; w<(iW + nO*2); w++) {
            Real tmp;
            buf >> tmp;
            assert(not std::isnan(tmp) & not std::isinf(tmp));
            *(_weights +w) = tmp;
        }
    }
    
    void save(std::ostringstream & o, Real* const _weights) const override
    {
        o << std::setprecision(10);

        for (int w=iW ; w<(iW + nO*2); w++)
            o << *(_weights +w);
    }
    
    void propagate(const Activation* const netFrom, Activation* const netTo, const Real* const weights) const override
    {
        die("You really should not be able to get here.\n");
    }
    
    void backPropagate(Activation* const netFrom, const Activation* const netTo, const Real* const weights, Real* const gradW) const override
    {
        die("You really should not be able to get here as well.\n");
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
    int layerSize;
	int firstNeuron_ID; //recurrPos, normalPos;
    int firstState_ID;
    int firstBias_ID;
    int firstBiasIG_ID, firstBiasFG_ID, firstBiasOG_ID;
    int firstBiasWhiten, firstBiasFG_ID, firstBiasOG_ID;
    int layerWidth, layerHeight, layerDepth;
    int padWidth, padHeight, featsWidth, featsHeight, featsNumber, strideWidth, strideHeight;
    vector<int> linkedTo;
    vector<Link*> * links;

    Graph() :
	input(false), output(false), RNN(false), LSTM(false), Conv2D(false), normalize(false),
	layerSize(0), firstNeuron_ID(0), firstState_ID(0), firstBias_ID(0), firstBiasWhiten(-1),
	firstBiasIG_ID(0), firstBiasFG_ID(0), firstBiasOG_ID(0), //LSTM
	layerWidth(0), layerHeight(0), layerDepth(0), padWidth(0), padHeight(0), //Conv2D
	featsWidth(0), featsHeight(0), featsNumber(0), strideWidth(0), strideHeight(0)
    {
    	links = new vector<Link*>();
    }
    
    ~Graph()
    {
        for (auto& link : *links)
        	_dispose_object(link);
        _dispose_object(links);
    }
    
    void initializeWeights(mt19937* const gen, Real* const _weights, Real* const _biases) const
    {
        uniform_real_distribution<Real> dis(-sqrt(6./layerSize),sqrt(6./layerSize));

        for (const auto & l : *(links))
            if(l not_eq nullptr) l->initialize(gen, _weights);

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

        if (firstBiasWhiten>0) {
        	for (int p=0 ; p<2; p++)
				for (int o=0 ; o<layerSize; o++)
					_biases[firstBiasWhiten + p*layerSize+o] = Real(1==p);
        }
    }
};
