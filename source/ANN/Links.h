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

using namespace std;

class Link
{
private:
	const int nW;
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
    const int iW, nI, iI, nO, iO;
    
    Link(int nI, int iI, int nO, int iO, int iW) : iW(iW), nI(nI), iI(iI), nO(nO), iO(iO), nW(nI*nO)
    {
		assert(nI>0 && nO>0 && iI>=0 && iO>=0 && iW>=0);
    }

    virtual void print() const;
    virtual void initialize(mt19937* const gen, Real* const _weights) const;
    virtual void restart(std::istringstream & buf, Real* const _weights) const;
    virtual void save(std::ostringstream & buf, const Real* const _weights) const;
    virtual void propagate(const Activation* const netFrom, const Activation* const netTo,
																	 const Real* const weights) const;
    virtual void backPropagate(const Activation* const netFrom, const Activation* const netTo,
											const Real* const weights, const Real* const gradW) const;
    void orthogonalize(const int n0, Real* const _weights, const int nOut=nO, const int nIn=nI) const;
};

class LinkToLSTM : public Link
{
private:
	const int nW;
public:
    /*
     if link is TO lstm, then the rules change a bit
     each LSTM block contains 4 neurons, one is the proper cell and then there are the 3 gates
     if a input signal is connected to one of the four, is also connected to the others
     thus we just need the index of the first weight for the 3 gates (could have skipped this, iWi = iW + nO*nI and so forth)
     additionally the LSTM contains a memory, contained in Activation->ostate
     memory and gates are treated differently than normal neurons, therefore are contained in separate array, and i keep track of the position with iC
     */
    const int iC, iWI, iWF, iWO;

	LinkToLSTM(int nI, int iI, int nO, int iO, int iC, int iW, int iWI, int iWF, int iWO) :
	Link(nI, iI, nO, iO, iW), iC(iC), iWI(iWI), iWF(iWF), iWO(iWO), nW(nI*nO) //i care nW per neuron, just for the asserts
	{
		assert(iC>=0 && iWI>=0 && iWF>=0 && iWO>=0);
	}

    void print() const override;
    void initialize(mt19937* const gen, Real* const _weights) const override;
    void restart(std::istringstream & buf, Real* const _weights) const override;
    void save(std::ostringstream & buf, const Real* const _weights) const override;
    void propagate(const Activation* const netFrom, const Activation* const netTo,
														 const Real* const weights) const override;
    void backPropagate(const Activation* const netFrom, const Activation* const netTo,
									const Real* const weights, const Real* const gradW) const override;
};


class LinkToConv2D : public Link
{
private:
	const int nW;
public:
    const int inputWidth, inputHeight, inputDepth;
    const int filterWidth, filterHeight;
    const int outputWidth, outputHeight, outputDepth;
	const int strideX, strideY, padX, padY;

	LinkToConv2D(int nI, int iI, int nO, int iO, int iW,
				int inW, int inH, int inD,
    			int fW, int fH, int fN, int outW, int outH,
				int sX=1, int sY=1, int pX=0, int pY=0) :
	Link(nI, iI, nO, iO, iW), inputWidth(inW), inputHeight(inH), inputDepth(inD), filterWidth(fW), filterHeight(fH),
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

    void initialize(mt19937* const gen, Real* const _weights) const override;
    void restart(std::istringstream & buf, Real* const _weights) const override;
    void save(std::ostringstream & buf, const Real* const _weights) const override;
    void propagate(const Activation* const netFrom, const Activation* const netTo,
														 const Real* const weights) const override;
    void backPropagate(const Activation* const netFrom, const Activation* const netTo,
									const Real* const weights, const Real* const gradW) const override;
};

class WhiteningLink : public Link
{
public:
	WhiteningLink(int nI, int iI, int nO, int iO, int iW) : Link(nI, iI, nO, iO, iW) { }
	void initialize(mt19937* const gen, Real* const _weights) const override;
    void restart(std::istringstream & buf, Real* const _weights) const override;
    void save(std::ostringstream & buf, const Real* const _weights) const override;
    void propagate(const Activation* const netFrom, const Activation* const netTo,
														 const Real* const weights) const override;
    void backPropagate(const Activation* const netFrom, const Activation* const netTo,
									const Real* const weights, const Real* const gradW) const override;
};

struct Graph //misleading, this is just the graph for a single layer
{
    bool input, output, RNN, LSTM;
    int layerSize;
	int firstNeuron_ID; //recurrPos, normalPos;
    int firstState_ID;
    int firstBias_ID;
    int firstBiasIG_ID, firstBiasFG_ID, firstBiasOG_ID;
    vector<int> linkedTo;
    Link * recurrent_link;//, *nl_recurrent;
    vector<Link*> * input_links_vec, * output_links_vec;

    Graph()
    : input(false), output(false), RNN(false), LSTM(false), layerSize(0), firstNeuron_ID(0),
	  firstState_ID(0), firstBias_ID(0), firstBiasIG_ID(0), firstBiasFG_ID(0), firstBiasOG_ID(0),
	  recurrent_link(nullptr)
    {
    	input_links_vec = new vector<Link*>();
    	output_links_vec = new vector<Link*>();
    }
    
    ~Graph()
    {
        _dispose_object(recurrent_link);
        for (auto& link : *input_links_vec)
        	_dispose_object(link);
        _dispose_object(input_links_vec);
        for (auto& link : *output_links_vec)
        	_dispose_object (link);
        _dispose_object(output_links_vec);
    }
    
    void initializeWeights(mt19937* const gen, Real* const _weights, Real* const _biases) const;
};
