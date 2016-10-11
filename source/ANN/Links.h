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
    const int nI, iI, nO, iO, iW;
    
    Link(int nI, int iI, int nO, int iO, int iW) : nI(nI), iI(iI), nO(nO), iO(iO), iW(iW)
    { }
    
    ///Link() : nI(0), iI(0), nO(0), iO(0), iW(0){ }
    
    //void set(int _nI, int _iI, int _nO, int _iO, int _iW);
    
    void print() const;

    virtual Real backPropagate(const Activation* const lab, const int ID_NeuronFrom, const Real* const weights) const;

    virtual Real propagate(const Activation* const lab, const int ID_NeuronTo, const Real* const weights) const;

    virtual void propagate(Real* const inputs, const Activation* const lab, const int ID_NeuronTo, const Real* const weights) const;

    virtual void computeGrad(const Activation* const activation_From, const Activation* const activation_To, Real* const dEdW) const;

    virtual void addUpGrads(const Activation* const activation_From, const Activation* const activation_To, Real* const dEdW) const;

    virtual void initialize(uniform_real_distribution<Real>& dis, mt19937* const gen, Real* const _weights) const;

    void orthogonalize(const int n0, Real* const _weights) const;
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
    const int iC, iWI, iWF, iWO;

    //LinkToLSTM() { }

	LinkToLSTM(int nI, int iI, int nO, int iO, int iC, int iW, int iWI, int iWF, int iWO) :
		Link(nI, iI, nO, iO, iW), iC(iC), iWI(iWI), iWF(iWF), iWO(iWO)
	    { }

    void print() const;

    //void set(int _nI, int _iI, int _nO, int _iO, int _iC, int _iW, int _iWI, int _iWF, int _iWO);

    Real backPropagate(const Activation* const lab, const int ID_NeuronFrom, const Real* const weights) const override;

    void propagate(Real* const inputs, const Activation* const lab, const int ID_NeuronTo, const Real* const weights) const override;

    Real propagate(const Activation* const lab, const int ID_NeuronTo, const Real* const weights) const override;

    void computeGrad(const Activation* const activation_From, const Activation* const activation_To, Real* const dEdW) const override;

    void addUpGrads(const Activation* const activation_From, const Activation* const activation_To, Real* const dEdW) const override;

    void initialize(uniform_real_distribution<Real>& dis, mt19937* const gen, Real* const _weights) const override;
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
