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
    int nI, iI, nO, iO, iW;
    
    Link(int nI, int iI, int nO, int iO, int iW) : nI(nI), iI(iI), nO(nO), iO(iO), iW(iW)
    { }
    
    Link() : nI(0), iI(0), nO(0), iO(0), iW(0)
    { }
    
    void set(int _nI, int _iI, int _nO, int _iO, int _iW);
    
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
    int iC, iWI, iWF, iWO;

    LinkToLSTM() { }

	LinkToLSTM(int nI, int iI, int nO, int iO, int iC, int iW, int iWI, int iWF, int iWO) :
		Link(nI, iI, nO, iO, iW), iC(iC), iWI(iWI), iWF(iWF), iWO(iWO)
	    { }

    void print() const;

    void set(int _nI, int _iI, int _nO, int _iO, int _iC, int _iW, int _iWI, int _iWF, int _iWO);

    Real backPropagate(const Activation* const lab, const int ID_NeuronFrom, const Real* const weights) const override;

    void propagate(Real* const inputs, const Activation* const lab, const int ID_NeuronTo, const Real* const weights) const override;

    Real propagate(const Activation* const lab, const int ID_NeuronTo, const Real* const weights) const override;

    void computeGrad(const Activation* const activation_From, const Activation* const activation_To, Real* const dEdW) const override;

    void addUpGrads(const Activation* const activation_From, const Activation* const activation_To, Real* const dEdW) const override;

    void initialize(uniform_real_distribution<Real>& dis, mt19937* const gen, Real* const _weights) const override;
};

class LinkToCNN : public Link
{
public:
    
    //maxPullSize = the size of the on side of the (square) maxPull
    int kernelWidth, kernelHeight, stride, zeroPadding, imageWidth, imageHeight, imagesNum, kernelNum, maxPullSize, kernelDepth;

    LinkToCNN() { }

	LinkToCNN(int nI, int iI, int nO, int iO, int iW, int kW, int kH, int st, int zP, int imW, int imH, int iN, int kN, int mPS, int kD) :
		Link(nI, iI, nO, iO, iW), kernelWidth(kW), kernelHeight(kH), stride(st), zeroPadding(zP), imageWidth(imW), imageHeight(imH), imagesNum(iN), kernelNum(kN), maxPullSize(mPS), kernelDepth(kD)
	    { }

    Real backPropagate(const Activation* const lab, const int ID_NeuronFrom, const Real* const weights) const;

    Real propagate(const Activation* const lab, const int ID_NeuronTo, const Real* const weights) const;

    void propagate(Real* const inputs, const Activation* const lab, const int ID_NeuronTo, const Real* const weights) const;

    void computeGrad(const Activation* const activation_From, const Activation* const activation_To, Real* const dEdW) const;

    void addUpGrads(const Activation* const activation_From, const Activation* const activation_To, Real* const dEdW) const;

    void initialize(uniform_real_distribution<Real>& dis, mt19937* const gen, Real* const _weights) const;
};

struct Graph //misleading, this is just the graph for a single layer
{
    bool first, last;
    int recurrSize, normalSize, recurrPos, normalPos;
    
    Link *rl_recurrent, *nl_recurrent;
    vector<Link*> *rl_inputs_vec, *rl_outputs_vec, *nl_inputs_vec, *nl_outputs_vec;
    
    int indState, wPeep, biasHL, biasIN, biasIG, biasFG, biasOG;

    Graph()
    : first(false), last(false), recurrSize(0), normalSize(0), recurrPos(0),  normalPos(0),
	  wPeep(0), indState(0), biasHL(0), biasIN(0), biasIG(0), biasFG(0), biasOG(0),
	  rl_recurrent(nullptr), nl_recurrent(nullptr)
    {
        rl_inputs_vec = new vector<Link*>();
        rl_outputs_vec = new vector<Link*>();
        nl_inputs_vec = new vector<Link*>();
        nl_outputs_vec = new vector<Link*>();
    }
    
    ~Graph()
    {
        _dispose_object( rl_recurrent);
        _dispose_object( nl_recurrent);
        for (auto & link : *rl_inputs_vec)
        	_dispose_object( link);
        _dispose_object( rl_inputs_vec);
        for (auto & link : *rl_outputs_vec)
        	_dispose_object ( link);
        _dispose_object( rl_outputs_vec);
        for (auto & link : *nl_inputs_vec)
        	_dispose_object ( link);
        _dispose_object( nl_inputs_vec);
        for (auto & link : *nl_outputs_vec)
        	_dispose_object( link);
        _dispose_object( nl_outputs_vec);
    }
    
    void initializeWeights(mt19937* const gen, Real* const _weights, Real* const _biases) const;
};
