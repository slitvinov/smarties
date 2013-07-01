/*
 *  Network.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 20.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <cmath>

#include "Network.h"
#include "../ErrorHandling.h"

//#include <boost/numeric/ublas/matrix.hpp>


using namespace ErrorHandling;

//namespace ANN
//{
	Network::Network(vector<int> layerSize, double eta, double alpha) :
	nInputs(layerSize.front()), nOutputs(layerSize.back()), nLayers(layerSize.size()), eta(eta), alpha(alpha), rng(0)
	{
		Layer* first = new Layer(nInputs, new Tanh);
		layers.push_back(first);
		
		for (int i=1; i<nLayers; i++)
		{
			Layer* l = (i == nLayers - 1) ? new Layer(layerSize[i],  new Linear) : new Layer(layerSize[i],  new Tanh);
			Layer* prev = layers.back();
			layers.push_back(l);
			prev->connect(l, &rng);
		}

		inputs.resize(nInputs);
		outputs.resize(nOutputs);
		errors.resize(nOutputs);
		
		first->connect2inputs(inputs);
		layers.back()->connect2outputs(outputs);
		layers.back()->connect2errors(errors);
	}
	
	void Network::predict(const vector<double>& inputs, vector<double>& outputs)
	{
		for (int i=0; i<nInputs; i++)
			*(this->inputs[i]) = inputs[i];
		
		for (int i=0; i<nLayers; i++)
			layers[i]->propagate();
		
		for (int i=0; i<nOutputs; i++)
			outputs[i] = *(this->outputs[i]);
	}
	
	void Network::improve(const vector<double>& errors)
	{
		for (int i=0; i<nOutputs; i++)
			*(this->errors[i]) = errors[i];
	
		for (int i=nLayers-1; i>=0; i--)
			layers[i]->backPropagate();
		
		for (int i=0; i<nLayers; i++)
			layers[i]->adjust(eta, alpha);
	}
	
	
	Layer::Layer(int nNeurons, ActivationFunction* func) : nNeurons(nNeurons+1)
	{
		neurons.resize(this->nNeurons);
		for (int i=0; i < this->nNeurons; i++)
			neurons[i] = new Neuron(func);
		neurons[this->nNeurons-1]->ival = -1;
	}
	
	void Layer::connect(Layer* next, RNG* rng)
	{
		for (int i=0; i<nNeurons; i++)
			for (int j=0; j<next->nNeurons-1; j++)
			{
				Link* lnk = new Link;
				lnk->neuronFrom = neurons[i];
				lnk->neuronTo   = next->neurons[j];
				lnk->w = rng->uniform(-1/(2.0*nNeurons), 1/(2.0*nNeurons));
				lnk->val    = 0;
				lnk->prevDw = 0;
				
				neurons[i]->outLinks.push_back(lnk);
				neurons[i]->hasOutputs = true;
				next->neurons[j]->inLinks.push_back(lnk);
				next->neurons[j]->hasInputs = true;
			}
		
	}
	
	void Layer::connect2inputs(vector<double*>& vals)
	{
		for (int i=0; i<nNeurons; i++)
			vals[i] = &(neurons[i]->ival);
	}

	void Layer::connect2outputs(vector<double*>& vals)
	{
		for (int i=0; i<nNeurons; i++)
			vals[i] = &(neurons[i]->oval);
	}
	
	void Layer::connect2errors(vector<double*>& errs)
	{
		for (int i=0; i<nNeurons; i++)
			errs[i] = &(neurons[i]->err);
	}
	
	void Layer::propagate()
	{
		for (int i=0; i<nNeurons; i++)
			neurons[i]->exec();
	}
	
	void Layer::backPropagate()
	{
		for (int i=0; i<nNeurons; i++)
			neurons[i]->backExec();
	}
	
	void Layer::adjust(double eta, double alpha)
	{
		for (int i=0; i<nNeurons; i++)
			neurons[i]->adjust(eta, alpha);
	}
	
	Neuron::Neuron(ActivationFunction* func) :
	hasInputs(false), hasOutputs(false), func(func) { };
	
	void Neuron::exec()
	{
		if (hasInputs)
		{
			ival = 0;
			for (int i=0; i<inLinks.size(); i++)
				ival += inLinks[i]->val * inLinks[i]->w;
		}
		
		oval = func->eval(ival);
		
		if (hasOutputs)
		{
			for (int i=0; i<outLinks.size(); i++)
				outLinks[i]->val = oval;
		}
	}
	
	void Neuron::backExec()
	{
		if (hasOutputs)
		{
			err = 0;
			for (int i=0; i<outLinks.size(); i++)
				err += outLinks[i]->val * outLinks[i]->w;
		}
		
		err = func->evalDiff(ival) * err;
		
		if (hasInputs)
		{
			for (int i=0; i<inLinks.size(); i++)
				inLinks[i]->val = err;
		}
	}
	
	void Neuron::adjust(double eta, double alpha)
	{
		if (hasInputs)
		{
			for (int i=0; i<inLinks.size(); i++)
			{
				Neuron* prev = inLinks[i]->neuronFrom;
				inLinks[i]->w += -eta * err * prev->oval + alpha * inLinks[i]->prevDw;
				inLinks[i]->prevDw = -eta * err * prev->oval;
			}
		}
	}	
//}



