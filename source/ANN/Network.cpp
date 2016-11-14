/*
 *  LSTMNet.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include "Network.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>
using namespace ErrorHandling;

void Network::build_normal_layer(Graph* const graph)
{
	vector<NormalLink*>* input_links = new vector<NormalLink*>();
	NormalLink* recurrent_link = nullptr;
	
    const int layerSize = graph->layerSize;
    const int nInputLinks = graph->linkedTo.size();
    const int firstNeuron_ID = nNeurons;
    assert(layerSize>0 && nInputLinks>0 && firstNeuron_ID>0);
    if(graph->output)
    	for(int i=0; i<layerSize; i++)
    		iOut.push_back(i+firstNeuron_ID);

    graph->firstNeuron_ID = firstNeuron_ID;
    nNeurons += layerSize; //move the counter
    graph->firstBias_ID = nBiases;
    nBiases += layerSize; //one bias per neuron

    for(int i = 0; i<nInputLinks; i++) {
    	assert(graph->linkedTo[i]>=0 || graph->linkedTo[i]<G.size());
    	Graph* const layerFrom = G[graph->linkedTo[i]];
    	const int firstNeuronFrom = layerFrom->firstNeuron_ID;
    	const int sizeLayerFrom = layerFrom->layerSize;
    	assert(firstNeuronFrom>=0 && firstNeuronFrom<firstNeuron_ID && sizeLayerFrom>0);
    	NormalLink* tmp = new NormalLink(sizeLayerFrom, firstNeuronFrom, layerSize, firstNeuron_ID, nWeights);
    	input_links->push_back(tmp);
		graph->links->push_back(tmp);
    	nWeights += sizeLayerFrom*layerSize; //fully connected
    }

    if (graph->RNN) { //connected  to past realization of current normal layer
        const int firstNeuronFrom = graph->normalize ? nNeurons : firstNeuron_ID;
    	recurrent_link = new NormalLink(layerSize, firstNeuronFrom, layerSize, firstNeuron_ID, nWeights);
		graph->links->push_back(recurrent_link);
    	nWeights += layerSize * layerSize;
    }

    Layer * l = nullptr;
    if (graph->output)
	l = new NormalLayer<Linear>(layerSize, firstNeuron_ID, graph->firstBias_ID, input_links, recurrent_link, graph->output);
    else
    l = new NormalLayer<SoftSign>(layerSize, firstNeuron_ID, graph->firstBias_ID, input_links, recurrent_link, graph->output);

    if (graph->output) printf( "Linear output\n");
    layers.push_back(l);
}

void Network::build_conv2d_layer(Graph* const graph)
{
	vector<LinkToConv2D*>* input_links = new vector<NormalLink*>();
    const int layerSize = graph->layerSize;
    const int nInputLinks = graph->linkedTo.size();
    const int firstNeuron_ID = nNeurons;
    assert(layerSize>0 && nInputLinks>0 && firstNeuron_ID>0);
    if(graph->output)
    	for(int i=0; i<layerSize; i++) iOut.push_back(i+firstNeuron_ID);

    graph->firstNeuron_ID = firstNeuron_ID;
    nNeurons += layerSize; //move the counter
    graph->firstBias_ID = nBiases;
    nBiases += layerSize; //one bias per neuron

    for(int i = 0; i<nInputLinks; i++) {
		assert(graph->linkedTo[i]>=0 || graph->linkedTo[i]<G.size());
		Graph* const layerFrom = G[graph->linkedTo[i]];
		const int firstNeuronFrom = layerFrom->firstNeuron_ID;
		const int sizeLayerFrom = layerFrom->layerSize;

		assert(firstNeuronFrom>=0 && firstNeuronFrom<firstNeuron_ID && sizeLayerFrom>0);
		LinkToConv2D* tmp = new LinkToConv2D(sizeLayerFrom, firstNeuronFrom, layerSize, firstNeuron_ID,
				nWeights, layerFrom->layerWidth, layerFrom->layerHeight, layerFrom->layerDepth,
				graph->featsWidth, graph->featsHeight, graph->featsNumber, graph->layerWidth, graph->layerHeight,
				graph->strideWidth, graph->strideHeight, graph->padWidth, graph->padHeight);
		input_links->push_back(tmp);
		graph->links->push_back(tmp);
		nWeights += graph->featsWidth*graph->featsHeight*graph->featsNumber*layerFrom->layerDepth;
    }

    Layer * l = nullptr;
    if (graph->output)
	l = new Conv2DLayer<Linear>(layerSize, firstNeuron_ID, graph->firstBias_ID, input_links, graph->output);
    else
    l = new Conv2DLayer<SoftSign>(layerSize, firstNeuron_ID, graph->firstBias_ID, input_links, graph->output);

    if (graph->output) printf( "Linear output\n");
    layers.push_back(l);
}

void Network::build_whitening_layer(Graph* const graph)
{	
    const int layerSize = graph->layerSize;
    const int firstNeuron_ID = nNeurons;
    const int firstNeuronFrom = graph->firstNeuron_ID;
    assert(layerSize>0 && firstNeuron_ID>0);
    //move for the next
	graph->firstNeuron_ID = firstNeuron_ID;
    nNeurons += layerSize; //move the counter
    
    
    WhiteningLink* link = new WhiteningLink(layerSize, firstNeuronFrom, layerSize, firstNeuron_ID, nWeights);
    graph->links->push_back(link);
    nWeights += 4*layerSize; //fully connected

    Layer * l = new WhiteningLayer(layerSize, firstNeuron_ID, link, gen);
    layers.push_back(l);
}

void Network::build_LSTM_layer(Graph* const graph)
{
	vector<LinkToLSTM*>* input_links = new vector<LinkToLSTM*>();
	LinkToLSTM* recurrent_link = nullptr;
	
	const int layerSize = graph->layerSize;
	const int nInputLinks = graph->linkedTo.size();
    const int firstNeuron_ID = nNeurons;
    const int firstCell_ID = nStates;
	assert(layerSize>0 && nInputLinks>0 && firstNeuron_ID>0 && firstCell_ID>=0);
	if(graph->output)
		for(int i=0; i<layerSize; i++)
			iOut.push_back(i+firstNeuron_ID);

	graph->firstNeuron_ID = firstNeuron_ID;
	nNeurons += layerSize; //move the counter
	graph->firstState_ID = firstCell_ID;
	nStates += layerSize; //move the counter
	graph->firstBias_ID = nBiases;
	graph->firstBiasIG_ID = graph->firstBias_ID   + layerSize;
	graph->firstBiasFG_ID = graph->firstBiasIG_ID + layerSize;
	graph->firstBiasOG_ID = graph->firstBiasFG_ID + layerSize;
	nBiases += 4*layerSize; //one bias per neuron

	for(int i=0; i<nInputLinks; i++) {
		assert(graph->linkedTo[i]>=0 || graph->linkedTo[i]<G.size());
		Graph* const layerFrom = G[graph->linkedTo[i]];
		const int firstNeuronFrom = layerFrom->firstNeuron_ID;
    	const int sizeLayerFrom = layerFrom->layerSize;
		assert(firstNeuronFrom>=0 && firstNeuronFrom<firstNeuron_ID && sizeLayerFrom>0);
    	const int firstWeightIG = nWeights 		+ sizeLayerFrom*layerSize;
    	const int firstWeightFG = firstWeightIG + sizeLayerFrom*layerSize;
    	const int firstWeightOG = firstWeightFG + sizeLayerFrom*layerSize;
    	LinkToLSTM* tmp = new LinkToLSTM(sizeLayerFrom, firstNeuronFrom, layerSize, firstNeuron_ID,
    			firstCell_ID, nWeights, firstWeightIG, firstWeightFG, firstWeightOG);
    	input_links->push_back(tmp);
		graph->links->push_back(tmp);
		nWeights += sizeLayerFrom*layerSize*4; //fully connected, 4 units per cell
	}

	{ //connected  to past realization of current recurrent layer
		const int firstWeightIG = nWeights 		+ layerSize*layerSize;
		const int firstWeightFG = firstWeightIG + layerSize*layerSize;
		const int firstWeightOG = firstWeightFG + layerSize*layerSize;
        const int firstNeuronFrom = graph->normalize ? nNeurons : firstNeuron_ID;
        
		recurrent_link = new LinkToLSTM(layerSize, firstNeuronFrom, layerSize, firstNeuron_ID,
    			firstCell_ID, nWeights, firstWeightIG, firstWeightFG, firstWeightOG);
		graph->links->push_back(recurrent_link);
		nWeights += layerSize*layerSize*4; //fully connected, 4 units per cell
	}

	Layer * l = nullptr;

    if (graph->output) //cell output func, gates func, cell input func
	l = new LSTMLayer<Linear,SoftSigm,Linear>(
					layerSize, firstNeuron_ID, firstCell_ID, graph->firstBias_ID,
	        		graph->firstBiasIG_ID, graph->firstBiasFG_ID, graph->firstBiasOG_ID,
					input_links, recurrent_link, graph->output);
	else
	l = new LSTMLayer<Linear,SoftSigm,Tanh>(
					layerSize, firstNeuron_ID, firstCell_ID, graph->firstBias_ID,
					graph->firstBiasIG_ID, graph->firstBiasFG_ID, graph->firstBiasOG_ID,
					input_links, recurrent_link, graph->output);

	if (graph->output) printf("Linear LSTM output\n");
	layers.push_back(l);
}

void Network::addInput(const int size, const bool normalize)
{
	if(bBuilt) die("Cannot build the network multiple times\n");
	if(size<=0) die("Requested an empty input layer\n");
	bAddedInput = true;
	nInputs += size;
	Graph * g = new Graph();
    g->normalize = normalize;
	if (normalize) nLayers++;
	g->layerSize = size;
	g->input = true;
	G.push_back(g);
}

void Network::add2DInput(const int size[3], const bool normalize)
{
	if(bBuilt) die("Cannot build the network multiple times\n");
	if(size[0]<=0 || size[1]<=0 || size[2]<=0) die("Requested an empty input layer\n");
	const int flatSize = size[0] * size[1] * size[2];
	bAddedInput = true;
	nInputs += flatSize;
	Graph * g = new Graph();
    g->normalize = normalize;
	if (normalize) nLayers++;
	g->layerSize = flatSize;
	//2d input: depth is actually num of color channels: todo standard notation for eventual 3d?
	g->layerWidth = size[0];
	g->layerHeight = size[1];
	g->layerDepth = size[2];
	g->input = true;
	G.push_back(g);
}

void Network::addConv2DLayer(const int filterSize[3], const int outSize[3], const int padding[2], const int stride[2],
							 const bool normalize, vector<int> linkedTo, const bool bOutput)
{
	g->Conv2D = true;
	if(not bAddedInput) die("First specify an input\n");
	if(bBuilt) die("Cannot build the network multiple times\n");
	if(filterSize[0]<=0 || filterSize[1]<=0 || filterSize[2]<=0) die("Bad request for conv2D layer\n");
	if(outSize[0]<=0 || outSize[1]<=0 || outSize[2]<=0) die("Bad request for conv2D layer\n");
	assert(filterSize[0] >= stride[0] && filterSize[1] >= stride[1]);
	assert(padding[0] < filterSize[0] && padding[1] < filterSize[1]);

	nLayers++;
	Graph * g = new Graph();
	//default link is to previous layer:
	if(linkedTo.size() == 0) linkedTo.push_back(G.size()-1);
	g->normalize = normalize;
	if (normalize) nLayers++;
	g->layerSize = outSize[0] * outSize[1] * outSize[2];
	g->layerWidth = outSize[0]; g->layerHeight = outSize[1]; g->layerDepth = outSize[2];
	g->featsWidth = filterSize[0]; g->featsHeight = filterSize[1]; g->featsNumber = filterSize[2];
	g->padWidth = padding[0]; g->padHeight = padding[1]; g->strideWidth = stride[0]; g->strideHeight = stride[1];

	const int nTmpLayers = G.size();
	const int inputLinks = linkedTo.size();

	assert(inputLinks > 0);
	for(int i = 0; i<inputLinks; i++) {
		g->linkedTo.push_back(linkedTo[i]);
		if(linkedTo[i]<0 || linkedTo[i]>=nTmpLayers) die("Proposed link not available\n");
		if(layerFrom->layerWidth==0 || layerFrom->layerHeight==0 || layerFrom->layerDepth==0)
			die("Incompatible with 1D input, place 2D input or resize to 2D... how? TODO\n");

		const Graph* const layerFrom = G[graph->linkedTo[i]];
		const int inW_withPadding = (outSize[0]-1)*stride[0] + filterSize[0];
		const int inH_withPadding = (outSize[1]-1)*stride[1] + filterSize[1];
		assert(inW_withPadding - (layerFrom->layerWidth  + padding[0]) >= 0);
		assert(inH_withPadding - (layerFrom->layerHeight + padding[1]) >= 0);
		assert(inW_withPadding - (layerFrom->layerWidth  + padding[0]) < filterSize[0]);
		assert(inH_withPadding - (layerFrom->layerHeight + padding[1]) < filterSize[1]);
	}
	G.push_back(g);
	if (bOutput) {
		G.back()->output = true;
		nOutputs += size;
	}
}

void Network::addLayer(const int size, const string type, const bool normalize, vector<int> linkedTo, 
                       const bool bOutput)
{
	if(not bAddedInput) die("First specify an input\n");
	if(bBuilt) die("Cannot build the network multiple times\n");
	if(size<=0) die("Requested an empty layer\n");

	nLayers++;
	assert(!type.empty());
	Graph * g = new Graph();
	//default link is to previous layer:
	if(linkedTo.size() == 0) linkedTo.push_back(G.size()-1);

	if(type == "Normal" || type == "normal" || type == "Feedforward" || type == "feedforward")
	{} //default type
	else if (type == "RNN")		//probably not supported somewhere else in the code, but the kernels are fine
		g->RNN = true; //non-LSTM recurrent neural network
	else if (type == "LSTM")
		g->LSTM = true;
	else
		die("Unknown layer type \n");

    g->normalize = normalize;
	if (normalize) nLayers++;
	g->layerSize = size;
	const int nTmpLayers = G.size();
	const int inputLinks = linkedTo.size();

	assert(inputLinks > 0);
	for(int i = 0; i<inputLinks; i++) {
		g->linkedTo.push_back(linkedTo[i]);
		if(linkedTo[i]<0 || linkedTo[i]>=nTmpLayers)
			die("Proposed link not available\n");
	}
	G.push_back(g);
	if (bOutput) {
		G.back()->output = true;
		nOutputs += size;
	}
}

//counts up the number or neurons, weights, biases and creates the links and layers
void Network::build()
{
	if(bBuilt) die("Cannot build the network multiple times\n");
	bBuilt = true;
	if(nLayers<2) die("Put at least one hidden layer.\n");

	nNeurons = nBiases = nWeights = nStates = 0;
	for (auto & graph : G) {
		if(graph->input) {
            graph->firstNeuron_ID = nNeurons;
			nNeurons += graph->layerSize;
            if(graph->normalize) build_whitening_layer(graph);
			continue; //input layer is not a layer
		}
		assert(!graph->input);
		if (graph->LSTM) build_LSTM_layer(graph);
		else if (graph->Conv2D) build_conv2d_layer(graph);
		else build_normal_layer(graph);
        
        if(graph->normalize) build_whitening_layer(graph);
	}
	assert(layers.size() == nLayers);
	assert(iOut.size() == nOutputs);

	runningAvg.resize(nNeurons);
	runningStd.resize(nNeurons);

	for (int i=0; i<nAgents; ++i) {
		Mem * m = new Mem(nNeurons, nStates);
		mem.push_back(m);
	}
	dump_ID.resize(nAgents);

	grad = new Grads(nWeights,nBiases);
	_allocateClean(weights, nWeights)
	_allocateClean(biases,  nBiases)

	Vgrad.resize(nThreads);
	for (int i=0; i<nThreads; ++i)
		Vgrad[i] = new Grads(nWeights, nBiases);

	for (auto & graph : G) //TODO use for save/restart
		graph->initializeWeights(gen, weights, biases);

	updateFrozenWeights();
	printf("%d %d %d %d %d %d %d\n",nInputs, nOutputs, nLayers, nNeurons, nWeights, nBiases, nStates);
}

Network::Network(const Settings & settings) :
Pdrop(settings.nnPdrop), nInputs(0), nOutputs(0), nLayers(0), nNeurons(0), nWeights(0), nBiases(0), nStates(0),
nAgents(settings.nAgents), nThreads(settings.nThreads), allocatedFrozenWeights(false),
allocatedDroputWeights(false), backedUp(false), gen(settings.gen), bDump(not settings.bTrain),
bBuilt(false), bAddedInput(false)
{ }

void Network::predict(const vector<Real>& _input, vector<Real>& _output,
						vector<Activation*>& timeSeries, const int n_step,
						const Real* const _weights, const Real* const _biases, const Real noise) const
{
	assert(bBuilt && n_step<timeSeries.size() && n_step>=0);
    
	Activation* const currActivation = timeSeries[n_step];
	Activation* const prevActivation = n_step==0 ? nullptr : timeSeries[n_step-1];

    for (int j=0; j<nInputs; j++) *(currActivation->outvals +j) = _input[j];
    
    for (int j=0; j<nLayers; j++)
        layers[j]->propagate(prevActivation,currActivation,_weights,_biases, noise);
    
    assert(static_cast<int>(_output.size())==nOutputs);

    for (int i=0; i<nOutputs; i++)
        _output[i] = *(currActivation->outvals + iOut[i]);
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output,
						Activation* const prevActivation, Activation* const currActivation,
						const Real* const _weights, const Real* const _biases, const Real noise) const
{
	assert(bBuilt);

    for (int j=0; j<nInputs; j++) *(currActivation->outvals +j) = _input[j];

    for (int j=0; j<nLayers; j++)
        layers[j]->propagate(prevActivation,currActivation,_weights,_biases, noise);

    assert(static_cast<int>(_output.size())==nOutputs);

    for (int i=0; i<nOutputs; i++)
        _output[i] = *(currActivation->outvals + iOut[i]);
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output,
		Activation* const net, const Real* const _weights, const Real* const _biases, const Real noise) const
{
	assert(bBuilt);
    for (int j=0; j<nInputs; j++)
    	net->outvals[j] = _input[j];

    for (int j=0; j<nLayers; j++)
        layers[j]->propagate(net,_weights,_biases, noise);

    assert(static_cast<int>(_output.size())==nOutputs);

    for (int i=0; i<nOutputs; i++)
        _output[i] = net->outvals[iOut[i]];
}

void Network::backProp(vector<Activation*>& timeSeries, const Real* const _weights, Grads* const _grads) const
{
	assert(bBuilt);
	const int last = timeSeries.size()-1;

    for (int i=1; i<=nLayers; i++)
        layers[nLayers-i]->backPropagate(timeSeries[last-1],timeSeries[last],(Activation*)nullptr, _grads, _weights);

    for (int k=last-1; k>=1; k--) 
	for (int i=1; i<=nLayers; i++)
        layers[nLayers-i]->backPropagate(timeSeries[k-1],timeSeries[k],timeSeries[k+1], _grads, _weights);

    for (int i=1; i<=nLayers; i++)
        layers[nLayers-i]->backPropagate((Activation*)nullptr,timeSeries[0],timeSeries[1], _grads, _weights);
}

void Network::backProp(const vector<Real>& _errors,
		Activation* const net, const Real* const _weights, Grads* const _grads) const
{
	assert(bBuilt);
	net->clearErrors();
	assert(static_cast<int>(_errors.size())==nOutputs);
	for (int i=0; i<nOutputs; i++)
		net->errvals[iOut[i]] = _errors[i];

    for (int i=1; i<=nLayers; i++)
        layers[nLayers-i]->backPropagate(net, _grads, _weights);
}

void Network::clearErrors(vector<Activation*>& timeSeries) const
{
	for (int k=0; k<timeSeries.size(); k--)
		timeSeries[k]->clearErrors();
}

void Network::setOutputDeltas(vector<Real>& _errors, Activation* const net) const
{
	assert(bBuilt);
    assert(static_cast<int>(_errors.size())==nOutputs);
    for (int i=0; i<nOutputs; i++)
    	net->errvals[iOut[i]] = _errors[i];
}

void Network::updateFrozenWeights()
{
    if (allocatedFrozenWeights==false) {
        _allocateQuick(tgt_weights, nWeights)
        _allocateQuick(tgt_biases,   nBiases)
        allocatedFrozenWeights = true;
    }
    
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int j=0; j<nWeights; j++)
            *(tgt_weights + j) = *(weights + j);
        
        #pragma omp for nowait
        for (int j=0; j<nBiases; j++)
            *(tgt_biases + j) = *(biases + j);
    }
}

void Network::moveFrozenWeights(const Real alpha)
{
    if (allocatedFrozenWeights==false) updateFrozenWeights();
    const Real _alpha = 1. - alpha;

    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int j=0; j<nWeights; j++)
            *(tgt_weights + j) = *(tgt_weights + j)*_alpha + *(weights + j)*alpha;

        #pragma omp for nowait
        for (int j=0; j<nBiases; j++)
            *(tgt_biases + j) = *(tgt_biases + j)*_alpha + *(biases + j)*alpha;
    }
}

void Network::loadMemory(Mem * _M, Activation * _N) const
{
    std::swap(_N->outvals,_M->outvals);
    std::swap(_N->ostates,_M->ostates);
}

Activation* Network::allocateActivation() const
{
	return new Activation(nNeurons,nStates);
}

vector<Activation*> Network::allocateUnrolledActivations(int length) const
{
	vector<Activation*> ret(length);
	for (int j=0; j<length; j++)
		ret[j] = new Activation(nNeurons,nStates);
	return ret;
}

void Network::deallocateUnrolledActivations(vector<Activation*>* const ret) const
{
	for (auto & trash : *ret) _dispose_object(trash);
}

void Network::appendUnrolledActivations(vector<Activation*>* const ret, int length) const
{
	for (int j=0; j<=length; j++)
		ret->push_back(new Activation(nNeurons,nStates));
}

void Network::assignDropoutMask()
{
    if (Pdrop > 0) {
    	die("You are probably using dropout wrong anyway\n");
    	//ISSUES:
    	//- stupidly slow, no priority allocated to improve it
    	//- does not work with threads, just for laziness: there is no obstacle to make it compatible to threads

        assert(Pdrop>0 && Pdrop<1 && backedUp==false);
        if (allocatedDroputWeights==false) {
            _allocateQuick(weights_DropoutBackup, nWeights)
            allocatedDroputWeights = true;
        }
        //backup the weights
        swap(weights_DropoutBackup,weights);
        backedUp = true;
        //probability of having a true in the bernoulli distrib:
        Real Pkeep = 1. - Pdrop;
        Real fac = 1./Pkeep; //the others have to compensate

        bernoulli_distribution dis(Pkeep);
        for (int j=0; j<nWeights; j++) {
            bool res = dis(*gen);
            *(weights + j) = (res) ? *(weights_DropoutBackup + j)*fac : 0.;
        }
    }
}

void Network::removeDropoutMask()
{
    if (allocatedDroputWeights && backedUp) {
        swap(weights_DropoutBackup,weights);
        backedUp = false;
    }
}

void Network::checkGrads(const vector<vector<Real>>& inputs, int seq_len)
{
    if (seq_len<0) seq_len = inputs.size();
	assert(bBuilt);
    printf("Checking gradients\n");
    vector<int> errorPlacements(seq_len);
    vector<Real> partialResults(seq_len);
    vector<Activation*> timeSeries = allocateUnrolledActivations(seq_len);
    assert(timeSeries.size() == seq_len);
    vector<Real> res(nOutputs); //allocate net output
    
    const Real incr = 1e-6;
    
    uniform_real_distribution<Real> dis(0.,1.);
    for (int i=0; i<seq_len; i++) //figure out where to place some errors at random in outputs
        errorPlacements[i] = nOutputs*dis(*gen);
    
    Grads * g = new Grads(nWeights,nBiases);
    Grads * G = new Grads(nWeights,nBiases);
    clearErrors(timeSeries);

    for (int k=0; k<seq_len; k++) {
    	predict(inputs[k], res, timeSeries, k);
        vector<Real> errs(nOutputs,0);
        errs[errorPlacements[k]] = -1.;
        setOutputDeltas(errs, timeSeries[k]);
    }

    backProp(timeSeries, G);
    
    for (int w=0; w<nWeights; w++) {
        //1
        weights[w] += incr;
        for (int k=0; k<seq_len; k++) {
        	predict(inputs[k], res, timeSeries, k);
            partialResults[k] = -res[errorPlacements[k]];
        }
        //2
        weights[w] -= 2*incr;
        for (int k=0; k<seq_len; k++) {
        	predict(inputs[k], res, timeSeries, k);
            partialResults[k] += res[errorPlacements[k]];
        }
        //0
        weights[w] += incr;
        
        Real grad(0);
        for (int k=0; k<seq_len; k++) grad += partialResults[k];
        g->_W[w] = grad/(2.*incr);
        
        //const Real scale = fabs(*(biases+w));
        const Real scale = std::max(std::fabs(G->_W[w]),std::fabs(g->_W[w]));
        const Real err = (G->_W[w] - g->_W[w])/scale;
        if (fabs(err)>1e-5) cout <<"W"<<w<<" analytical:"<<G->_W[w]<<" finite:"<<g->_W[w]<<" error:"<<err<<endl;
    }
    
    for (int w=0; w<nBiases; w++) {
        //1
        *(biases+w) += incr;
        for (int k=0; k<seq_len; k++) {
			predict(inputs[k], res, timeSeries, k);
			partialResults[k] = -res[errorPlacements[k]];
		}
        //2
        *(biases+w) -= 2*incr;
        for (int k=0; k<seq_len; k++) {
			predict(inputs[k], res, timeSeries, k);
			partialResults[k] += res[errorPlacements[k]];
		}
        //0
        *(biases+w) += incr;
        
        Real grad(0);
        for (int k=0; k<seq_len; k++) grad += partialResults[k];
        g->_B[w] = grad/(2.*incr);
        
        //const Real scale = fabs(*(biases+w));
        const Real scale = std::max(std::fabs(G->_B[w]), std::fabs(g->_B[w]));
        const Real err = (G->_B[w] - g->_B[w])/scale;
        if (fabs(err)>1e-5) cout <<"B"<<w<<" analytical:"<<G->_B[w]<<" finite:"<<g->_B[w]<<" error:"<<err<<endl;
    }

    deallocateUnrolledActivations(&timeSeries);
    printf("\n"); fflush(0);
}

void Network::save(const string fname)
{
    printf("Saving into %s\n", fname.c_str());
    fflush(0);
    string nameBackup = fname + "_tmp";
    ofstream out(nameBackup.c_str());

    if (!out.good()) die("Unable to open save into file %s\n", fname.c_str());

    out.precision(20);
    out << nWeights << " "  << nBiases << " " << nLayers  << " " << nNeurons << endl;

    for (int i=0; i<nWeights; i++) {
        if (std::isnan(*(weights + i)) || std::isinf(*(weights + i))) {
            *(weights + i) = 0.0;
            out << 0.0 << "\n";
        } else {
            out << *(weights + i) << "\n";
        }
    }

    for (int i=0; i<nBiases; i++) {
       if (std::isnan(*(biases + i)) || std::isinf(*(biases + i))) {
           *(biases + i) = 0.0;
            out << 0.0 << "\n";
        } else {
            out << *(biases + i) << "\n";
        }
    }

    out.flush();
    out.close();

    //Prepare copying command
    string command = "cp " + nameBackup + " " + fname;

    //Submit the command to the system
    system(command.c_str());
}

void Network::dump(const int agentID)
{
    if (not bDump) return;
    char buf[500];
    sprintf(buf, "%07d", (int)dump_ID[agentID]);
    string nameNeurons  = "neuronOuts_" + to_string(agentID) + "_" + string(buf) + ".dat";
    string nameMemories = "cellStates_" + to_string(agentID) + "_" + string(buf) + ".dat";
    string nameOut_Mems = "out_states_" + to_string(agentID) + "_" + string(buf) + ".dat";
    {
        ofstream out(nameOut_Mems.c_str());
        if (!out.good()) die("Unable to open save into file %s\n", nameOut_Mems.c_str());
        for (int j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << " ";
        for (int j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << " ";
        out << "\n";
        out.close();
    }
    {
        ofstream out(nameNeurons.c_str());
        if (!out.good()) die("Unable to open save into file %s\n", nameNeurons.c_str());
        for (int j=0; j<nNeurons; j++) out << *(mem[agentID]->outvals +j) << " ";
        out << "\n";
        out.close();
    }
    {
        ofstream out(nameMemories.c_str());
        if (!out.good()) die("Unable to open save into file %s\n", nameMemories.c_str());
        for (int j=0; j<nStates;  j++) out << *(mem[agentID]->ostates +j) << " ";
        out << "\n";
        out.close();
    }
    dump_ID[agentID]++;
}

bool Network::restart(const string fname)
{
    string nameBackup = fname;

    ifstream in(nameBackup.c_str());
    debug1("Reading from %s\n", nameBackup.c_str());
    if (!in.good()) {
        error("WTF couldnt open file %s (ok keep going mofo)!\n", fname.c_str());
        return false;
    }

    int readTotWeights, readTotBiases, readNNeurons, readNLayers;
    in >> readTotWeights  >> readTotBiases >> readNLayers >> readNNeurons;

    if (readTotWeights != nWeights || readTotBiases != nBiases || readNLayers != nLayers || readNNeurons != nNeurons)
    die("Network parameters differ!");

    Real tmp;
    for (int i=0; i<nWeights; i++) {
        in >> tmp;
        if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
        *(weights + i) = tmp;
    }

    for (int i=0; i<nBiases; i++) {
        in >> tmp;
        if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
        *(biases + i) = tmp;
    }

    in.close();

    updateFrozenWeights();
    return true;
}
