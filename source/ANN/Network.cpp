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
    	assert(firstNeuronFrom>=0 && firstNeuronFrom<firstNeuron_ID);
    	Link* tmp = new Link(sizeLayerFrom, firstNeuronFrom, layerSize, firstNeuron_ID, nWeights);
    	graph->input_links_vec->push_back(tmp);
    	layerFrom->output_links_vec->push_back(tmp);
    	nWeights += sizeLayerFrom*layerSize; //fully connected
    }

    if (graph->RNN) { //connected  to past realization of current normal layer
    	Link* tmp=new Link(layerSize, firstNeuron_ID, layerSize, firstNeuron_ID, nWeights);
    	graph->recurrent_link = tmp;
    	nWeights += layerSize * layerSize;
    }

#ifndef _scaleR_
    const Response * f = (graph->output) ? new Response : new SoftSign;
    if (graph->output) printf( "Linear output\n");
#else
    const Response * f = new SoftSign;
    if (graph->output) printf( "Logic output\n");
#endif

    NormalLayer * l = new NormalLayer(layerSize, firstNeuron_ID, graph->firstBias_ID,
    		graph->input_links_vec, graph->recurrent_link, graph->output_links_vec, f, graph->output);
    layers.push_back(l);
}

void Network::build_LSTM_layer(Graph* const graph)
{
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
		assert(firstNeuronFrom>=0 && firstNeuronFrom<firstNeuron_ID);
    	const int firstWeightIG = nWeights 		+ sizeLayerFrom*layerSize;
    	const int firstWeightFG = firstWeightIG + sizeLayerFrom*layerSize;
    	const int firstWeightOG = firstWeightFG + sizeLayerFrom*layerSize;
    	Link* tmp = new LinkToLSTM(sizeLayerFrom, firstNeuronFrom, layerSize, firstNeuron_ID,
    			firstCell_ID, nWeights, firstWeightIG, firstWeightFG, firstWeightOG);
		graph->input_links_vec->push_back(tmp);
		layerFrom->output_links_vec->push_back(tmp);
		nWeights += sizeLayerFrom*layerSize*4; //fully connected, 4 units per cell
	}

	{ //connected  to past realization of current recurrent layer
		const int firstWeightIG = nWeights 		+ layerSize*layerSize;
		const int firstWeightFG = firstWeightIG + layerSize*layerSize;
		const int firstWeightOG = firstWeightFG + layerSize*layerSize;
    	Link* tmp = new LinkToLSTM(layerSize, firstNeuron_ID, layerSize, firstNeuron_ID,
    			firstCell_ID, nWeights, firstWeightIG, firstWeightFG, firstWeightOG);
		graph->recurrent_link = tmp;
		nWeights += layerSize*layerSize*4; //fully connected, 4 units per cell
	}

#if  1
        #ifndef _scaleR_
        const Response * fI = (graph->output) ? new Response : new SoftSign2;
        const Response * fG = new SoftSigm;
        const Response * fO = (graph->output) ? new Response : new Tanh;
        if (graph->output) printf("Linear output\n");
        #else
        const Response * fI = new SoftSign2;
        const Response * fG = new SoftSigm;
        const Response * fO = new Tanh;
        if (graph->output) printf("Logic output\n");
        #endif
#else
        #ifndef _scaleR_
        const Response * fI = (graph->output) ? new Response : new Tanh2;
        const Response * fG = new Sigm;
        const Response * fO = (graph->output) ? new Response : new Tanh;
        if (graph->output) printf("Linear output\n");
        #else
        const Response * fI = new Tanh2;
        const Response * fG = new Sigm;
        const Response * fO = new Tanh;
        if (graph->output) printf("Logic output\n");
        #endif
#endif
        
        NormalLayer * l = new LSTMLayer(layerSize, firstNeuron_ID, firstCell_ID, graph->firstBias_ID,
        		graph->firstBiasIG_ID, graph->firstBiasFG_ID, graph->firstBiasOG_ID, graph->input_links_vec,
				graph->recurrent_link, graph->output_links_vec, fI, fG, fO, graph->output);
        layers.push_back(l);
}

void Network::addInput(const int size)
{
	if(bBuilt) die("Cannot build the network multiple times\n");
	if(size<=0) die("Requested an empty input layer\n");
	bAddedInput = true;
	nInputs += size;
	Graph * g = new Graph();
	g->layerSize = size;
	g->input = true;
	G.push_back(g);
}

void Network::addLayer(const int size, const string type, vector<int> linkedTo, const bool bOutput)
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
			nNeurons += graph->layerSize;
			continue; //input layer is not a layer
		}
		assert(!graph->input);
		if (graph->LSTM) build_LSTM_layer(graph);
		else build_normal_layer(graph);
	}
	assert(layers.size() == nLayers);
	assert(iOut.size() == nOutputs);

	for (int i=0; i<nAgents; ++i) {
		Mem * m = new Mem(nNeurons, nStates);
		mem.push_back(m);
	}
	dump_ID.resize(nAgents);
	allocateSeries(3);

	grad = new Grads(nWeights,nBiases);
	_allocateClean(weights, nWeights)
	_allocateClean(biases,  nBiases)

	if (nThreads>1) {
		Vgrad.resize(nThreads);
		for (int i=0; i<nThreads; ++i)
			Vgrad[i] = new Grads(nWeights, nBiases);
	} else {
		_grad    = new Grads(nWeights, nBiases);
	}

	for (auto & graph : G)
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

void Network::predict(const vector<Real>& _input, vector<Real>& _output, const Activation* const _M, Activation* const _N, const Real* const _weights, const Real* const _biases) const
{
	assert(bBuilt);
    for (int j=0; j<nInputs; j++)
        *(_N->outvals +j) = _input[j];
    
    for (int j=0; j<nLayers; j++)
        layers[j]->propagate(_M,_N,_weights,_biases);
    
    assert(static_cast<int>(_output.size())==nOutputs);

    for (int i=0; i<nOutputs; i++) {
        *(_N->errvals + iOut[i]) = 0.;
        _output[i] = *(_N->outvals + iOut[i]);
    }
}

void Network::predict(const vector<Real>& _input, vector<Real>& _output, Activation* const _N, const Real* const _weights, const Real* const _biases) const
{
	assert(bBuilt);
    for (int j=0; j<nInputs; j++)
    *(_N->outvals +j) = _input[j];
    
    for (int j=0; j<nLayers; j++)
        layers[j]->propagate(_N,_weights,_biases);
    
    assert(static_cast<int>(_output.size())==nOutputs);

    for (int i=0; i<nOutputs; i++) {
        *(_N->errvals + iOut[i]) = 0.;
        _output[i] = *(_N->outvals + iOut[i]);
    }
}

void Network::setOutputErrors(vector<Real>& _errors, Activation* const _N)
{
	assert(bBuilt);
    assert(static_cast<int>(_errors.size())==nOutputs);
    for (int i=0; i<nOutputs; i++) {
        *(_N->errvals + iOut[i]) = _errors[i];
    }
}

//No time dependencies
void Network::computeDeltas(Activation* const _series, const Real* const _weights, const Real* const _biases) const
{
	assert(bBuilt);
    for (int i=1; i<=nLayers; i++)
        layers[nLayers-i]->backPropagateDelta(_series,_weights,_biases);
}

void Network::computeGrads(const Activation* const lab, Grads* const _Grad, const Real* const _weights) const
{
	assert(bBuilt);
    for (int i=0; i<nLayers; i++)
    layers[i]->backPropagateGrads(lab,_Grad, _weights); //grad is zero-equal
}

void Network::computeAddGrads(const Activation* const lab, Grads* const _Grad, const Real* const _weights) const
{
	assert(bBuilt);
    for (int i=0; i<nLayers; i++)
    layers[i]->backPropagateAddGrads(lab,_Grad, _weights);  //grad is add-equal
}

//Back Prop Through Time:
//compute deltas: start from last activation, propagate deltas back to first
// ISSUES:  this array of activations disturbs me deeply as it is deeply inelegant
//			but was the most robust and easiest path towards parallelism
void Network::computeDeltasSeries(vector<Activation*>& _series, const int first, const int last, const Real* const _weights, const Real* const _biases) const
{
	assert(bBuilt);
#ifdef _BPTT_
    for (int i=1; i<=nLayers; i++) {
        layers[nLayers-i]->backPropagateDeltaLast(series[last-1],series[last],_weights,_biases);
    }
    
    for (int k=last-1; k>=first+1; k--) {
        for (int i=1; i<=nLayers; i++)
        layers[nLayers-i]->backPropagateDelta(series[k-1],series[k],series[k+1],_weights,_biases);
    }
    
    for (int i=1; i<=nLayers; i++) {
        layers[nLayers-i]->backPropagateDeltaFirst(series[first],series[first+1],_weights,_biases);
    }
#else
    for (int k=first; k>=last; k--) {
        for (int i=1; i<=nLayers; i++)
        layers[nLayers-i]->backPropagateDelta(series[k],_weights,_biases);
    }
#endif
}

void Network::computeGradsSeries(const vector<Activation*>& _series, const int k, Grads* const _Grad, const Real* const _weights) const
{
	assert(bBuilt);
    for (int i=0; i<nLayers; i++)
        layers[i]->backPropagateGrads(series[k-1],series[k],_Grad, _weights);
}

void Network::computeAddGradsSeries(const vector<Activation*>& _series, const int first, const int last, Grads* const _Grad, const Real* const _weights) const
{
	assert(bBuilt);
    for (int i=0; i<nLayers; i++)
        layers[i]->backPropagateAddGrads(series[first],_Grad, _weights);
    
    for (int k=first+1; k<=last; k++)
        for (int i=0; i<nLayers; i++)
            layers[i]->backPropagateAddGrads(series[k-1],series[k],_Grad, _weights);
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

void Network::expandMemory(Mem * _M, Activation * _N) const
{
    std::swap(_N->outvals,_M->outvals);
    std::swap(_N->ostates,_M->ostates);
}

void Network::allocateSeries(int _k, vector<Activation*> & _series)
{
    for (int j=static_cast<int>(_series.size()); j<=_k; j++) {
        Activation * ns = new Activation(nNeurons,nStates);
        series.push_back(ns);
    }
}

void Network::assignDropoutMask()
{
    if (Pdrop > 0)
    {
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

void Network::checkGrads(const vector<vector<Real>>& inputs, const int lastn)
{
	assert(bBuilt);
    printf("Checking gradients\n");
    vector<int> errorPlacements(lastn);
    vector<Real> partialResults(lastn);

    int nseries = inputs.size();
    vector<Real> res(nOutputs);
    allocateSeries(nseries+1);
    
    const Real incr = 1e-6;
    
    uniform_real_distribution<Real> dis(0.,1.);
    for (int i=0; i<lastn; i++)
        errorPlacements[i] = nOutputs*dis(*gen);
    
    Grads * g = new Grads(nWeights,nBiases);
    Grads * G = new Grads(nWeights,nBiases);
    
    {
        predict(inputs[0], res, series[0]);
        
        vector<Real> errs(nOutputs,0);
        errs[errorPlacements[0]] = -1.;
        setOutputErrors(errs, series[0]);
    }
    
    for (int k=1; k<lastn; k++)
    {
        predict(inputs[k], res, series[k-1], series[k]);
        
        vector<Real> errs(nOutputs,0);
        errs[errorPlacements[k]] = -1.;
        setOutputErrors(errs, series[k]);
    }

    computeDeltasSeries(series, 0, lastn-1);
    computeAddGradsSeries(series, 0, lastn-1, G);
    
    
    for (int w=0; w<nWeights; w++) {
        //1
        *(weights+w) += incr;
        
        predict(inputs[0], res, series[0]);
        partialResults[0] =- res[errorPlacements[0]];
        
        for (int k=1; k<lastn; k++) {
            predict(inputs[k], res, series[k-1], series[k]);
            partialResults[k] =- res[errorPlacements[k]];
        }
        
        //2
        *(weights+w) -= 2*incr;
        
        predict(inputs[0], res, series[0]);
        partialResults[0] += res[errorPlacements[0]];
            
        for (int k=1; k<lastn; k++) {
            predict(inputs[k], res, series[k-1], series[k]);
            partialResults[k] += res[errorPlacements[k]];
        }
        
        //0
        *(weights+w) += incr;
        
        Real grad(0);
        for (int k=0; k<lastn; k++)
            grad += partialResults[k];
        *(g->_W+w) = grad/(2.*incr);
        
        //const Real scale = fabs(*(biases+w));
        const Real scale = max(fabs(*(G->_W+w)),fabs(*(g->_W+w)));
        const Real err = (*(G->_W+w)-*(g->_W+w))/scale;
        if (fabs(err)>1e-4) cout <<"W"<<w<<" "<<*(G->_W+w)<<" "<<*(g->_W+w)<<" "<<err<<endl;
    }
    
    for (int w=0; w<nBiases; w++) {
        //1
        *(biases+w) += incr;
        
        predict(inputs[0], res, series[0]);
        partialResults[0] =- res[errorPlacements[0]];
        
        for (int k=1; k<lastn; k++) {
            predict(inputs[k], res, series[k-1], series[k]);
            partialResults[k] =- res[errorPlacements[k]];
        }
        
        //2
        *(biases+w) -= 2*incr;
        
        predict(inputs[0], res, series[0]);
        partialResults[0] += res[errorPlacements[0]];
        
        for (int k=1; k<lastn; k++) {
            predict(inputs[k], res, series[k-1], series[k]);
            partialResults[k] += res[errorPlacements[k]];
        }
        
        //0
        *(biases+w) += incr;
        
        Real grad(0);
        for (int k=0; k<lastn; k++)
            grad += partialResults[k];
        *(g->_B+w) = grad/(2.*incr);
        
        //const Real scale = fabs(*(biases+w));
        const Real scale = max(fabs(*(G->_B+w)),fabs(*(g->_B+w)));
        const Real err = (*(G->_B+w)-*(g->_B+w))/scale;
        if (fabs(err)>1e-4) cout <<"B"<<w<<" "<<*(G->_B+w)<<" "<<*(g->_B+w)<<" "<<err<<endl;
    }
    printf("\n"); fflush(0);
}

/*
void Network::checkGrads(const vector<vector<Real>>& inputs, const int lastn)
{
    printf("Checking gradients\n");
    const int ierr = 0;
 
    int nseries = inputs.size();
    vector<Real> res(nOutputs), errs(nOutputs,0);
    allocateSeries(nseries+1);
    
    Grads * g = new Grads(nWeights,nBiases);
    Grads * G = new Grads(nWeights,nBiases);
    const Real eps = 1e-5;
    
    predict(inputs[0], res, series[0]);
    
    for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
    
    errs[ierr] = -1.;
    setOutputErrors(errs, series[lastn-1]);
    computeDeltasSeries(series, 0, lastn-1);
    computeAddGradsSeries(series, 0, lastn-1, G);
    
    for (int w=0; w<nWeights; w++) {
        *(weights+w) += eps;
        predict(inputs[0], res, series[0]);
        for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
        const Real out1 = - *(series[lastn-1]->outvals+iOut[ierr]);
        
        *(weights+w) -= 2*eps;
        predict(inputs[0], res, series[0]);
        for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
        const Real out2 = - *(series[lastn-1]->outvals+iOut[ierr]);
        
        *(weights+w) += eps;
        *(g->_W+w) = (out1-out2)/(2*eps);
        
        //const Real scale = fabs(*(biases+w));
        const Real scale = max(fabs(*(G->_W+w)),fabs(*(g->_W+w)));
        const Real err = (*(G->_W+w)-*(g->_W+w))/scale;
        if (fabs(err)>1e-4) cout <<"W"<<w<<" "<<*(G->_W+w)<<" "<<*(g->_W+w)<<" "<<err<<endl;
    }
    
    for (int w=0; w<nBiases; w++) {
        *(biases+w) += eps;
        predict(inputs[0], res, series[0]);
        for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
        const Real out1 = - *(series[lastn-1]->outvals+iOut[ierr]);
        
        *(biases+w) -= 2*eps;
        predict(inputs[0], res, series[0]);
        for (int k=1; k<lastn; k++) predict(inputs[k], res, series[k-1], series[k]);
        const Real out2 = - *(series[lastn-1]->outvals+iOut[ierr]);
        
        *(biases+w) += eps;
        *(g->_B+w) = (out1-out2)/(2*eps);
        
        //const Real scale = fabs(*(biases+w));
        const Real scale = max(fabs(*(G->_B+w)),fabs(*(g->_B+w)));
        const Real err = (*(G->_B+w)-*(g->_B+w))/scale;
        if (fabs(err)>1e-4) cout <<"B"<<w<<" "<<*(G->_B+w)<<" "<<*(g->_B+w)<<" "<<err<<endl;
    }
    printf("\n"); fflush(0);
 }
 */

void Network::computeDeltasInputs(vector<Real>& grad, const Activation* const _series, const Real* const _weights, const Real* const _biases) const
{//no weight grad to care about, no recurrent links
    assert(static_cast<int>(grad.size())==nInputs);
    int k = 0;
    for (auto & graph : G) if(graph->input)
    for (int n=0; n<graph->layerSize; n++) {
        Real dEdy(0);
        for (const auto & link : *(graph->output_links_vec)) //loop over all layers to which this layer is connected to
        	dEdy += link->backPropagate(_series, n, _weights);
        grad[k++] = dEdy; //no response function on inputs
    }
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
