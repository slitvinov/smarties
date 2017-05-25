/*
 *  LSTMNet.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cassert>
#include "Network.h"

#include <fstream>

class Builder
{
	Function* readFunction(const string name, const bool bOutput=false) const
	{
		if (bOutput || name == "Linear") return new Linear();
		else
		if (name == "Tanh") 	return new Tanh();
		else
		if (name == "TwoTanh") return new TwoTanh();
		else
		if (name == "Sigm") return new Sigm();
		else
		if (name == "SoftSign") return new SoftSign();
		else
		if (name == "TwoSoftSign") return new TwoSoftSign();
		else
		if (name == "SoftSigm") return new SoftSigm();
		else
		if (name == "Relu") return new Relu();
		else
		if (name == "PRelu") return new PRelu();
		else
		if (name == "ExpPlus") return new ExpPlus();
		else
		if (name == "SoftPlus") return new SoftPlus();
		else
		die("Activation function not recognized\n");
		return (Function*)nullptr;
	}
	static const int simdWidth = __vec_width__/sizeof(Real);
	int roundUpSimd(const int size) const
	{
		return std::ceil(size/(Real)simdWidth)*simdWidth;
	}

	void build_LSTM_layer(Graph* const graph)
	{
	  assert(graph->written == true && !graph->built);
		assert(nNeurons%simdWidth==0 && nBiases%simdWidth==0 && nWeights%simdWidth==0);
		vector<LinkToLSTM*>* input_links = new vector<LinkToLSTM*>();
		LinkToLSTM* recurrent_link = nullptr;

	  graph->layerSize_simd = std::ceil(graph->layerSize/(Real)simdWidth)*simdWidth;
	  assert(graph->layerSize>0 && graph->layerSize_simd>=graph->layerSize);
	  assert(graph->linkedTo.size()>0 && nNeurons>0 && nStates>=0);

	  graph->firstNeuron_ID = nNeurons;
	  nNeurons += graph->layerSize_simd; //move the counter
	  graph->firstState_ID = nStates;
	  nStates += graph->layerSize_simd; //one state per cell
		graph->firstBias_ID   = nBiases;
		graph->firstBiasIG_ID = graph->firstBias_ID   + graph->layerSize_simd;
		graph->firstBiasFG_ID = graph->firstBiasIG_ID + graph->layerSize_simd;
		graph->firstBiasOG_ID = graph->firstBiasFG_ID + graph->layerSize_simd;
		nBiases += 4*graph->layerSize_simd; //one bias per neuron

		for(int i=0; i<graph->linkedTo.size(); i++)
	  {
	    assert(graph->linkedTo[i]>=0 || graph->linkedTo[i]<G.size());
	    const Graph* const layerFrom = G[graph->linkedTo[i]];
	    assert(layerFrom->firstNeuron_ID<graph->firstNeuron_ID);
	    assert(layerFrom->written && layerFrom->built);
	    assert(layerFrom->firstNeuron_ID>=0);
	    assert(layerFrom->layerSize>0);

	  	int firstWeightIG =nWeights      +layerFrom->layerSize*graph->layerSize_simd;
	  	int firstWeightFG =firstWeightIG +layerFrom->layerSize*graph->layerSize_simd;
	  	int firstWeightOG =firstWeightFG +layerFrom->layerSize*graph->layerSize_simd;

	  	LinkToLSTM* tmp = new LinkToLSTM(
	      layerFrom->layerSize, layerFrom->firstNeuron_ID,
	      graph->layerSize, graph->firstNeuron_ID, graph->firstState_ID,
	      nWeights, firstWeightIG, firstWeightFG, firstWeightOG,
	      graph->layerSize_simd
	    );

	  	input_links->push_back(tmp);
			graph->links->push_back(tmp);
			nWeights += layerFrom->layerSize*graph->layerSize_simd*4;
		}
		{ //connected  to past realization of current recurrent layer
			int firstWeightIG = nWeights 		  +graph->layerSize*graph->layerSize_simd;
			int firstWeightFG = firstWeightIG +graph->layerSize*graph->layerSize_simd;
			int firstWeightOG = firstWeightFG +graph->layerSize*graph->layerSize_simd;

			recurrent_link = new LinkToLSTM(
	      graph->layerSize, graph->firstNeuron_ID,
	      graph->layerSize, graph->firstNeuron_ID, graph->firstState_ID,
	      nWeights, firstWeightIG, firstWeightFG, firstWeightOG,
	      graph->layerSize_simd
	    );

			graph->links->push_back(recurrent_link);
			nWeights += graph->layerSize * graph->layerSize_simd*4;
		}

		Layer * l = new LSTMLayer(
	    graph->layerSize, graph->firstNeuron_ID, graph->firstState_ID,
	    graph->firstBias_ID, graph->firstBiasIG_ID, graph->firstBiasFG_ID,
	    graph->firstBiasOG_ID, input_links, recurrent_link,
	    graph->func, graph->gate, graph->cell, graph->layerSize_simd
	  );

		layers.push_back(l);
	  graph->built = true;
	}

	void build_normal_layer(Graph* const graph)
	{
	    assert(graph->written == true && !graph->built);
	  	assert(nNeurons%simdWidth==0 && nBiases%simdWidth==0 && nWeights%simdWidth==0);
	  	vector<NormalLink*>* input_links = new vector<NormalLink*>();
	  	NormalLink* recurrent_link = nullptr;

	    graph->layerSize_simd = std::ceil(graph->layerSize/(Real)simdWidth)*simdWidth;
	    assert(graph->layerSize>0 && graph->layerSize_simd>=graph->layerSize);
	    assert(graph->linkedTo.size()>0 && nNeurons>0);

	    graph->firstNeuron_ID = nNeurons;
	    nNeurons += graph->layerSize_simd; //move the counter
	    graph->firstBias_ID = nBiases;
	    nBiases += graph->layerSize_simd; //one bias per neuron

	    for(int i = 0; i<graph->linkedTo.size(); i++)
	    {
	    	assert(graph->linkedTo[i]>=0 || graph->linkedTo[i]<G.size());
	    	const Graph* const layerFrom = G[graph->linkedTo[i]];
	      assert(layerFrom->firstNeuron_ID<graph->firstNeuron_ID);
	      assert(layerFrom->written && layerFrom->built);
	    	assert(layerFrom->firstNeuron_ID>=0);
	      assert(layerFrom->layerSize>0);

	    	NormalLink* tmp = new NormalLink(
	        layerFrom->layerSize, layerFrom->firstNeuron_ID,
	        graph->layerSize, graph->firstNeuron_ID,
	        nWeights, graph->layerSize_simd
	      );

	    	input_links->push_back(tmp);
			  graph->links->push_back(tmp);
	    	nWeights += layerFrom->layerSize*graph->layerSize_simd;
	    }

	    if (graph->RNN)
			{ //connected  to past realization of current normal layer
	    	recurrent_link = new NormalLink(
	        graph->layerSize, graph->firstNeuron_ID,
	        graph->layerSize, graph->firstNeuron_ID,
	        nWeights, graph->layerSize_simd
				);

			  graph->links->push_back(recurrent_link);
	    	nWeights += graph->layerSize * graph->layerSize_simd;
	    }

		  Layer * l = new NormalLayer(
	      graph->layerSize, graph->firstNeuron_ID,
	      graph->firstBias_ID,
	      input_links, recurrent_link,
	      graph->func, graph->layerSize_simd
	    );

	    layers.push_back(l);
	    graph->built = true;
	}

	void build_conv2d_layer(Graph* const graph)
	{
	    assert(graph->written == true && !graph->built);
	  	assert(nNeurons%simdWidth==0 && nBiases%simdWidth==0 && nWeights%simdWidth==0);
	  	vector<LinkToConv2D*>* input_links = new vector<LinkToConv2D*>();

	    assert(graph->featsNumber==graph->layerDepth);
	    graph->layerDepth_simd = std::ceil(graph->layerDepth/(Real)simdWidth)*simdWidth;
	    graph->featsNumber_simd = graph->layerDepth_simd;
	    graph->layerSize_simd = graph->layerWidth*graph->layerHeight*graph->layerDepth_simd;
	    assert(graph->layerSize>0 && graph->layerSize_simd>=graph->layerSize);
	    assert(graph->linkedTo.size()==1 && nNeurons>0);

	    graph->firstNeuron_ID = nNeurons;
	    nNeurons += graph->layerSize_simd; //move the counter
	    graph->firstBias_ID = nBiases;
	    nBiases += graph->layerSize_simd; //one bias per neuron

			const Graph* const layerFrom = G[graph->linkedTo[0]];
	    assert(layerFrom->firstNeuron_ID < graph->firstNeuron_ID);
	    assert(layerFrom->written && layerFrom->built);
	  	assert(layerFrom->firstNeuron_ID>=0);
	    assert(layerFrom->layerSize>0);

			LinkToConv2D* tmp = new LinkToConv2D(
	      layerFrom->layerSize, layerFrom->firstNeuron_ID,
	      graph->layerSize, graph->firstNeuron_ID,
	      nWeights, graph->layerDepth_simd,
	      layerFrom->layerWidth, layerFrom->layerHeight, layerFrom->layerDepth,
	      graph->featsWidth, graph->featsHeight, graph->featsNumber,
	      graph->layerWidth, graph->layerHeight,
	      graph->strideWidth, graph->strideHeight,
	      graph->padWidth, graph->padHeight
	    );

			input_links->push_back(tmp);
			graph->links->push_back(tmp);
			nWeights += graph->featsWidth*graph->featsHeight*graph->featsNumber_simd
	                * layerFrom->layerDepth;

	    Layer * l = new Conv2DLayer(
	      graph->layerSize, graph->firstNeuron_ID,
	      graph->firstBias_ID, input_links,
	      graph->func, graph->layerSize_simd
	    );

	    layers.push_back(l);
	    graph->built = true;
	}

public:
		int nAgents, nThreads;
		int nInputs=0,  nOutputs=0, nLayers=0;
		int nNeurons=0, nWeights=0, nBiases=0, nStates=0;
		std::vector<std::mt19937>& generators;
		bool bAddedInput = false;
		bool bBuilt = false;
		vector<int> iOut;
		vector<int> iInp;
		vector<Graph*> G;
		vector<Layer*> layers;
		Settings & settings;
		Grads* grad;
		vector<Grads*> Vgrad;
		Real* weights;
		Real* biases;
		Real* tgt_weights;
		Real* tgt_biases;


		Builder(Settings & _settings): settings(_settings), nAgents(_settings.nAgents),
		nThreads(_settings.nThreads), generators(_settings.generators) {}

    Network* build()
		{
			if(bBuilt) die("Cannot build the network multiple times\n");
			bBuilt = true;
			if(nLayers<2) die("Put at least one hidden layer.\n");

			nNeurons = nBiases = nWeights = nStates = 0;
			for (auto & graph : G)
			{
				if(graph->input)
				{
					assert(graph->written == true && !graph->built);
		      graph->firstNeuron_ID = nNeurons;
		      graph->layerSize_simd = roundUpSimd(graph->layerSize);
		      nNeurons += graph->layerSize_simd;
					graph->built = true;
					continue; //input layer is not a layer
				}
				assert(!graph->input);
				if      (graph->LSTM)
					build_LSTM_layer(graph);
				else if (graph->Conv2D)
					build_conv2d_layer(graph);
				else
					build_normal_layer(graph);

				graph->check();
			}
			printf("nInputs:%d, nOutputs:%d, nLayers:%d, nNeurons:%d, nWeights:%d, nBiases:%d, nStates:%d\n",
				nInputs, nOutputs, nLayers, nNeurons, nWeights, nBiases, nStates);
		  assert(layers.size() == nLayers);
			{
				assert(!iInp.size());
				for (const auto & g : G)
					if(g->input)
						for(int i=0; i<g->layerSize; i++)
							iInp.push_back(i + g->firstNeuron_ID);
		 	 	assert(iInp.size() == nInputs);
				assert(iInp[0] == 0);
			}
			{
				assert(!iOut.size());
				for (const auto & g : G)
					if(g->output)
						for(int i=0; i<g->layerSize; i++)
							iOut.push_back(i + g->firstNeuron_ID);
		 	 	assert(iOut.size() == nOutputs);
			}

			_allocateQuick(tgt_weights, nWeights)
		  _allocateQuick(tgt_biases, nBiases)
			_allocateClean(weights, nWeights)
			_allocateClean(biases, nBiases)

			grad = new Grads(nWeights,nBiases);
			Vgrad.resize(nThreads);
			for (int i=0; i<nThreads; ++i)
				Vgrad[i] = new Grads(nWeights, nBiases);

			for (const auto & l : layers)
				l->initialize(&generators[0], weights, biases);

			return new Network(this, settings);
		}

    int getLastLayerID() const {return G.size()-1;}

		void addConv2DLayer(const int filterSize[3], const int outSize[3],
		  const int padding[2], const int stride[2], const string funcType,
		  const bool bOutput = false)
		{
			if(not bAddedInput) die("First specify an input\n");
			if(bBuilt) die("Cannot build the network multiple times\n");

		  assert(filterSize[2]==outSize[2]);
			if(filterSize[0]<=0 || filterSize[1]<=0 || filterSize[2]<=0)
		    die("Bad request for conv2D layer\n");
			if(outSize[0]<=0 || outSize[1]<=0 || outSize[2]<=0)
		    die("Bad request for conv2D layer\n");
			assert(filterSize[0] >= stride[0] && filterSize[1] >= stride[1]);
			assert(padding[0] < filterSize[0] && padding[1] < filterSize[1]);

			nLayers++;
			Graph * g = new Graph();

		  assert(!funcType.empty());
			g->Conv2D = true;
		  g->func = readFunction(funcType, bOutput);
		  g->layerSize = outSize[0] * outSize[1] * outSize[2];
			g->layerWidth = outSize[0];
		  g->layerHeight = outSize[1];
		  g->layerDepth = outSize[2];
			g->featsWidth = filterSize[0];
		  g->featsHeight = filterSize[1];
		  g->featsNumber = filterSize[2];
			g->padWidth = padding[0];
		  g->padHeight = padding[1];
		  g->strideWidth = stride[0];
		  g->strideHeight = stride[1];

			{
		    // link only to previous layer:
				if(G.size()<1) die("Conv2D link not available\n");
				g->linkedTo.push_back(G.size()-1);
		    assert(g->linkedTo.size() == 1);
				const Graph* const layerFrom = G[g->linkedTo[0]];

				if(layerFrom->layerWidth<=0 || layerFrom->layerHeight<=0 || layerFrom->layerDepth<=0)
					die("Incompatible with 1D input, place 2D input or resize to 2D... how?\n");

				assert((outSize[0]-1)*stride[0]+filterSize[0]-(layerFrom->layerWidth +padding[0])>=0);
				assert((outSize[0]-1)*stride[0]+filterSize[0]-(layerFrom->layerWidth +padding[0])<filterSize[0]);
				assert((outSize[1]-1)*stride[1]+filterSize[1]-(layerFrom->layerHeight+padding[1])>=0);
				assert((outSize[1]-1)*stride[1]+filterSize[1]-(layerFrom->layerHeight+padding[1])<filterSize[1]);
			}

			if (bOutput) {
		    assert(false);
				g->output = true;
				nOutputs += g->layerSize;
			}

		  g->written = true;
			G.push_back(g);
		}

		void addLayer(const int size, const string layerType, const string funcType,
		  vector<int> linkedTo, const bool bOutput=false)
		{
			if(bBuilt) die("Cannot build the network multiple times\n");
			if(not bAddedInput) die("First specify an input\n");
			if(size<=0) die("Requested an empty layer\n");

		  nLayers++;
			Graph * g = new Graph();

		  assert(!layerType.empty());
		  if (layerType == "RNN")  g->RNN = true; //non-LSTM recurrent neural network
			else
			if (layerType == "LSTM") g->LSTM = true;

			assert(!funcType.empty());
		  if(g->LSTM) {
		    g->func = new Linear();
		    g->cell = new SoftSigm();
		    g->gate = readFunction(funcType, bOutput);
		  } else g->func = readFunction(funcType, bOutput);

			g->layerSize = size;
			const int nPrevLayers = G.size();
			//default link is to previous layer:
			if(linkedTo.size() == 0)
		    linkedTo.push_back(nPrevLayers-1);

			const int inputLinks = linkedTo.size();
			assert(inputLinks > 0 && g->linkedTo.size() == 0);

			for(int i = 0; i<inputLinks; i++) {
				g->linkedTo.push_back(linkedTo[i]);
				if(linkedTo[i]<0 || linkedTo[i]>=nPrevLayers)
					die("Proposed link not available\n");
			}

		  if (bOutput) {
				g->output = true;
				nOutputs += size;
			}

		  g->written = true;
			G.push_back(g);
		}

		void addLayer(const int size, const string layerType, const string funcType,
			const bool bOutput=false)
		{
			addLayer(size,layerType,funcType,vector<int>(),bOutput);
		}
		void addOutput(const int size, const string layerType, vector<int> linkedTo)
		{
      addLayer(size,layerType,"Linear",linkedTo,true);
		}
    void addOutput(const int size, const string layerType)
		{
      addLayer(size,layerType,"Linear",vector<int>(),true);
		}

		void addInput(const int size)
		{
			if(bBuilt) die("Cannot build the network multiple times\n");
			if(size<=0) die("Requested an empty input layer\n");

			bAddedInput = true;

			nInputs += size;

			Graph * g = new Graph();

			g->layerSize = size;
			g->input = true;
		  g->written = true;
			G.push_back(g);
		}

		void add2DInput(const int size[3])
		{
			if(bBuilt) die("Cannot build the network multiple times\n");
			if(size[0]<=0 || size[1]<=0 || size[2]<=0)
		    die("Requested an empty input layer\n");

		  bAddedInput = true;

			const int flatSize = size[0] * size[1] * size[2];
			nInputs += flatSize;

			Graph * g = new Graph();

			g->layerSize = flatSize;
			//2d input: depth is actually num of color channels: todo standard notation for eventual 3d?
			g->layerWidth = size[0];
			g->layerHeight = size[1];
			g->layerDepth = size[2];
			g->input = true;
		  g->input2D = true;
		  g->written = true;
			G.push_back(g);
		}
};
