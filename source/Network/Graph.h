/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Links.h"

struct Graph //misleading, this is just the graph for a single layer
{
	bool written = false; //is this graph already used?
	bool built = false; //is this graph already used?

	bool input = false;
	bool input2D = false;
	bool output = false;
	bool RNN = false;
	bool LSTM = false;
	bool Conv2D = false;

	int layerSize = -1;
	int layerSize_simd = -1;
	int firstNeuron_ID = -1;
	int firstState_ID = -1;
	int firstBias_ID = -1;

	//LSTM gates:
	int firstBiasIG_ID = -1, firstBiasFG_ID = -1, firstBiasOG_ID = -1;
	//Convolutions: feature map size:
	int layerWidth = -1, layerHeight = -1, layerDepth = -1;
	int layerDepth_simd = -1;
	//Convolutions: padding on bottom side of map:
	int padWidth = -1, padHeight = -1;
	//Convolutions: kernel size
	int featsWidth = -1, featsHeight = -1, featsNumber = -1;
	int featsNumber_simd = -1;
	//Convolutions: strides:
	int strideWidth = -1, strideHeight = -1;

	vector<int> linkedTo;
	vector<Link*> links;

	Function* func = nullptr;
	Function* cell = nullptr;
	Function* gate = nullptr;

	Graph() { }
	~Graph() { }

	void check() const
	{
		assert(built);
		assert(written);
		assert(layerSize>0);
		assert(layerSize_simd>=layerSize);

		assert((input && func == nullptr) || (!input && func not_eq nullptr));
		assert((input && firstNeuron_ID>=0) || (!input && firstNeuron_ID>0));
		assert((input && firstBias_ID<0)   || (!input && firstBias_ID>=0));

		assert(!input2D || (input2D && input));
		if(input) assert(!RNN);
		if(input) assert(!LSTM);
		if(input) assert(!Conv2D);
		if(input) assert(!output);

		assert((!LSTM && cell == nullptr) || (LSTM && cell not_eq nullptr));
		assert((!LSTM && gate == nullptr) || (LSTM && gate not_eq nullptr));
		assert((!LSTM && firstBiasIG_ID<0) || (LSTM && firstBiasIG_ID>=0));
		assert((!LSTM && firstBiasFG_ID<0) || (LSTM && firstBiasFG_ID>=0));
		assert((!LSTM && firstBiasOG_ID<0) || (LSTM && firstBiasOG_ID>=0));
		assert((!LSTM && firstState_ID<0) || (LSTM && firstState_ID>=0));
		#ifndef NDEBUG
		const bool layer2D = Conv2D || input2D;
		#endif
		assert((!layer2D && layerWidth<0) || (layer2D && layerWidth>=0));
		assert((!layer2D && layerHeight<0) || (layer2D && layerHeight>=0));
		assert((!layer2D && layerDepth<0) || (layer2D && layerDepth>=0));
		assert((!layer2D && layerDepth_simd<0) || (layer2D && layerDepth_simd>=0));

		assert((!Conv2D && padWidth<0) || (Conv2D && padWidth>=0));
		assert((!Conv2D && padHeight<0) || (Conv2D && padHeight>=0));
		assert((!Conv2D && strideWidth<0) || (Conv2D && strideWidth>=0));
		assert((!Conv2D && strideHeight<0) || (Conv2D && strideHeight>=0));

		assert((!Conv2D && featsWidth<0) || (Conv2D && featsWidth>=0));
		assert((!Conv2D && featsHeight<0) || (Conv2D && featsHeight>=0));
		assert((!Conv2D && featsNumber<0) || (Conv2D && featsNumber>=0));
		assert((!Conv2D && featsNumber_simd<0) || (Conv2D && featsNumber_simd>=0));
	}
};
