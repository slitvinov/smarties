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

struct Graph
{
	const int layer_ID;
	bool written = false; //is this graph already used?
	bool built = false; //is this graph already used?

	bool input = false;
	bool input2D = false;
	bool output = false;
	bool RNN = false;
	bool LSTM = false;
	bool Conv2D = false;

	Uint layerSize = 0;
	Uint layerSize_simd = 0;
	Uint firstNeuron_ID = 0;
	Uint firstState_ID = 0;
	Uint firstBias_ID = 0;

	//LSTM gates:
	Uint firstBiasIG_ID = 0, firstBiasFG_ID = 0, firstBiasOG_ID = 0;
	//Convolutions: feature map size:
	Uint layerWidth = 0, layerHeight = 0, layerDepth = 0;
	Uint layerDepth_simd = 0;
	//Convolutions: padding on bottom side of map:
	Uint padWidth = 0, padHeight = 0;
	//Convolutions: kernel size
	Uint featsWidth = 0, featsHeight = 0, featsNumber = 0;
	Uint featsNumber_simd = 0;
	//Convolutions: strides:
	Uint strideWidth = 0, strideHeight = 0;

	vector<Uint> linkedTo;
	vector<Link*> links;

	Function* func = nullptr;
	Function* cell = nullptr;
	Function* gate = nullptr;
	
	Real weight_init_factor = -1;

	Graph(const int layerid = -1) : layer_ID(static_cast<int>(layerid)) { }
	~Graph() { }

	void check() const
	{
		assert(layer_ID>=0);
		assert(built);
		assert(written);
		assert(layerSize>0);
		assert(layerSize_simd>=layerSize);

		assert((input && func == nullptr) || (!input && func not_eq nullptr));
		assert((input) || (!input && firstNeuron_ID>0));
		//assert((input && firstBias_ID<0)   || (!input && firstBias_ID>=0));

		assert(!input2D || (input2D && input));
		if(input) assert(!RNN);
		if(input) assert(!LSTM);
		if(input) assert(!Conv2D);
		if(input) assert(!output);

		assert((!LSTM && cell == nullptr) || (LSTM && cell not_eq nullptr));
		assert((!LSTM && gate == nullptr) || (LSTM && gate not_eq nullptr));
		assert((!LSTM && firstBiasIG_ID==0) || (LSTM && firstBiasIG_ID>0));
		assert((!LSTM && firstBiasFG_ID==0) || (LSTM && firstBiasFG_ID>0));
		assert((!LSTM && firstBiasOG_ID==0) || (LSTM && firstBiasOG_ID>0));
#ifndef NDEBUG
		const bool layer2D = Conv2D || input2D;
#endif
		assert((!layer2D && layerWidth==0) || (layer2D && layerWidth>0));
		assert((!layer2D && layerHeight==0) || (layer2D && layerHeight>0));
		assert((!layer2D && layerDepth==0) || (layer2D && layerDepth>0));
		assert((!layer2D && layerDepth_simd==0) || (layer2D && layerDepth_simd>0));

		assert((!Conv2D && strideWidth==0) || (Conv2D && strideWidth>0));
		assert((!Conv2D && strideHeight==0) || (Conv2D && strideHeight>0));

		assert((!Conv2D && featsWidth==0) || (Conv2D && featsWidth>0));
		assert((!Conv2D && featsHeight==0) || (Conv2D && featsHeight>0));
		assert((!Conv2D && featsNumber==0) || (Conv2D && featsNumber>0));
		assert((!Conv2D && featsNumber_simd==0) || (Conv2D && featsNumber_simd>0));
	}
};
