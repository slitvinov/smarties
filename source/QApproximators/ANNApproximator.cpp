/*
 *  ANNApproximator.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 24.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "ANNApproximator.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm> 
#include <vector>
#include <cmath>

#include "../ErrorHandling.h"
#include "../Misc.h"
#include "../Settings.h"

using namespace ErrorHandling;

ANNApproximator::ANNApproximator(StateInfo newSInfo, ActionInfo newActInfo) : QApproximator(newSInfo, newActInfo), scaledInp(sInfo.dim), rng(rand())
{
	// TODO: multidimensional actions
	nActions = actInfo.bounds[0];
	nStateDims = sInfo.dim;
	batchSize  = round(settings.nnAlpha);
	
	vector<int> lsize;
	lsize.push_back(nStateDims);
	lsize.push_back(settings.nnLayer1);
	lsize.push_back(settings.nnLayer2);
	lsize.push_back(1);
	
	//for (int i=0; i<nActions; i++) ann.push_back(new Network(lsize, settings.nnEta, settings.nnAlpha));
	for (int i=0; i<nActions; i++) ann.push_back(new NetworkLM(lsize, round(settings.nnEta), batchSize));
	prediction.resize(1);
	
	batch.resize(nActions);
}

ANNApproximator::~ANNApproximator()
{
}

double ANNApproximator::get(const State& s, const Action& a)
{
	s.scale(scaledInp);	
	ann[a.vals[0]]->Network::predict(scaledInp, prediction);
	
	if (batch[a.vals[0]].size() < batchSize)
		batch[a.vals[0]].push_back(scaledInp);
	return prediction[0];
}

void ANNApproximator::set(const State& s, const Action& a, double val)
{
	s.scale(scaledInp);	
	ann[a.vals[0]]->predict(scaledInp, prediction);
	prediction[0] = prediction[0] - val;
	ann[a.vals[0]]->improve(prediction);
}



