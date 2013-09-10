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
#include "../ANN/Network.h"
#include "../ANN/WaveletNet.h"

using namespace ErrorHandling;


ANNApproximator::ANNApproximator(StateInfo newSInfo, ActionInfo newActInfo, StateType tp) :
QApproximator(newSInfo, newActInfo), scaledInp(sInfo.dim), rng(rand()), sType(tp)
{
	// TODO: multidimensional actions
	nActions = actInfo.bounds[0];
	nStateDims = sInfo.dim;
	batchSize  = round(settings.nnAlpha);
	
	vector<int> lsize;
	lsize.push_back(nStateDims);
	lsize.push_back(settings.nnLayer1);
	//lsize.push_back(settings.nnLayer2);
	lsize.push_back(1);
	
	//for (int i=0; i<nActions; i++) ann.push_back(new WaveletNet(lsize, 1, 1, batchSize));
	if (sType == WAVE)
		for (int i=0; i<nActions; i++)
			ann.push_back(new WaveletNetLM(lsize, batchSize));
	
	//for (int i=0; i<nActions; i++) ann.push_back(new Network(lsize, settings.nnEta, settings.nnAlpha));
	if (sType == ANN)
		for (int i=0; i<nActions; i++)
			ann.push_back(new NetworkLM(lsize, round(settings.nnEta), batchSize));
	prediction.resize(1);
}

ANNApproximator::~ANNApproximator()
{
}

double ANNApproximator::get(const State& s, const Action& a)
{
	s.scale(scaledInp);	
	ann[a.vals[0]]->predict(scaledInp, prediction);
	
	return prediction[0];
}

void ANNApproximator::set(const State& s, const Action& a, double val)
{
	s.scale(scaledInp);	
	ann[a.vals[0]]->predict(scaledInp, prediction);
	prediction[0] = prediction[0] - val;
	ann[a.vals[0]]->improve(scaledInp, prediction);
}

void ANNApproximator::correct(const State& s, const Action& a, double err)
{
	s.scale(scaledInp);	
	prediction[0] = -err;
	ann[a.vals[0]]->improve(scaledInp, prediction);
}

void ANNApproximator::save(string name)
{
	for (int i=0; i<nActions; i++)
	{
		string suff;
		if (sType == ANN)  suff = "ANN_act";
		if (sType == WAVE) suff = "WAVE_act";
		
		ann[i]->save(name + suff + to_string(i));
	}
}

bool ANNApproximator::restart(string name)
{
	bool res = true;
	for (int i=0; i<nActions; i++)
	{
		string suff;
		if (sType == ANN)  suff = "ANN_act";
		if (sType == WAVE) suff = "WAVE_act";
		
		res = ann[i]->restart(name + suff + to_string(i)) && res;
	}
	return res;
}










