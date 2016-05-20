/*
 *  QApproximator.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 24.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "../StateAction.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <algorithm>
#include <fstream>
#include "../Settings.h"

struct trainData
{
    trainData() : weight(1) {}
    Real weight, MSE, avgQ, minQ, maxQ;
};

class QApproximator
{
protected:
    int nAgents;
	StateInfo  sInfo;
    ActionInfo actInfo;
    Real gamma, lRate;
public:    
	QApproximator(StateInfo newSInfo, ActionInfo newActInfo, Settings & settings) : nAgents(settings.nAgents), sInfo(newSInfo), actInfo(newActInfo), gamma(settings.gamma), lRate(settings.lRate)
    {  }
    trainData stats;
    
    virtual void  get(const State& sOld, vector<Real> & Qold, const State& s, vector<Real> & Q, int iAgent) = 0;
	virtual Real get(const State& s, const Action& a, int nAgent)	= 0;
    virtual void set(const State& s, const Action& a, Real value, int nAgent) = 0;
    virtual Real getMax(const State& s, Action& a, int nAgent) = 0;
	virtual void correct(const State& s, const Action& a, Real error, int nAgent) = 0;
    
	virtual void save(string name) = 0;
	virtual bool restart(string name) = 0;
    virtual void Train(const vector<vector<Real>> & sOld, const vector<int> & a, const vector<Real> & r, const vector<vector<Real>> & s) =0;
    virtual void updateFrozenWeights() {};
    
};
