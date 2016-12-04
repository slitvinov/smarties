/*
 *  Approximator.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 04.09.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Activations.h"

using namespace std;
//typedef int st;

class Approximator
{
public:
    //virtual void improve(const vector<Real>& error, int iAgent) = 0;

    void predict(const vector<Real>& input, vector<Real>& output, int iAgent)=0;

    void predict(const vector<Real>& S1, vector<Real>& Q1,
                 const vector<Real>& S2, vector<Real>& Q2, int iAgent) = 0;

	void save(string name) = 0;
	bool restart(string name) = 0;
    void setBatchsize(int size) = 0;

    void updateFrozenWeights() = 0;
    void resetMemories(int iAgent=0) {};

    void trainDQ(const vector<vector<Real>> & sOld, const vector<int> & a,
                 const vector<Real> & r, const vector<vector<Real>> & s) = 0;
};
