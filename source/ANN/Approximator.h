/*
 *  Approximator.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 04.09.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

using namespace std;

class Approximator
{
public:
	virtual void predict(const vector<double>& inputs, vector<double>& outputs)      = 0;
	virtual void improve(const vector<double>& inputs, const vector<double>& errors) = 0;
	
	virtual void   save(string name)    = 0;
	virtual bool   restart(string name) = 0;
};