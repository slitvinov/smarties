/*
 *  MultiTable.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <map>
#include <string>
#include <vector>

#include "StateAction.h"

class MultiTable
{
	int dim;
	StateInfo  sInfo;
	ActionInfo actInfo;
	
	map<long int, double> data;
	map<long int, double> maxStateVal;
	vector<long int> shifts;
	
	inline long int _encodeIdx(const State& s, const Action& a) const;
	inline long int _encodeIdx(const long int sId, const Action& a) const;
	inline long int _encodeState(const State& s) const;

	//inline int _lines(const char * const filename);
	
public:
	// Costructor-Destructor
	MultiTable(StateInfo newSInfo, ActionInfo newActInfo);
	~MultiTable();
	
	// Methods
	double get(const State& s, const Action& a) const;
	double getMax(const State& s);
	double getBest(const State& s, Action& a, double& reward); //TODO
	void   set(const State& s, const Action& a, double value);
	double usage() const;
	
	void   save(string name);
	bool   restart(string name);
};
