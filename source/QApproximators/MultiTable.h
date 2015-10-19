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

#include "QApproximator.h"

class MultiTable : public QApproximator
{
	int dim;
    ActionIterator actionsIt;
	map<long int, double> data;
	map<long int, double> maxStateVal;
	vector<long int> shifts;
    double gamma;
	inline long int _encodeIdx(const State& s, const Action& a) const;
	inline long int _encodeIdx(const long int sId, const Action& a) const;

	template <typename F>
	inline long int _encodeState(const State& s, F&& _discr) const;

	//inline int _lines(const char * const filename);
	
public:
	// Costructor-Destructor
	MultiTable(StateInfo newSInfo, ActionInfo newActInfo, double gamma);
	~MultiTable();
	
	// Methods
    double get (const State& s, const Action& a, int nAgent = 0);
    double get (const State * s, const Action * a, int nAgent = 0);
    double getsmooth (const State& s, const Action& a, int nAgent = 0);
    double test(const State& s, const Action& a, int nAgent = 0)
    {
        return getsmooth(s, a, nAgent);
    }
    double advance(const State& s, const Action& a, int nAgent = 0)
    {
        return get(s, a);
    }
	double getMax (const State& s, int nAgent);
	void   set    (const State& s, const Action& a, double value, int nAgent = 0);
	void   correct(const State& s, const Action& a, double error, int nAgent = 0);
	double usage() const;
    double   Train();
	inline map<long int, double>& getData()  { return data; }
	
	void   save(string name);
	bool   restart(string name);
};
