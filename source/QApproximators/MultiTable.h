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
	map<long int, Real> data;
	map<long int, Real> maxStateVal;
	vector<long int> shifts;
    Real gamma;
	inline long int _encodeIdx(const State& s, const Action& a) const;
	inline long int _encodeIdx(const long int sId, const Action& a) const;

	template <typename F>
	inline long int _encodeState(const State& s, F&& _discr) const;

	//inline int _lines(const char * const filename);
	
public:
	// Costructor-Destructor
	MultiTable(StateInfo newSInfo, ActionInfo newActInfo, Real gamma);
	~MultiTable();
	
	// Methods
    Real get (const State& s, const Action& a, int nAgent = 0);
    Real get (const State * s, const Action * a, int nAgent = 0);
    Real getsmooth (const State& s, const Action& a, int nAgent = 0);
    Real test(const State& s, const Action& a, int nAgent = 0)
    {
        return getsmooth(s, a, nAgent);
    }
    Real advance(const State& s, const Action& a, int nAgent = 0)
    {
        return get(s, a);
    }
	Real getMax (const State& s, int & nAct, int nAgent);
	void   set    (const State& s, const Action& a, Real value, int nAgent = 0);
	void   correct(const State& s, const Action& a, Real error, int nAgent = 0);
	Real usage() const;
    Real   Train();
	inline map<long int, Real>& getData()  { return data; }
	
	void   save(string name);
	bool   restart(string name);
};
