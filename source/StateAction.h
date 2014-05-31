/*
 *  StateAction.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <sstream>

#include "rng.h"
#include "ErrorHandling.h"
#include "Misc.h"

using namespace std;
using namespace ErrorHandling;

enum StateType {DISCR, ANN, WAVE};

struct StateInfo
{
	StateType type;
	
	int dim;
	vector<int> bounds;
	vector<double> bottom;
	vector<double> top;
	vector<bool>   belowBottom;
	vector<bool>   aboveTop;
};

class State
{
public:
	StateInfo sInfo;

	vector<double> vals;
	
	State() {};
	State(const StateInfo& newSInfo) : sInfo(newSInfo)
	{
		vals.resize(sInfo.dim);
	};
	
	State& operator= (const State& s)
	{
		if (sInfo.dim != s.sInfo.dim) die("Dimension of states differ!!!\n");
		for (int i=0; i<sInfo.dim; i++)
			vals[i] = s.vals[i];
		return *this;
	}
	
	string print()
	{
		ostringstream o;
		o << "[";
		for (int i=0; i<sInfo.dim; i++)
		{
			o << vals[i];
			if (i < sInfo.dim-1) o << " ";
		}
		o << "]";
		return o.str();
	}
	
	string printScaled()
	{
		ostringstream o;
		o << "[";
		for (int i=0; i<sInfo.dim; i++)
		{
			double res = (vals[i]-sInfo.bottom[i]) / (sInfo.top[i] - sInfo.bottom[i])*4 - 2;
			if (res > 2)  res = 2;
			if (res < -2) res = -2;
			
			o << res;
			if (i < sInfo.dim-1) o << " ";
		}
		o << "]";
		return o.str();
	}
	
	void scale(vector<double>& res) const
	{
		for (int i=0; i<sInfo.dim; i++)
		{
			res[i] = (vals[i]-sInfo.bottom[i]) / (sInfo.top[i] - sInfo.bottom[i])*4 - 2;
			if (res[i] > 2)  res[i] = 2;
			if (res[i] < -2) res[i] = -2;
		}
	}
    
    void pack(byte* buf)
    {
        double* dbuf = (double*) buf;
        for (int i=0; i<sInfo.dim; i++)
            dbuf[i] = vals[i];
    }
    
    void unpack(byte* buf)
    {
        double* dbuf = (double*) buf;
        for (int i=0; i<sInfo.dim; i++)
            vals[i] = dbuf[i];
    }
	
};

inline State decode(const StateInfo& sInfo, long int idx)
{
	State res(sInfo);
	
	for(int i=0; i<sInfo.dim; i++)
	{
		res.vals[sInfo.dim - i - 1] = idx % sInfo.bounds[i];
		idx /= sInfo.bounds[i];
	}
	return res;
}

//**************************************************************************************************************************************
//
//**************************************************************************************************************************************

struct ActionInfo
{
	int dim;
	vector<int> bounds;
};

class Action
{
private:
	ActionInfo actInfo;
	
public:
	vector<int> vals;
	
	Action(const ActionInfo& newActInfo) : actInfo(newActInfo)
	{
		vals.resize(actInfo.dim);
	}
	
	Action& operator= (const Action& a)
	{
		if (actInfo.dim != a.actInfo.dim) die("Dimension of actions differ!!!\n");
		for (int i=0; i<actInfo.dim; i++)
			vals[i] = a.vals[i];
		return *this;
	}
	
	string print()
	{
		ostringstream o;
		o << "[";
		for (int i=0; i<actInfo.dim-1; i++)
			o << vals[i] << " ";
		o << vals[actInfo.dim-1] << "]";
		return o.str();
	}
    
    void pack(byte* buf)
    {
        double* dbuf = (double*) buf;
        for (int i=0; i<actInfo.dim; i++)
            dbuf[i] = vals[i];
    }
    
    void unpack(byte* buf)
    {
        double* dbuf = (double*) buf;
        for (int i=0; i<actInfo.dim; i++)
            vals[i] = dbuf[i];
    }
	
};


class ActionIterator
{
private:
	Action currAction, storedAction, rAction;
	ActionInfo actInfo;
	
public:
	ActionIterator(const ActionInfo& newActInfo);
	Action& getRand(RNG* rng);
	
	Action& next();
	bool    done();
	void    memorize();
	Action& recall();
	void    reset();
};






