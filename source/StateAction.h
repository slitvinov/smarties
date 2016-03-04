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
#include "Settings.h"
#include "rng.h"
#include "ErrorHandling.h"
#include "Misc.h"

using namespace std;
using namespace ErrorHandling;

enum StateType {DISCR, ANN, WAVE, LSTM};

struct StateInfo
{
	StateType type;
	
	int dim;
	vector<int> bounds;
	vector<Real> bottom;
	vector<Real> top;
	vector<bool>   belowBottom;
	vector<bool>   aboveTop;
    vector<bool>   isLabel;
    vector<Real> values;
};

class State
{
public:
	StateInfo sInfo;
	vector<Real> vals;
	
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
    
    string printClean()
	{
		ostringstream o;
		for (int i=0; i<sInfo.dim; i++)
		{
			o << vals[i]<< " ";
            //if (i < sInfo.dim-1) o << " ";
		}
		return o.str();
	}
	
	string printScaled()
	{
		ostringstream o;
		o << "[";
		for (int i=0; i<sInfo.dim; i++)
		{
            Real res = vals[i];//(vals[i]-sInfo.bottom[i]) / (sInfo.top[i] - sInfo.bottom[i])*2 - 1;
			
			o << res;
			if (i < sInfo.dim-1) o << " ";
		}
		o << "]";
		return o.str();
	}
	
	void scale(vector<Real>& res) const
	{
		for (int i=0; i<sInfo.dim; i++)
		{
            res[i] = vals[i];//(vals[i]-sInfo.bottom[i]) / (sInfo.top[i] - sInfo.bottom[i])*2. - 1.;
            if (sInfo.isLabel[i]) res[i] = sInfo.values[vals[i]];
        }

	}
    
    void pack(byte* buf)
    {
        Real* dbuf = (Real*) buf;
        for (int i=0; i<sInfo.dim; i++)
            dbuf[i] = vals[i];
    }
    
    void unpack(byte* buf)
    {
        Real* dbuf = (Real*) buf;
        for (int i=0; i<sInfo.dim; i++)
            vals[i] = dbuf[i];
    }
    
    void set(vector<Real> data)
    {
        for (int i=0; i<sInfo.dim; i++)
            vals[i] = data[i];
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
    int zeroact;
	vector<int> bounds;
    vector<Real> values;
};

class Action
{
public:
	ActionInfo actInfo;
	vector<int> vals;
	
	Action() {};
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
	
    void initAct()
    {
        for (int i=0; i<actInfo.dim; i++)
            vals[i] = rand() % actInfo.bounds[i];
        //if (actInfo.dim > 0) vals[0] = -1;
    }
    
    void getRand(RNG* rng)
    {
        for (int i=0; i<actInfo.dim; i++)
            vals[i] = rng->rand_int32() % actInfo.bounds[i];
        //if (actInfo.dim > 0) vals[0] = -1;
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
    
    string printClean()
	{
		ostringstream o;
		for (int i=0; i<actInfo.dim; i++)
		{
			o << vals[i]<< " ";
            //if (i < actInfo.dim-1) o << " ";
		}
		return o.str();
	}
    
    void pack(byte* buf)
    {
        Real* dbuf = (Real*) buf;
        for (int i=0; i<actInfo.dim; i++)
            dbuf[i] = vals[i];
    }
    
    void unpack(byte* buf)
    {
        Real* dbuf = (Real*) buf;
        for (int i=0; i<actInfo.dim; i++)
            vals[i] = dbuf[i];
    }
    
    void set(vector<int> data)
    {
        for (int i=0; i<actInfo.dim; i++)
            vals[i] = data[i];
    }
    
    void scale(vector<Real>& res) const
	{
		for (int i=0; i<actInfo.dim; i++)
            res[res.size() - actInfo.dim + i] = actInfo.values[vals[i]];
	}
};


class ActionIterator
{
private:
	Action currAction, storedAction, rAction;
	
public:
    ActionInfo actInfo;
	ActionIterator(const ActionInfo& newActInfo);
	Action& getRand(RNG* rng);
	
	Action& next();
	bool    done();
	void    memorize();
	Action& recall();
    Action& show();
	void    reset();
    void    initAct();
};






