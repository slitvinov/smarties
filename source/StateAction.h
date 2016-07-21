/*
 *  StateAction.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Settings.h"
#include "Misc.h"

#include <sstream>
#include <math.h>

using namespace std;
using namespace ErrorHandling;


struct StateInfo
{
	int dim, dimUsed;
	vector<int> bounds;
	vector<Real> bottom, top, isLabel, inUse;
    
    StateInfo& operator= (const StateInfo& stateInfo)
    {
        dim     = stateInfo.dim;
        dimUsed = stateInfo.dimUsed;
        bounds.resize(dim); bottom.resize(dim); top.resize(dim);
        isLabel.resize(dim); inUse.resize(dim);
        for (int i=0; i<dim; i++) {
            bounds[i]=     (stateInfo.bounds[i]);
            bottom[i]=     (stateInfo.bottom[i]);
            top[i]=        (stateInfo.top[i]);
            isLabel[i]=    (stateInfo.isLabel[i]);
            inUse[i]=      (stateInfo.isLabel[i]);
        }
    }
};

class State
{
public:
	StateInfo sInfo;
	vector<Real> vals;
	
	State(const StateInfo& newSInfo) : sInfo(newSInfo)
	{
		vals.resize(sInfo.dim);
	};
	
	State& operator= (const State& s)
	{
		if (sInfo.dim != s.sInfo.dim) die("Dimension of states differ!!!\n");
		for (int i=0; i<sInfo.dim; i++) vals[i] = s.vals[i];
		return *this;
	}
	
	string print() const
	{
		ostringstream o;
		o << "[";
		for (int i=0; i<sInfo.dim; i++) {
			o << vals[i];
			if (i < sInfo.dim-1) o << " ";
		}
		o << "]";
		return o.str();
	}
    
    string printClean() const
	{
		ostringstream o;
		for (int i=0; i<sInfo.dim; i++) {
			o << vals[i]<< " ";
		}
		return o.str();
	}
	
	string printScaled()
	{
		ostringstream o;
		o << "[";
		for (int i=0; i<sInfo.dim; i++) {
            Real res = 2.*(vals[i]-sInfo.bottom[i]) / (sInfo.top[i] - sInfo.bottom[i]) - 1.;
			o << res;
			if (i < sInfo.dim-1) o << " ";
		}
		o << "]";
		return o.str();
	}
	
    void scaleUsed(vector<Real>& res) const
    {
        int k(0);
        for (int i=0; i<sInfo.dim; i++)
        if (sInfo.inUse[i]) {
            res[k] = 2.*(vals[i]-sInfo.bottom[i]) / (sInfo.top[i] - sInfo.bottom[i]) - 1.;
            k++;
        }
    }
    
	void scale(vector<Real>& res) const
	{
		for (int i=0; i<sInfo.dim; i++) {
            res[i] = 2.*(vals[i]-sInfo.bottom[i]) / (sInfo.top[i] - sInfo.bottom[i]) - 1.;
        }
	}
    
    void copy(vector<Real>& res) const
    {
        for (int i=0; i<sInfo.dim; i++)
            res[i] = vals[i];
    }
    
    void pack(byte* buf) const
    {
        Real* dbuf = (Real*) buf;
        for (int i=0; i<sInfo.dim; i++)
            dbuf[i] = (Real) vals[i];
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

struct ActionInfo
{
    bool realValues; //finite set, continuous
	int dim; //number of actions per turn
    //discrete actions
    int zeroact; //if finite set: one that corresponds to 0
	vector<int> bounds, shifts; //if finite set, number of choices per "dim"
    vector<vector<Real>> values; //used for rescaling, would be used if action is input to NN
    vector<Real> upperBounds, lowerBounds;
    
    ActionInfo() : realValues(false) {}
    
    ActionInfo& operator= (const ActionInfo& actionInfo)
    {
        realValues = actionInfo.realValues;
        dim = actionInfo.dim;
        zeroact = actionInfo.zeroact;
        
        values.clear(); bounds.resize(dim); upperBounds.resize(dim); lowerBounds.resize(dim);
        for (int i=0; i<dim; i++) values.push_back(actionInfo.values[i]);
        bounds = actionInfo.bounds;
        shifts = actionInfo.shifts;
        upperBounds = actionInfo.upperBounds;
        lowerBounds = actionInfo.lowerBounds;
    }
    
};

class Action
{
public:
	ActionInfo actInfo;
    vector<int>  vals;
    vector<Real> valsContinuous;
    mt19937 * gen;
    
    int pack() const
    {
        int lab=vals[0];
        for (int i=1; i<actInfo.dim; i++)
            lab += actInfo.shifts[i]*vals[i];
        return lab;
    }
    
    void unpack(int lab)
    {
        for (int i=actInfo.dim-1; i>=0; i--) {
            vals[i] = lab/actInfo.shifts[i];
            valsContinuous[i] = actInfo.values[i][vals[i]];
            lab     = lab%actInfo.shifts[i];
        }
    }
    
	Action(const ActionInfo& newActInfo, mt19937 * g) : actInfo(newActInfo), gen(g)
	{
		vals.resize(actInfo.dim);
        valsContinuous.resize(actInfo.dim);
	}
	
	Action& operator= (const Action& a)
	{
		if (actInfo.dim != a.actInfo.dim) die("Dimension of actions differ!!!\n");
        if (actInfo.realValues != a.actInfo.realValues) die("Contunuous/discrete actions mismatch!!!\n");
        
		for (int i=0; i<actInfo.dim; i++)
            vals[i] = a.vals[i];
        for (int i=0; i<actInfo.dim; i++)
            valsContinuous[i] = a.valsContinuous[i];
		return *this;
	}
	
    void getRand()
    {
        std::normal_distribution<Real> dist(0.,0.5);
        for (int i=0; i<actInfo.dim; i++) {
            valsContinuous[i] = actInfo.lowerBounds[i] + (.5+.5*std::tanh(dist(*gen)))*(actInfo.upperBounds[i]-actInfo.lowerBounds[i]);
            
            vals[i] = (actInfo.bounds[i]-1)*(valsContinuous[i]-actInfo.values[i].front()) /
            (actInfo.values[i].back()-actInfo.values[i].front()) +.49;
            
            if (vals[i]<0) vals[i]=0;
            if (vals[i] > actInfo.bounds[i]-1) vals[i] = actInfo.bounds[i]-1;
        }
    }
    
	string print() const
	{
		ostringstream o;
		o << "[";
            for (int i=0; i<actInfo.dim-1; i++)
                o << valsContinuous[i] << " ";
            o << valsContinuous[actInfo.dim-1];
        o << "]";
		return o.str();
	}
    
    string printClean() const
	{
        ostringstream o;
            for (int i=0; i<actInfo.dim; i++)
                o << valsContinuous[i] << " ";
		return o.str();
	}
    
    void pack(byte* buf) const
    {
        Real* dbuf = (Real*) buf;
        for (int i=0; i<actInfo.dim; i++)
            dbuf[i] = valsContinuous[i];
    }
    
    void unpack(byte* buf)
    {
        Real* dbuf = (Real*) buf;
        for (int i=0; i<actInfo.dim; i++)
            valsContinuous[i] = dbuf[i];
    }

    void set(vector<Real> data)
    {
        for (int i=0; i<actInfo.dim; i++) {
            valsContinuous[i] = data[i];
            vals[i] = (actInfo.bounds[i]-1)*(data[i]-actInfo.values[i].front()) /
                           (actInfo.values[i].back()-actInfo.values[i].front()) +.49;
            if (vals[i]<0) vals[i]=0;
            if (vals[i] > actInfo.bounds[i]-1) vals[i] = actInfo.bounds[i]-1;
        }
    }
    
    vector<Real> scale() const
    {
        vector<Real> res(actInfo.dim);
        for (int i=0; i<actInfo.dim; i++)
            res[i] = 2.*(valsContinuous[i]-actInfo.lowerBounds[i])/(actInfo.upperBounds[i] - actInfo.lowerBounds [i]) -1.;
        return res;
    }
    
    void descale(vector<Real> data)
    {
        for (int i=0; i<actInfo.dim; i++) {
            valsContinuous[i] = actInfo.lowerBounds[i] + .5*(data[i]+1.)*(actInfo.upperBounds[i] - actInfo.lowerBounds [i]);
            vals[i] = (actInfo.bounds[i]-1)*(valsContinuous[i]-actInfo.values[i].front()) /
                                     (actInfo.values[i].back()-actInfo.values[i].front()) +.49;
            if (vals[i]<0) vals[i]=0;
            if (vals[i] > actInfo.bounds[i]-1) vals[i] = actInfo.bounds[i]-1;
        }
    }
};

