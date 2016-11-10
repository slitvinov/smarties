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

#include <cassert>
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
        assert(dimUsed<=dim);
        
        bounds.resize(dim); bottom.resize(dim); top.resize(dim);
        isLabel.resize(dim); inUse.resize(dim);
        for (int i=0; i<dim; i++) {
            top[i] = (stateInfo.top[i]);
            bottom[i] = (stateInfo.bottom[i]);
            bounds[i] = (stateInfo.bounds[i]);
            isLabel[i] = (stateInfo.isLabel[i]);
            inUse[i] = (stateInfo.inUse[i]);
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
		for (int i=0; i<sInfo.dim; i++) if (sInfo.inUse[i]) {
            Real res = 2.*(vals[i]-sInfo.bottom[i]) / (sInfo.top[i] - sInfo.bottom[i]) - 1.;
			o << res << " ";
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
	
	for(int i=0; i<sInfo.dim; i++) {
		res.vals[sInfo.dim - i - 1] = idx % sInfo.bounds[i];
		idx /= sInfo.bounds[i];
	}
	return res;
}

struct ActionInfo
{
	int dim; //number of actions per turn
    //discrete actions
	vector<int> bounds, shifts; //if finite set, number of choices per "dim"
    vector<vector<Real>> values; //used for rescaling, would be used if action is input to NN
    vector<Real> upperBounds, lowerBounds;
    
    ActionInfo() {}
    
    ActionInfo& operator= (const ActionInfo& actionInfo)
    {
        dim = actionInfo.dim;
        
        assert(actionInfo.values.size()==dim && actionInfo.bounds.size()==dim && actionInfo.shifts.size()==dim 
        		 && actionInfo.lowerBounds.size() == dim &&  actionInfo.upperBounds.size() == dim);
        
        values = actionInfo.values;
        bounds = actionInfo.bounds;
        shifts = actionInfo.shifts;
        upperBounds = actionInfo.upperBounds;
        lowerBounds = actionInfo.lowerBounds;
        
        assert(values.size()==dim && bounds.size()==dim && shifts.size()==dim 
        		 && lowerBounds.size() == dim &&  upperBounds.size() == dim);
    }

    //from action indices to unique label (for tables, DQN)
    int actionToLabel(vector<Real> vals) const
    {
        int lab=0;
        for (int i=0; i<dim; i++) lab += shifts[i]*realActionToIndex(vals[i],i);
        assert(lab>=0);
        return lab;
    }
    
    vector<Real> labelToAction(int lab) const
    {
    	vector<Real> ret(dim);
        for (int i=actInfo.dim-1; i>=0; i--) {
            ret[i] = indexToRealAction(lab/shifts[i]);
            lab = lab % shifts[i];
        }
        return ret;
    }
    
    Real indexToRealAction(const int lab, const int i) const
	{
    	assert(lab>=0 && i>=0 && i<values.size() && lab<values[i].size());
		return values[i][lab];
	}
	
	int realActionToIndex(const Real val, const int i) const
	{ //From cont. action, convert to an action index using chosen values in environment
		assert(values[i].size() == bounds[i]);
		Real dist = 1e3; int ret = -1;
		for (int j=0; j<bounds[i]; j++) {
			const Real _dist = std::fabs(values[i][j]-val);
			if (_dist<dist) { dist = _dist; ret = j; }
		}
		assert(ret>=0);
		return ret
	}
	
    Real realToScaledReal(const Real action, const int i) const
    { //i have smth in valsContinuous, i want a scaled value with upper and lower bounds
        return 2.*(action - lowerBounds[i])/(upperBounds[i] - lowerBounds[i])-1.;
    }
    
    Real scaledRealToReal(const Real scaled, const int i) const
    { //i have a scaled quantity with lower and upper bnd, i want to get a dimensional action
        return lowerBounds[i] + 0.5*(scaled+1.)*(upperBounds[i] - );
    }
};

class Action
{
private:
    
public:
	ActionInfo actInfo;
    vector<Real> vals;
    mt19937 * gen;
    
	Action(const ActionInfo& newActInfo, mt19937 * g) : actInfo(newActInfo), gen(g)
	{
		vals.resize(actInfo.dim);
	}
	
	Action& operator= (const Action& a)
	{
		if (actInfo.dim != a.actInfo.dim) die("Dimension of actions differ!!!\n");
		for (int i=0; i<actInfo.dim; i++) vals[i] = a.vals[i];
		return *this;
	}
    
	string print() const
	{
		ostringstream o;
		o << "[";
		for (int i=0; i<actInfo.dim-1; i++) o << vals[i] << " ";
		o << vals[actInfo.dim-1];
        o << "]";
		return o.str();
	}
    
    string printClean() const
	{
        ostringstream o;
		for (int i=0; i<actInfo.dim; i++)   o << vals[i] << " ";
		return o.str();
	}
    
    //pack and unpack for MPI comm
    void pack(byte* buf) const
    {
        Real* dbuf = (Real*) buf;
        for (int i=0; i<actInfo.dim; i++) dbuf[i] = vals[i];
    }
    
    void unpack(byte* buf)
    {
        Real* dbuf = (Real*) buf;
        for (int i=0; i<actInfo.dim; i++) vals[i] = dbuf[i];
    }
    
    void set(vector<Real> data)
    {
        for (int i=0; i<actInfo.dim; i++) vals[i] = data[i];
    }
    
    vector<Real> scale() const
    {
        vector<Real> res(actInfo.dim);
        for (int i=0; i<actInfo.dim; i++) res[i] = actInfo.realToScaledReal(vals[i], i);
        return res;
    }
    
    void set_fromScaled(vector<Real> data)
    {
        for (int i=0; i<actInfo.dim; i++) vals[i] = actInfo.scaledRealToReal(data[i], i);
    }
    
    void getRandom(const int iRand = -1)
    {
        std::normal_distribution<Real> dist(0.,0.5);
        
        if ( iRand<0 || iRand >= actInfo.dim ) {
        	//select all random actions
            for (int i=0; i<actInfo.dim; i++) {
                const Real uB = actInfo.values[i].back();
                const Real lB = actInfo.values[i].front();
                valsContinuous[i]=    lB+.5*(std::tanh(dist(*gen))+1.)*(uB-lB);
            }
        } else {  //select just one
				const Real uB = actInfo.values[iRand].back();
				const Real lB = actInfo.values[iRand].front();
				valsContinuous[iRand]=lB+.5*(std::tanh(dist(*gen))+1.)*(uB-lB);
        }
    }

    int getActionLabel()
    {
    	return actInfo.actionToLabel(vals);
    }
};

