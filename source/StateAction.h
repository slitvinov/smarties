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
    
    ActionInfo() : {}
    
    ActionInfo& operator= (const ActionInfo& actionInfo)
    {
        dim = actionInfo.dim;
        
        values.clear(); bounds.resize(dim);
        upperBounds.resize(dim); lowerBounds.resize(dim);
        for (int i=0; i<dim; i++) values.push_back(actionInfo.values[i]);
        bounds = actionInfo.bounds;
        shifts = actionInfo.shifts;
        upperBounds = actionInfo.upperBounds;
        lowerBounds = actionInfo.lowerBounds;
    }
    
};

class Action
{
private:
    void indexToRealAction(const int i)
    {
        valsContinuous[i] = actInfo.values[i][vals[i]];
    }
    
    void realActionToIndex(const int i)
    { //From cont. action, convert to an action index using chosen values in environment
        assert(actInfo.values[i].size() == actInfo.bounds[i]);
        const vector<Real> values(actInfo.values[i]);
        const int nBounds = actInfo.bounds[i];
        Real dist = 1e3;
        for (int j=0; j<nBounds; j++) {
            const Real _dist = std::fabs(values[j]-valsContinuous[i]);
            if (_dist<dist) { dist = _dist; vals[i] = j; }
        }
    }
    
    Real realToScaledReal(const int i) const
    { //i have smth in valsContinuous, i want a scaled value with upper and lower bounds
        const Real upperBound = actInfo.upperBounds[i];
        const Real lowerBound = actInfo.lowerBounds[i];
        return 2.*(valsContinuous[i] - lowerBound)/(upperBound - lowerBound)-1.;
    }
    
    Real scaledRealToReal(const Real scaled, const int i) const
    { //i have a scaled quantity with lower and upper bnd, i want to get a dimensional action
        const Real upperBound = actInfo.upperBounds[i];
        const Real lowerBound = actInfo.lowerBounds[i];
        return lowerBound + 0.5*(scaled+1.)*(upperBound - lowerBound);
    }
    
public:
	ActionInfo actInfo;
    vector<int>  vals;
    vector<Real> valsContinuous;
    mt19937 * gen;
    
	Action(const ActionInfo& newActInfo, mt19937 * g) : actInfo(newActInfo), gen(g)
	{
		vals.resize(actInfo.dim);
        valsContinuous.resize(actInfo.dim);
	}
	
	Action& operator= (const Action& a)
	{
		if (actInfo.dim != a.actInfo.dim) die("Dimension of actions differ!!!\n");

		for (int i=0; i<actInfo.dim; i++) vals[i] = a.vals[i];
        for (int i=0; i<actInfo.dim; i++) valsContinuous[i] = a.valsContinuous[i];
		return *this;
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
    
    //pack and unpack for MPI comm
    void pack(byte* buf) const
    {
        Real* dbuf = (Real*) buf;
        for (int i=0; i<actInfo.dim; i++)
            dbuf[i] = valsContinuous[i];
    }
    
    void unpack(byte* buf)
    {
        Real* dbuf = (Real*) buf;
        for (int i=0; i<actInfo.dim; i++) {
            valsContinuous[i] = dbuf[i];
            realActionToIndex(i);
        }
    }

    //from action indices to unique label (for tables, DQN)
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
            lab     = lab%actInfo.shifts[i];
            indexToRealAction(i);
            //printf("%d %d %f\n",lab,vals[i],valsContinuous[i]);
        }
    }
    
    void set(vector<Real> data)
    {
        for (int i=0; i<actInfo.dim; i++) {
            valsContinuous[i] = data[i];
            realActionToIndex(i);
        }
    }
    
    void getRandom(const int iRand = -1)
    {
        std::normal_distribution<Real> dist(0.,0.5);
        
        if ( iRand<0 || iRand >= actInfo.dim )
        {
            for (int i=0; i<actInfo.dim; i++) {
                const Real uB = actInfo.values[i].back();
                const Real lB = actInfo.values[i].front();
                valsContinuous[i]=lB+.5*(std::tanh(dist(*gen))+1.)*(uB-lB);
                realActionToIndex(i);
            }
        }
        else
        {
            const Real uB = actInfo.values[iRand].back();
            const Real lB = actInfo.values[iRand].front();
            valsContinuous[iRand]=lB+.5*(std::tanh(dist(*gen))+1.)*(uB-lB);
            realActionToIndex(iRand);
        }
    }
    
    vector<Real> scale() const
    {
        vector<Real> res(actInfo.dim);
        for (int i=0; i<actInfo.dim; i++) res[i] = realToScaledReal(i);
        //printf("%f %f\n",res[0],valsContinuous[0]);
        return res;
    }
    
    void descale(vector<Real> data)
    {
        for (int i=0; i<actInfo.dim; i++) {
            valsContinuous[i] = scaledRealToReal(data[i], i);
            realActionToIndex(i);
        }
    }
};

