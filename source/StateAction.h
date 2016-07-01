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
	vector<Real> bottom, top;
	vector<bool> belowBottom, aboveTop, isLabel, inUse;
    vector<Real> values;
    
    StateInfo& operator= (const StateInfo& stateInfo)
    {
        dim     = stateInfo.dim;
        dimUsed = stateInfo.dimUsed;
        bounds.resize(dim); bottom.resize(dim); top.resize(dim);
        belowBottom.resize(dim); aboveTop.resize(dim); isLabel.resize(dim); inUse.resize(dim);
        
        for (int i=0; i<dim; i++)
        {
            bounds[i]=     (stateInfo.bounds[i]);
            bottom[i]=     (stateInfo.bottom[i]);
            top[i]=        (stateInfo.top[i]);
            isLabel[i]=    (stateInfo.isLabel[i]);
            inUse[i]=      (stateInfo.isLabel[i]);
        }
        
        //values.clear();
        //for (int i=0; i<stateInfo.values.size(); i++)
        //    values.push_back(stateInfo.values[i]);
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
            //if (i < sInfo.dim-1) o << " ";
		}
		return o.str();
	}
	
	string printScaled()
	{
		ostringstream o;
		o << "[";
		for (int i=0; i<sInfo.dim; i++) {
            Real res = 2.*(vals[i]-sInfo.bottom[i]) / (sInfo.top[i] - sInfo.bottom[i]) - 1.;
			//if (sInfo.isLabel[i]) res = sInfo.values[vals[i]];
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
            //if (sInfo.isLabel[i]) res[k] = sInfo.values[vals[i]];
            k++;
        }
        
    }
    
	void scale(vector<Real>& res) const
	{
		for (int i=0; i<sInfo.dim; i++) {
            res[i] = 2.*(vals[i]-sInfo.bottom[i]) / (sInfo.top[i] - sInfo.bottom[i]) - 1.;
            //if (sInfo.isLabel[i]) res[i] = sInfo.values[vals[i]];
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
        for (int i=0; i<sInfo.dim; i++) {
            vals[i] = data[i];
            //if (sInfo.isLabel[i]) vals[i] = sInfo.values[round(data[i])];
        }
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
    bool realValues; //finite set, continuous
	int dim; //number of actions per turn
    
    //discrete actions
    int zeroact; //if finite set: one that corresponds to 0
	vector<int> bounds, shifts; //if finite set, number of choices per "dim"
    vector<vector<Real>> values; //used for rescaling, would be used if action is input to NN
    
    //continuous actions
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
        //if (actInfo.realValues) {
            for (int i=0; i<actInfo.dim-1; i++)
                o << valsContinuous[i] << " ";
            o << valsContinuous[actInfo.dim-1];
        /*} else {
            for (int i=0; i<actInfo.dim-1; i++)
                o << vals[i] << " ";
            o << vals[actInfo.dim-1];
        }*/
        o << "]";
		return o.str();
	}
    
    string printClean() const
	{
        ostringstream o;
        //if (actInfo.realValues) {
            for (int i=0; i<actInfo.dim; i++)
                o << valsContinuous[i] << " ";
        /*} else {
            for (int i=0; i<actInfo.dim; i++)
                o << vals[i] << " ";
        }*/
		return o.str();
	}
    
    void pack(byte* buf) const
    {
        //if (actInfo.realValues) {
            Real* dbuf = (Real*) buf;
            for (int i=0; i<actInfo.dim; i++)
                dbuf[i] = valsContinuous[i];
        /*} else {
            int* dbuf = (int*) buf;
            for (int i=0; i<actInfo.dim; i++)
                dbuf[i] = vals[i];
        }*/
    }
    
    void unpack(byte* buf)
    {
        //if (actInfo.realValues) {
            Real* dbuf = (Real*) buf;
            for (int i=0; i<actInfo.dim; i++)
                valsContinuous[i] = dbuf[i];
        /*} else {
            int* dbuf = (int*) buf;
            for (int i=0; i<actInfo.dim; i++)
                vals[i] = dbuf[i];
        }*/
    }

    void set(vector<Real> data)
    {
        for (int i=0; i<actInfo.dim; i++) {
            valsContinuous[i] = data[i];
            vals[i] = (actInfo.bounds[i]-1)*(data[i]-actInfo.values[i].front()) /
                           (actInfo.values[i].back()-actInfo.values[i].front()) +.49;
            if (vals[i]<0) vals[i]=0;
            if (vals[i] > actInfo.bounds[i]-1) vals[i] = actInfo.bounds[i]-1;
            //printf("%f %d\n", valsContinuous[i],vals[i]);
        }
    }
    
    /*
    void set(vector<int> data)
    {
        for (int i=0; i<actInfo.dim; i++)
        {
            valsContinuous[i] = actInfo.values[i][round(data[i])];
            vals[i] = data[i];
        }
    }
    */
    
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
    /*
    void scale(vector<Real>& res) const
	{
		for (int i=0; i<actInfo.dim; i++)
            res[res.size() - actInfo.dim + i] = actInfo.values[i][vals[i]];
	}*/
};

/*
struct TraceContainer
{
    int maxLen, nAgents;
    
    real *stData, *rData;
    int  *actData;
    bool *termData;
    
    //vector<real> es;
    vector<int> start;
    
    TraceContainer(int maxLen, int nAgents, int sdims) :
    maxLen(maxLen), nAgents(nAgents), sdims(sdims), start(maxLen)//, es(maxLen)
    {
        start.resize(maxLen);
        posix_memalign((void**)&stData,  64, sizeof(real) * maxLen * nAgents * sdims);
        posix_memalign((void**)&rData,   64, sizeof(real) * maxLen * nAgents);
        posix_memalign((void**)&actData, 64, sizeof(int)  * maxLen * nAgents);
        posix_memalign((void**)&termData,64, sizeof(bool) * maxLen * nAgents);
        
        memset(stData,  0, sizeof(real) * maxLen * nAgents * sdims);
        memset(rData,   0, sizeof(real) * maxLen * nAgents);
        memset(actData, 0, sizeof(int)  * maxLen * nAgents);
        memset(termData,0, sizeof(bool) * maxLen * nAgents);
        
        //es[start] = 1;
        debug("allocated traceContainer for %d agents", nAgents);
    };
    
    inline void getCurrBuffs(real*& states, int*& actions, real*& rewards, bool*& term, const int n)
    {
        if (start[n] >= maxLen || start[n] <0) die("data corruption");
        states  = stData  + (start[n] + n*maxLen)*sdims;
        actions = actData + (start[n] + n*maxLen);
        rewards = rData   + (start[n] + n*maxLen);
        term    = termData+ (start[n] + n*maxLen);
    }
    
    inline void advance(real lambda, const int n)
    {
        //for (auto& e : es) e *= lambda;
        start[n] = (start[n] == maxLen-1) ? 0 : start[n] + 1;
        //es[start] = 1;
    }
    
    inline void access(State& s, Action& a, real& r, bool& final, const int n, const int k) const
    {
        int ind = start[n] - k;
        if (ind < 0)       ind += maxLen;
        if (ind >= maxLen) ind -= maxLen;
        
        s.vals = stData + (ind + n*maxLen)*sdims;
        s.ndims = sdims;
        
        a     =  actData[ind + n*maxLen];
        r     =    rData[ind + n*maxLen];
        final = termData[ind + n*maxLen];
        //e = es[ind];
    }
    
    inline void access(State& s, const int n, const int k) const
    {
        int ind = start[n] - k;
        if (ind < 0)       ind += maxLen;
        if (ind >= maxLen) ind -= maxLen;
        
        s.vals = stData + (ind + n*maxLen)*sdims;
        s.ndims = sdims;
    }
    
    
    inline void access(State& s, bool& final, const int n, const int k) const
    {
        int ind = start[n] - k;
        if (ind < 0)       ind += maxLen;
        if (ind >= maxLen) ind -= maxLen;
        
        s.vals = stData + (ind + n*maxLen)*sdims;
        s.ndims = sdims;
        final = termData[ind + n*maxLen];
    }
    
    inline void access(Action& a, const int n, const int k) const
    {
        int ind = start[n] - k;
        if (ind < 0)       ind += maxLen;
        if (ind >= maxLen) ind -= maxLen;
        
        a = actData[ind + n*maxLen];
    }
    
    inline void set(Action& a, const int n, const int k) const
    {
        int ind = start[n] - k;
        if (ind < 0)       ind += maxLen;
        if (ind >= maxLen) ind -= maxLen;
        
        actData[ind + n*maxLen] = a;
    }
};
*/

