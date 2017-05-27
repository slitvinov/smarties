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
#include <algorithm>
#include <sstream>
#include <math.h>

using namespace std;
using namespace ErrorHandling;


struct StateInfo
{
	Uint dim, dimUsed;
	vector<bool> inUse;
	vector<Real> mean, scale;

    StateInfo& operator= (const StateInfo& stateInfo)
    {
        dim     = stateInfo.dim;
        dimUsed = stateInfo.dimUsed;
        assert(dimUsed<=dim);
        inUse.resize(dim);
        for (Uint i=0; i<dim; i++)
        	inUse[i] = (stateInfo.inUse[i]);
        return *this;
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
		for (Uint i=0; i<sInfo.dim; i++) vals[i] = s.vals[i];
		return *this;
	}

	inline string print() const
	{
		ostringstream o;
		o << "[";
		for (Uint i=0; i<sInfo.dim; i++) {
			o << vals[i];
			if (i < sInfo.dim-1) o << " ";
		}
		o << "]";
		return o.str();
	}

    inline string printClean() const
	{
		ostringstream o;
		for (Uint i=0; i<sInfo.dim; i++) {
			o << vals[i]<< " ";
		}
		return o.str();
  }

  inline void copy_observed(vector<Real>& res, const Uint append=0) const
  {
			//copy state into res, append is used to chain multiple states together
      Uint k = append*sInfo.dimUsed;
			assert(res.size() >= k+sInfo.dimUsed);
      for (Uint i=0; i<sInfo.dim; i++)
      if (sInfo.inUse[i]) {
          res[k] = vals[i];
          k++;
      }
  }

	inline vector<Real> copy_observed() const
  {
			vector<Real> ret(sInfo.dimUsed);
      for (Uint i=0, k=0; i<sInfo.dim; i++)
      if (sInfo.inUse[i]) ret[k++] = vals[i];
			return ret;
  }

  inline void copy(vector<Real>& res) const
  {
      for (Uint i=0; i<sInfo.dim; i++)
          res[i] = vals[i];
  }

	//pack and unpack for MPI comm
  inline void pack(double*const buf) const
  {
      for (Uint i=0; i<sInfo.dim; i++)
			buf[i] = (double)vals[i];
  }

  inline void unpack(const double*const buf)
  {
      for (Uint i=0; i<sInfo.dim; i++)
			vals[i] = (Real)buf[i];
  }

  inline void set(const vector<Real> data)
  {
      for (Uint i=0; i<sInfo.dim; i++)
          vals[i] = data[i];
  }
};

struct ActionInfo
{
	Uint dim, maxLabel; //number of actions per turn
	vector<Uint> bounded; //whether action have a lower && upper bounded (bool)
	//vector<int> boundedTOP, boundedBOT; TODO

	//each component of action vector has a vector of possible values that action can take with DQN
  vector<vector<Real>> values; //max and min of this vector also used for rescaling
	vector<Uint> shifts; //used by DQN to map int to an (entry in each component of values)

  ActionInfo() {}

  ActionInfo& operator= (const ActionInfo& actionInfo)
	{
      dim = actionInfo.dim;
			maxLabel = actionInfo.maxLabel;
      assert(actionInfo.values.size() ==dim);
			assert(actionInfo.shifts.size() ==dim);
			assert(actionInfo.bounded.size()==dim);
      values = actionInfo.values;
      shifts = actionInfo.shifts;
			bounded = actionInfo.bounded;
      assert(values.size()==dim && shifts.size()==dim);
      return *this;
  }

	inline Uint nDiscrVals(const Uint i)  const
	{
		return values[i].size();
	}

	void updateShifts()
	{
		shifts.resize(dim);
    shifts[0] = 1;
    for (Uint i=1; i < dim; i++) {
        assert(nDiscrVals(i) == values[i].size());
        shifts[i] = shifts[i-1] * nDiscrVals(i-1);
    }
		maxLabel = shifts[dim-1] * nDiscrVals(dim-1);
	}

	inline Real getActMaxVal(const Uint i) const
	{
		assert(i<dim && dim==values.size());
		assert(values[i].size()>1); //otherwise scaling is impossible
		return *std::max_element(std::begin(values[i]), std::end(values[i]));
	}

	inline Real getActMinVal(const Uint i) const
	{
		assert(i<dim && dim==values.size());
		assert(values[i].size()>1); //otherwise scaling is impossible
		return *std::min_element(std::begin(values[i]), std::end(values[i]));
	}

	inline Real getScaled(const Real unscaled, const Uint i) const
	{
		//unscaled value and i is to which component of action vector it corresponds
		//if action space is bounded, return the scaled component, else return unscaled
		//scaling is between max and min of values vector (user specified in environment)
		//scaling function is x/(1+abs(x)) (between -1 and 1 for x in -inf, inf)
		Real ret;
		const Real min_a = getActMinVal(i);
		const Real max_a = getActMaxVal(i);
		assert(max_a-min_a > std::numeric_limits<Real>::epsilon());
		if (bounded[i]) {
			const Real soft_sign = unscaled/(1. + std::fabs(unscaled));
			ret = min_a + 0.5*(max_a - min_a)*(soft_sign + 1);
		} else {
			ret = min_a + 0.5*(max_a - min_a)*(unscaled + 1);
		}
		return ret;
	}

	inline Real getDactDscale(const Real unscaled, const Uint i) const
	{
		//derivative of scaled action wrt to unscaled action, see getScaled()
		Real ret = 1;
		const Real min_a = getActMinVal(i);
		const Real max_a = getActMaxVal(i);
		if (bounded[i]) {
			const Real denom = 1. + std::fabs(unscaled);
			ret = 0.5*(max_a-min_a)/denom/denom;
    } else {
			ret = 0.5*(max_a-min_a);
		}
		return ret;
	}

	inline vector<Real> getScaled(vector<Real> unscaled) const
	{
		//see per-component getScaled
		vector<Real> ret(dim);
		assert(unscaled.size()==dim);
		for (Uint i=0; i<dim; i++) ret[i] = getScaled(unscaled[i], i);

		return ret;
	}

	inline vector<Real> getInvScaled(vector<Real> scaled) const
	{
		//opposite operation
		vector<Real> ret = scaled;
		assert(ret.size()==dim);
		for (Uint i=0; i<dim; i++) {
				const Real min_a = getActMinVal(i);
				const Real max_a = getActMaxVal(i);
				assert(max_a-min_a > std::numeric_limits<Real>::epsilon());
				if (bounded[i]) {
					assert(scaled[i]>min_a && scaled[i]<max_a);
					const Real y = 2*(scaled[i] - min_a)/(max_a - min_a) -1;
					assert(std::fabs(y) < 1);
					ret[i] =  y/(1.-std::fabs(y));
				} else {
					ret[i] = 2*(scaled[i] - min_a)/(max_a - min_a) -1;
				}
		}

		return ret;
	}

  inline Uint actionToLabel(const vector<Real> vals) const
	{
		//map from discretized action (entry per component of values vectors) to int
      Uint lab=0;
      for (Uint i=0; i<dim; i++)
				lab += shifts[i]*realActionToIndex(vals[i],i);
      return lab;

		#ifndef NDEBUG
			vector<Uint> test(dim);
			Uint max = 1;
			for (Uint i=0; i < dim; i++) {
					test[i] = i==0 ? 1 : test[i-1] * nDiscrVals(i-1);
					assert(test[i] == shifts[i]);
					max *= nDiscrVals(i);
			}
			assert(max == maxLabel);
		#endif
  }

  inline vector<Real> labelToAction(Uint lab) const
  {
		//map an int to the corresponding entries in the values vec
  	vector<Real> ret(dim);
    for (Uint i=dim; i>0; i--) {
			Uint tmp = lab/shifts[i-1]; //in opposite op: add shifts*index
      ret[i-1] = indexToRealAction(tmp, i-1);
      lab = lab % shifts[i-1];
    }
    return ret;
  }

  inline Real indexToRealAction(const Uint lab, const Uint i) const
	{
    	assert(i<values.size() && lab<values[i].size());
			return values[i][lab];
	}

	inline Uint realActionToIndex(const Real val, const Uint i) const
	{
		//From continous action for i-th component of action vector
		// convert to an entry in values vector
		Real dist = 1e9; Uint ret = 0;
		for (Uint j=0; j<nDiscrVals(i); j++) {
			const Real _dist = std::fabs(values[i][j]-val);
			if (_dist<dist) { dist = _dist; ret = j; }
		}
		return ret;
	}

	inline Real getUniformProbability() const
	{
		Real P = 1;
		for (Uint i=0; i<dim; i++) {
			const Real lB = getActMinVal(i);
			const Real uB = getActMaxVal(i);
			assert(uB-lB > std::numeric_limits<Real>::epsilon());
			P /= (uB-lB);
		}
		return P;
	}
	inline Real addedVariance(const Real i) const
	{
			return 1;
	    //if (bounded[i]) return 1; //enough to sample sigmoid
	    //else return getActMaxVal(i) - getActMinVal(i);
	}
};

class Action
{
 public:
	ActionInfo actInfo;
  vector<Real> vals;
  mt19937 * gen;

	Action(const ActionInfo& newActInfo, mt19937 * g) :
	actInfo(newActInfo), gen(g)
	{
		vals.resize(actInfo.dim);
	}

	Action& operator= (const Action& a)
	{
		if (actInfo.dim != a.actInfo.dim)
			die("Dimension of actions differ!!!\n");
		for (Uint i=0; i<actInfo.dim; i++)
			vals[i] = a.vals[i];
		return *this;
	}

	inline string print() const
	{
		ostringstream o;
		o << "[";
		for (Uint i=0; i<actInfo.dim-1; i++) o << vals[i] << " ";
		o << vals[actInfo.dim-1];
        o << "]";
		return o.str();
	}

    inline string printClean() const
	{
        ostringstream o;
		for (Uint i=0; i<actInfo.dim; i++)   o << vals[i] << " ";
		return o.str();
	}

    //pack and unpack for MPI comm
    inline void pack(double*const buf) const
    {
        for (Uint i=0; i<actInfo.dim; i++)
				buf[i] = (double)vals[i];
    }

    inline void unpack(const double*const buf)
    {
        for (Uint i=0; i<actInfo.dim; i++)
				vals[i] = (Real)buf[i];
    }

    inline void set(vector<Real> data)
    {
			assert(data.size() == actInfo.dim);
			vals = data;
    }

    inline void set(const Uint label)
    {
			vals = actInfo.labelToAction(label);
    }

		inline void getRandom(int iRand = -1)
		{
			if (iRand<0) iRand = actInfo.dim;
			return getRandom(static_cast<Uint> (iRand));
		}

    inline void getRandom(const Uint iRand)
    {
				std::uniform_real_distribution<Real> dist(0,1);
				if (iRand >= actInfo.dim ) {
        	//select all random actions
            for (Uint i=0; i<actInfo.dim; i++) {
							const Real lB = actInfo.getActMinVal(i);
							const Real uB = actInfo.getActMaxVal(i);
            	vals[i] = lB + dist(*gen)*(uB-lB);
            }
        } else {  //select just one
					const Real lB = actInfo.getActMinVal(iRand);
					const Real uB = actInfo.getActMaxVal(iRand);
					vals[iRand] = lB + dist(*gen)*(uB-lB);
        }
    }

    inline Uint getActionLabel() const
    {
    	return actInfo.actionToLabel(vals);
    }
};
