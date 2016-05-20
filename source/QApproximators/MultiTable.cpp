 /*
 *  MultiTable.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm> 
#include <vector>
#include <cmath>

#include "MultiTable.h"
#include "../ErrorHandling.h"
#include "../Misc.h"

using namespace ErrorHandling;

const Real eps = 1e-9;

MultiTable::MultiTable(StateInfo SInfo, ActionInfo ActInfo, Settings & settings) : QApproximator(SInfo, ActInfo, settings), actionsIt(ActInfo)
{
	dim = sInfo.dim + actInfo.dim;
	
	if (dim == 0) return;
	
	shifts.resize(dim);
	shifts[0] = 1;

	for (int i=1; i < sInfo.dim+1; i++)
		shifts[i] = (long int)shifts[i-1] * sInfo.bounds[i-1];
	
	for (int i = sInfo.dim+1; i<dim; i++)
		shifts[i] = (long int)shifts[i-1] * actInfo.bounds[i - sInfo.dim - 1];
	
	data.clear();
	maxStateVal.clear();
}

MultiTable::~MultiTable()
{
	dim = 0;
	data.clear();
}

inline long int MultiTable::_encodeIdx(const State& s, const Action& a) const
{
	long int res = _encodeState(s, _discretize);
	return _encodeIdx(res, a);
}

inline long int MultiTable::_encodeIdx(const long int sId, const Action& a) const
{
	long int res = sId;
	
	for(int i=0; i<actInfo.dim; i++)
		res += a.vals[i] * shifts[i+sInfo.dim];
	
	return res;
}

template <typename F>
inline long int MultiTable::_encodeState(const State& s, F&& _discr) const
{
	long int res = 0;
	for(int i=0; i<sInfo.dim; i++)
		res += _discr(s.vals[i], s.sInfo.bottom[i], s.sInfo.top[i], s.sInfo.bounds[i], s.sInfo.belowBottom[i], s.sInfo.aboveTop[i]) * shifts[i];
	
	return res;
}

Real MultiTable::get(const State& s, const Action& a, int iAgent)
{
	long int id = _encodeIdx(s, a);
    //_info("final ID %d\n", id);
    //printf("Final ID from normal method=%d\n\n",id);
	if (data.find(id) == data.end()) return 0; 
	return data.find(id)->second;
}

Real MultiTable::get(const State * s, const Action * a, int iAgent)
{
    long int res = 0;
    
    for(int i=0; i<sInfo.dim; i++)
    {
        res += _discretize(s->vals[i], s->sInfo.bottom[i], s->sInfo.top[i], s->sInfo.bounds[i], s->sInfo.belowBottom[i], s->sInfo.aboveTop[i]) * shifts[i];
        //printf("sval %f, sid %d shift %d , ",s->vals[i],_discretize(s->vals[i], s->sInfo.bottom[i], s->sInfo.top[i], s->sInfo.bounds[i], s->sInfo.belowBottom[i], s->sInfo.aboveTop[i]), shifts[i]);
    }

    for(int i=0; i<actInfo.dim; i++)
    {
        res += a->vals[i] * shifts[i+sInfo.dim];
        //printf("aid %d shift %d , ",a->vals[i], shifts[i+sInfo.dim]);
    }
    //printf("\n Final ID from pointer method=%d\n",res);
    if (data.find(res) == data.end()) return 0;
    return data.find(res)->second;
}

void MultiTable::get(const State& sOld, vector<Real> & Qold, const State& s, vector<Real> & Q, int iAgent)
{
    int j = 0;
       Q.resize(actionsIt.actInfo.bounds[0]);
    Qold.resize(actionsIt.actInfo.bounds[0]);
    
    actionsIt.reset();
    while (!actionsIt.done())
    {
        Q[j] = get(s, actionsIt.next());
        Qold[j] = get(sOld, actionsIt.next());
        j++;
    }
}

Real MultiTable::getMax(const State& s, Action& a, int iAgent)
{
	long int id = _encodeState(s, _discretize);
	if (maxStateVal.find(id) == maxStateVal.end()) return 0;
	return maxStateVal[id];
}

void MultiTable::Train(const vector<vector<Real>> & sOld, const vector<int> & a, const vector<Real> & r, const vector<vector<Real>> & s)
{
    /*
    Action a(actInfo);
    debug("Offline training of multitable with %d samples.\n", samples->Set.size());
    Real err(0.0);
    vector<Real> Qold, Q;
    int Nbest, NoldBest;
    
    for (int i=0; i<samples->Set.size(); i++)
    { //target values
        //we transition to state s' and get the Qold
        Real Vnew(-1e10), Vold(-1e10);
        
        get(samples->Set[i].sOld, Qold, samples->Set[i].sNew, Q);
        
        for (int i=0; i<Q.size(); i++)
        {
            if (Q[i]>Vnew)
            {
                Nbest = i;
                Vnew = Q[i];
            }
            if (Qold[i]>Vold)
            {
                NoldBest = i;
                Vold = Qold[i];
            }
        }
        
        a.vals[0] = Nbest;
        Real Aold = Qold(a.vals[0]);
        
        //Real err = (Vold + (r + gamma*Vnew - Vold)/.2 - Aold);
        Real target = (samples->Set[i].reward + gamma*Vnew - Aold);
        correct(samples->Set[i].sOld, samples->Set[i].a, target, 0);
        err += fabs(target);
    }
    
    debug("Learning state: average error %f.\n", err/samples.Set.size());
    //return err/samples.Set.size();
  */
}

void MultiTable::set(const State& s, const Action& a, Real val, int iAgent)
{
	long int sId = _encodeState(s, _discretize);
	long int id = _encodeIdx(sId, a);
		
	if (fabs(val) > eps)
	{
		data[id] = val;
	//	if (val > maxStateVal[sId]) maxStateVal[sId] = val;
	}
}

void MultiTable::correct(const State& s, const Action& a, Real err, int iAgent)
{
    err *= lRate;
	long int sId = _encodeState(s, _discretize);
	long int id = _encodeIdx(sId, a);
	//_info("sID %d, final ID %d\n", sId, id);
	data[id] += err;
	//	if (val > maxStateVal[sId]) maxStateVal[sId] = val;
}

Real MultiTable::usage() const
{
	return (Real) data.size() / (shifts[0] * actInfo.bounds[0]);
}

void MultiTable::save(string fname)
{
	_info("save %s\n", fname.c_str());
	
	string nameBackup = fname + "_backup";
	ofstream out(nameBackup.c_str());
	
	if (!out.good()) die("Unable to open save into file %s\n", fname.c_str());
	
	out.precision(20);
	
	out << sInfo.dim << "  ";
	for(int i=0; i<sInfo.dim; i++)
		out << sInfo.bounds[i] << "  ";
	out << endl;
	
	out << actInfo.dim << "  ";
	for(int i=0; i<actInfo.dim; i++)
		out << actInfo.bounds[i] << "  ";
	out << endl;
	
	unsigned int counter = 0;
	for(map<long int, Real>::iterator it = data.begin(); it != data.end(); it++)
		if (fabs(it->second) > eps)
		{
			out << it->first << " " << scientific << it->second << endl;
			out.flush();
			counter++;
		}
	out.flush();
	out.close();
		
	// Prepare copying command
	std::string command = "cp ";
	std::string nameOriginal = fname;
	command = command + nameBackup + " " + nameOriginal;
	
	// Submit the command to the system
	FILE *ptr = popen(command.c_str(), "r");
	//if( ptr != NULL){ std::cout << command << std::endl; std::cout << "successful policy backup!" << std::endl; }
	//else{ std::cout << command << std::endl; std::cout << "policy backup failed" << std::endl; }
	pclose( ptr );
}

bool MultiTable::restart(string fname)
{
	bool res = true;
	
	if(dim<=0) die("Policy dimension was not set!\n");
	
	string nameBackup = fname;// + "_backup";
		
	ifstream in(nameBackup.c_str());
	_info("%s\n", nameBackup.c_str());
	
	if(in.good())
	{
		int sDim, aDim;
		
		// Reading state info
		in >> sDim;
		if (sInfo.dim != sDim ) die("Saved state and current state do not match in dimensionality!\n");
		else                    _info("State dimension = %d\n", sDim);
		
		for(int i=0; i<sDim; i++)
		{
			int dummy = 0;
			in >> dummy;
			if(dummy != sInfo.bounds[i])  die("Saved state and current state do not match in dimensionality!\n");
			else                          _info("StateDim[%d] = %d\n", i, dummy);
		}
		
		// Reading action info
		in >> aDim;
		if (actInfo.dim != aDim ) die("Saved action and current action do not match in dimensionality!\n");
		else                      _info("Action dimension = %d\n", aDim);
		
		for(int i=0; i<aDim; i++)
		{
			int dummy = 0;
			in >> dummy;
			if(dummy != actInfo.bounds[i])  die("Saved action and current action do not match in dimensionality!\n");
			else                            _info("ActionDim[%d] = %d\n", i, dummy);
		}
		
		unsigned counter = 0;
		while (in.good())
		{
			long int key;
			Real dummy;
			in >> key >> dummy;
			data[key] = dummy;
			counter++;
			
			debug("%ld ", key);
			debug("---> %10.10e\n", dummy);
		}
	}
	else
	{
		error("WTF couldnt open file '%s' %s\n", fname.c_str(), " (ok keep going mofo)!\n");
		res = false;
	}
	
	in.close();
	
	return res;
}
	
