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

const double eps = 1e-9;

MultiTable::MultiTable(StateInfo newSInfo, ActionInfo newActInfo, double gamma=.9) : QApproximator(newSInfo, newActInfo), actionsIt(newActInfo), gamma(gamma)
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

double MultiTable::get(const State& s, const Action& a, int nAgent)
{
	long int id = _encodeIdx(s, a);
    //_info("final ID %d\n", id);
    //printf("Final ID from normal method=%d\n\n",id);
	if (data.find(id) == data.end()) return 0; 
	return data.find(id)->second;
}

double MultiTable::get(const State * s, const Action * a, int nAgent)
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

double MultiTable::getsmooth(const State& s, const Action& a, int nAgent)
{
    long int id = _encodeIdx(s, a);
    //_info("final ID %d\n", id);
    if (data.find(id) == data.end()) return 0;
    if (data.find(id)->second == 0.0)
    {
        double avg = 0.0;
        double wgt = 0.0;
        for (int i=0; i<dim; ++i)
        {
            if (data.find(id + shifts[i]) != data.end() && id - shifts[i]>=0)
            {
                double Qplus = data.find(id + shifts[i])->second;
                double Qminu = data.find(id - shifts[i])->second;
                wgt += 1.0;
                avg += .5*(Qplus+Qminu);
            }
        }
        warn("Trying to interpolate state-action using %f adjacent states. This is bad if you are offline!\n", wgt);
        return avg/wgt;
    }
    return data.find(id)->second;
}

double MultiTable::getMax(const State& s, int nAgent)
{
	long int id = _encodeState(s, _discretize);
	if (maxStateVal.find(id) == maxStateVal.end()) return 0; 
	return maxStateVal[id];
}

double  MultiTable::Train()
{
    Action a(actInfo);
    debug("Offline training of multitable with %d samples.\n", samples.Set.size());
    double err(0.0);
    
    for (int i=0; i<samples.Set.size(); i++)
    { //target values
        //we transition to state s' and get the Qold
        
        double Qold = get(samples.Set[i].sOld, samples.Set[i].a, 0);
        //double Qtest = get(*samples.Set[i].sOld, *samples.Set[i].a, 0);
        
        actionsIt.reset();
        double best = -1e10;
        while (!actionsIt.done())
        {
            a = actionsIt.next();
            double test = get(*samples.Set[i].sNew, a, 0);
            if (test >= best + 1e-12)
            {
                best = test; // best current Q option
                actionsIt.memorize();
            }
        }
        double reward = samples.Set[i].reward - fabs(samples.Set[i].sOld->vals[1]) - fabs(samples.Set[i].sOld->vals[2])/1.57079632679;
        double Qnew = reward + gamma*best;
        
        double target = 0.1*(Qnew - Qold);
        correct(*samples.Set[i].sOld, *samples.Set[i].a, target, 0);
        err += fabs(target);
    }
    
    debug("Learning state: average error %f.\n", err/samples.Set.size());
    return err/samples.Set.size();
}

void MultiTable::set(const State& s, const Action& a, double val, int nAgent)
{
	long int sId = _encodeState(s, _discretize);
	long int id = _encodeIdx(sId, a);
		
	if (fabs(val) > eps)
	{
		data[id] = val;
	//	if (val > maxStateVal[sId]) maxStateVal[sId] = val;
	}
}

void MultiTable::correct(const State& s, const Action& a, double err, int nAgent)
{
	long int sId = _encodeState(s, _discretize);
	long int id = _encodeIdx(sId, a);
	//_info("sID %d, final ID %d\n", sId, id);
	data[id] += err;
	//	if (val > maxStateVal[sId]) maxStateVal[sId] = val;
}

double MultiTable::usage() const
{
	return (double) data.size() / (shifts[0] * actInfo.bounds[0]);
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
	for(map<long int, double>::iterator it = data.begin(); it != data.end(); it++)
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
			double dummy;
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
	
