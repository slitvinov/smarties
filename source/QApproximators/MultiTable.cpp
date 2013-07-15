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

MultiTable::MultiTable(StateInfo newSInfo, ActionInfo newActInfo) : QApproximator(newSInfo, newActInfo)
{
	dim = sInfo.dim + actInfo.dim;
	
	if (dim == 0) return;
	
	shifts.resize(dim);
	shifts[dim-1] = 1;
	
	for (int i=dim-2; i>=sInfo.dim; i--)
		shifts[i] = (long int)shifts[i+1] * actInfo.bounds[i+1 - sInfo.dim];
	for (int i=sInfo.dim - 1; i>=0; i--)
		shifts[i] = (long int)shifts[i+1] * sInfo.bounds[i];
	
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
	long int res = _encodeState(s);
	
	return _encodeIdx(res, a);
}

inline long int MultiTable::_encodeIdx(const long int sId, const Action& a) const
{
	long int res = sId;
	
	for(int i=0; i<actInfo.dim; i++)
		res += a.vals[i] * shifts[i+sInfo.dim];
	
	return res;
}

inline long int MultiTable::_encodeState(const State& s) const
{
	long int res = 0;
	int (*_discr) (double, double, double, int, bool, bool) = &_discretize;	
	
	for(int i=0; i<sInfo.dim; i++)
		res += (*_discr)(s.vals[i], s.sInfo.bottom[i], s.sInfo.top[i], s.sInfo.bounds[i], s.sInfo.belowBottom[i], s.sInfo.aboveTop[i]) * shifts[i];
	
	return res;
}

double MultiTable::get(const State& s, const Action& a)
{
	long int id = _encodeIdx(s, a);
	if (data.find(id) == data.end()) return 0; 
	return data.find(id)->second;
}

double MultiTable::getMax(const State& s)
{
	long int id = _encodeState(s);
	if (maxStateVal.find(id) == maxStateVal.end()) return 0; 
	return maxStateVal[id];
}

void MultiTable::set(const State& s, const Action& a, double val)
{
	long int sId = _encodeState(s);
	long int id = _encodeIdx(sId, a);
		
	if (fabs(val) > eps)
	{
		data[id] = val;
	//	if (val > maxStateVal[sId]) maxStateVal[sId] = val;
	}
}

double MultiTable::usage() const
{
	return (double) data.size() / (shifts[0] * actInfo.bounds[0]);
}

void MultiTable::save(string fname)
{
	info("save %s\n", fname.c_str());
	
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
	info("%s\n", nameBackup.c_str());
	
	if(in.good())
	{
		int sDim, aDim;
		
		// Reading state info
		in >> sDim;
		if (sInfo.dim != sDim ) die("Saved state and current state do not match in dimensionality!\n");
		else                   info("State dimension = %d\n", sDim);
		
		for(int i=0; i<sDim; i++)
		{
			int dummy = 0;
			in >> dummy;
			if(dummy != sInfo.bounds[i])  die("Saved state and current state do not match in dimensionality!\n");
			else                         info("StateDim[i] = %d\n", dummy);
		}
		
		// Reading action info
		in >> aDim;
		if (actInfo.dim != aDim ) die("Saved action and current action do not match in dimensionality!\n");
		else                     info("Action dimension = %d\n", aDim);
		
		for(int i=0; i<aDim; i++)
		{
			int dummy = 0;
			in >> dummy;
			if(dummy != actInfo.bounds[i])  die("Saved action and current action do not match in dimensionality!\n");
			else                           info("ActionDim[i] = %d\n", dummy);
		}
		
		unsigned counter = 0;
		while (in.good())
		{
			long int key;
			double dummy;
			in >> key >> dummy;
			data[key] = dummy;
			counter++;
			
			debug("%d ", key);
			debug("---> %10.10e\n", dummy);
		}
	}
	else
	{
		error("WTF couldnt open file %s %s\n", fname.c_str(), " (ok keep going mofo)!\n");
		res = false;
	}
	
	in.close();
	
	return res;
}
	
