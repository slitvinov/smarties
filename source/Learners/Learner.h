/*
 *  Learner.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 15.07.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <string>
#include <list>

#include "../Agents/Agent.h"
#include "../Environments/Environment.h"
#include "../MRAGProfiler.h"
#include "../Savers/Saver.h"

using namespace std;

class Learner
{
protected:
	double dt;
	System system;
	vector<Agent*> agents;
		
	list<Saver*> savers;
	
	MRAG::Profiler* profiler;
	
	inline void execSavers(double t)
	{
		for (list<Saver*>::iterator it = savers.begin(), end = savers.end(); it != end; ++it)
		{
			if ( ((int)(t/dt) % (*it)->getPeriod()) == 0) (*it)->exec();
		}	
	}
	
	
public:
	Learner(System newSystem, double newDt, MRAG::Profiler* newProfiler = NULL) :
	system(newSystem), agents(newSystem.agents), dt(newDt), profiler(newProfiler) { };
	
	virtual void evolve(double t) = 0;
	virtual void savePolicy(string prefix)  = 0;
	virtual void try2restart(string prefix) = 0;

	inline void registerSaver(Saver* saver, int period)
	{
		saver->setEnvironment(system.env);
		saver->setPeriod(period);
		savers.push_back(saver);
	}
	
};