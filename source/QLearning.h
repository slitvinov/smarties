/*
 *  QLearning.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <string>
#include <list>

#include "Agents/Agent.h"
#include "Environments/Environment.h"
#include "MultiTable.h"
#include "QApproximator.h"
#include "ANNApproximator.h"
#include "MRAGProfiler.h"
#include "Savers/Saver.h"

using namespace std;

class QLearning
{
private:
	System& system;
	vector<Agent*> agents;
	vector<QApproximator*> Qarray;
	vector<State> s0, s1;
	vector<Action> a0, a1;
	vector<double> r;
	vector<double> bestActionVals;
	vector<ActionIterator> actionsIt;
	map<string, QApproximator*> QMap;
	
	list<Saver*> savers;
	
	double gamma, greedyEps, lRate, dt;
	
	MRAG::Profiler* profiler;
	RNG* rng;
	
	void agentsChoose(double t);
	void agentsUpdate(double t);
	void agentsAct(double t);
	void agentsMove();
	
	void execSavers(double t);
	
public:
	QLearning(System newSystem, double newGamma, double newGreedyEps, double newLRate, double newDt, MRAG::Profiler* profiler = NULL);
	void evolve(double t);
	void savePolicy(string prefix);
	void try2restart(string prefix);
	
	void registerSaver(Saver* saver, int period);
};