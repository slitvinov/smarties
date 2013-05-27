/*
 *  QLearning.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <map>

#include "StateAction.h"
#include "QLearning.h"
#include "Settings.h"

// TODO: action iterator

QLearning::QLearning(System newSystem, double newGamma, double newGreedyEps, double newLRate, double newDt, MRAG::Profiler* newProfiler) :
system(newSystem), agents(newSystem.agents), gamma(newGamma), greedyEps(newGreedyEps), lRate(newLRate), dt(newDt), profiler(newProfiler)
{
	// All agents of the same type share same policy
		
	for (int i=0; i<agents.size(); i++)
	{
		s0.push_back(*(new State (agents[i]->getStateDims())));
		s1.push_back(*(new State (agents[i]->getStateDims())));
		a0.push_back(*(new Action(agents[i]->getActionDims())));
		a1.push_back(*(new Action(agents[i]->getActionDims())));
		actionsIt.push_back(*(new ActionIterator(agents[i]->getActionDims())));
		
		string name = agents[i]->getName();
		if (agents[i]->getType() != IDLER && QMap.find(name) == QMap.end())
			QMap[name] = new MultiTable(agents[i]->getStateDims(), agents[i]->getActionDims());
	}
	
	if  (settings.randSeed == -1 )  srand(time(NULL));
	else							srand(settings.randSeed);
	rng = new RNG(rand());
	
	bestActionVals.resize(agents.size());
	r.resize(agents.size());
}

void QLearning::agentsChoose(double t)
{
	debug("Agents choose best actions\n");
	int n = agents.size();
	
#pragma omp parallel for
	for (int i = 0; i<n; i++)
	{
		Agent* agent = agents[i];
		if (agent->getType() == IDLER || t - agent->getLastLearned() < agent->getLearningInterval()) continue;

		MultiTable* Q = QMap[agent->getName()];
		ActionIterator* actions = &(actionsIt[i]);
		double best = -1e10;
		
		actions->reset();
		while (!actions->done())
		{
			double val;
			if ((val = Q->get(s1[i], actions->next())) > best)
			{
				best = val;
				actions->memorize();
			}
		}
		bestActionVals[i] = best;
		a1[i] = actions->recall();
		
		debug1("\n   Agent of type %s, #%d\n", agent->getName().c_str(), i);
		debug1("Best action is %s  with value %f\n", a1[i].print().c_str(), best);
		
		r[i]  = agent->getReward();
		agent->getState(s1[i]);
	}
}

void QLearning::agentsUpdate(double t)
{
	debug("Agents update current policy\n");
	
	for (int i = 0; i<agents.size(); i++)
	{
		Agent* agent = agents[i];
		if (agent->getType() == IDLER || t - agent->getLastLearned() < agent->getLearningInterval()) continue;

		MultiTable* Q = QMap[agent->getName()];
		
		double Qsa = Q->get(s0[i], a0[i]);
		Q->set(s0[i], a0[i], Qsa + lRate * (r[i] + gamma * bestActionVals[i] - Qsa));
		
		debug1("\n   Agent of type %s, #%d\n", agent->getName().c_str(), i);
		debug1("Prev state: %s\n", s0[i].print().c_str());
		debug1("Curr state: %s\n", s1[i].print().c_str());
		debug1("Reward: %f\n", r[i]);
		debug1("Q(s, a): %f --> %f\n", Qsa, Qsa + lRate * (r[i] + gamma * bestActionVals[i] - Qsa));
	}
}

void QLearning::agentsAct(double t)
{
	debug("Agents act according to best possible action\n");
	int n = agents.size();

#pragma omp parallel for
	for (int i = 0; i<n; i++)
	{
		Agent* agent = agents[i];
		if (agent->getType() == IDLER || t - agent->getLastLearned() < agent->getLearningInterval()) continue;
		
		ActionIterator* actions = &(actionsIt[i]);
		
		debug1("\n   Agent of type %s, #%d\n", agent->getName().c_str(), i);
		
		if (rng->uniform(0, 1) < settings.greedyEps)
		{
			a1[i] = actions->getRand(rng);
			debug1("Exploring!! Chose action %s\n", a0[i].print().c_str());
		}
		else
		{
			debug1("Chose best action %s\n", a1[i].print().c_str());
		}
		
		s0[i] = s1[i];
		a0[i] = a1[i];
		agent->act(a1[i]);
		agent->setLastLearned(t);
	}
}

void QLearning::agentsMove()
{
	debug("Agents move\n");
	int n = agents.size();
	
#pragma omp parallel for
	for (int i = 0; i<n; i++)
	{
		agents[i]->move(dt);
	}
}

void QLearning::execSavers(double t)
{
	for (list<Saver*>::iterator it = savers.begin(), end = savers.end(); it != end; ++it)
	{
		if ( ((int)(t/dt) % (*it)->getPeriod()) == 0) (*it)->exec();
	}	
}

void QLearning::evolve(double t)
{
	debug("\n****************************************************************\n");
	debug("Processing agents at time %f\n", t);

	// Observe current state of one agent
	// Take action a according to e-greedy policy derived from Q(s, a)
	// Get reward r, observe next state s'
	// Modify Q(s, a) += lRate[ r + gamma * Q(s', a') - Q(s, a) ]
	// If agent can't learn, just let him move
		
	if (profiler != NULL)
	{
		profiler->push_start("Compute new values");
		agentsChoose(t);
		profiler->pop_stop();

		profiler->push_start("Update Q(s,a)");
		agentsUpdate(t);
		profiler->pop_stop();

		profiler->push_start("Taking actions");
		agentsAct(t);
		profiler->pop_stop();

		profiler->push_start("Moving");
		agentsMove();
		profiler->pop_stop();
		
		profiler->push_start("Environment evolution");
		system.env->evolve(t);
		profiler->pop_stop();
	}
	else
	{
		agentsChoose(t);
		agentsUpdate(t);
		agentsAct(t);
		agentsMove();
	}
	
	execSavers(t);
}

void QLearning::try2restart(string prefix)
{
	info("Restarting from saved policy...\n");
	bool fl = true;
	for (map<string, MultiTable*>::iterator it=QMap.begin(); it!=QMap.end(); it++)
	{
		string fname = prefix + it->first + "_backup";
		if ( !(it->second->restart(fname)) ) fl = false;
	}
	if (fl) info("Restart successful, moving on...\n");
	else    info("Not all policies restarted, therefore assumed zero. Moving on...\n");
}
		
void QLearning::savePolicy(string prefix)
{
	info("\nSaving all policies...\n");
	for (map<string, MultiTable*>::iterator it=QMap.begin(); it!=QMap.end(); it++)
	{
		string fname = prefix + it->first + "_backup";
		it->second->save(fname);
	}
	info("Done\n");
}

void QLearning::registerSaver(Saver* saver, int period)
{
	saver->setEnvironment(system.env);
	saver->setPeriod(period);
	savers.push_back(saver);
}

		