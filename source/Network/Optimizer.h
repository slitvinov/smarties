/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include <fstream>
#include "Network.h"
#include "../Profiler.h"

class Optimizer
{ //basic momentum update
protected:
	const Uint nWeights, nBiases, bTrain;
	Network * const net;
	Profiler * const profiler;
	nnReal* const _1stMomW;
	nnReal* const _1stMomB;

	void update(nnReal* const dest, nnReal* const grad, nnReal* const _1stMom,
			const Uint N, const Uint batchsize) const;

public:
	const Real eta, lambda, alpha = 0.5;
	long unsigned nepoch = 0;

	Optimizer(Network* const _net, Profiler* const _prof, Settings & settings);

	virtual ~Optimizer()
	{
		_myfree(_1stMomW);
		_myfree(_1stMomB);
	}
	virtual void update(Grads* const G, const Uint batchsize);

	virtual void stackGrads(Grads* const G, const Grads* const g) const;
	virtual void stackGrads(Grads* const G, const vector<Grads*> g) const;

	virtual void save(const string fname);
	virtual bool restart(const string fname);
	virtual void moveFrozenWeights(const Real _alpha);

	void save_recurrent_connections(const string fname)
	{
		const Uint nNeurons(net->getnNeurons());
		const Uint nAgents(net->getnAgents()), nStates(net->getnStates());
		string nameBackup = fname + "_mems_tmp";
		ofstream out(nameBackup.c_str());
		if (!out.good())
			_die("Unable to open save into file %s\n", nameBackup.c_str());

		for(Uint agentID=0; agentID<nAgents; agentID++) {
			for (Uint j=0; j<nNeurons; j++)
				out << net->mem[agentID]->outvals[j] << "\n";
			for (Uint j=0; j<nStates;  j++)
				out << net->mem[agentID]->ostates[j] << "\n";
		}
		out.flush();
		out.close();
		string command = "cp " + nameBackup + " " + fname + "_mems";
		system(command.c_str());
	}

	bool restart_recurrent_connections(const string fname)
	{
		const Uint nNeurons(net->getnNeurons());
		const Uint nAgents(net->getnAgents()), nStates(net->getnStates());

		string nameBackup = fname + "_mems";
		ifstream in(nameBackup.c_str());
		debugN("Reading from %s\n", nameBackup.c_str());
		if (!in.good()) {
			error("Couldnt open file %s \n", nameBackup.c_str());
			return false;
		}

		nnReal tmp;
		for(Uint agentID=0; agentID<nAgents; agentID++) {
			for (Uint j=0; j<nNeurons; j++) {
				in >> tmp;
				if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
				net->mem[agentID]->outvals[j] = tmp;
			}
			for (Uint j=0; j<nStates; j++) {
				in >> tmp;
				if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
				net->mem[agentID]->ostates[j] = tmp;
			}
		}
		in.close();
		return true;
	}
};

class AdamOptimizer: public Optimizer
{ //Adam optimizer
protected:
	const Real beta_1, beta_2, epsilon;
	Real beta_t_1, beta_t_2;
	nnReal* const _2ndMomW;
	nnReal* const _2ndMomB;

	void update(nnReal* const dest, nnReal* const grad, nnReal* const _1stMom,
		nnReal* const _2ndMom, const Uint N, const Uint batchsize, const Real _eta);

public:
	AdamOptimizer(Network* const _net,Profiler* const _prof,Settings& settings,
			const Real B1 = 0.8, const Real B2 = 0.999);

	~AdamOptimizer()
	{
		_myfree(_2ndMomW);
		_myfree(_2ndMomB);
	}
	void update(Grads* const G, const Uint batchsize) override;

	void save(const string fname) override;
	bool restart(const string fname) override;
};

class EntropySGD: public AdamOptimizer
{
protected:
	const Real alpha_eSGD, gamma_eSGD, eta_eSGD, eps_eSGD;
	const Uint L_eSGD;
	nnReal* const _muW_eSGD;
	nnReal* const _muB_eSGD;

	void update(nnReal*const dest, const nnReal*const target, nnReal*const grad,
		nnReal*const _1stMom, nnReal*const _2ndMom, nnReal*const _mu, const Uint N,
		const Uint batchsize, const Real _eta);
public:

	EntropySGD(Network* const _net,Profiler* const _prof,Settings& settings);

	~EntropySGD()
	{
		_myfree(_muW_eSGD);
		_myfree(_muB_eSGD);
	}
	void update(Grads* const G, const Uint batchsize) override;
	bool restart(const string fname) override;
	void moveFrozenWeights(const Real _alpha) override;
};
