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

using namespace ErrorHandling;

class Optimizer
{ //basic momentum update
 protected:
	const int nWeights, nBiases, bTrain;
	Network * const net;
	Profiler * const profiler;
	Real* const _1stMomW;
	Real* const _1stMomB;

	inline Real* init(const int N, const Real ini=0) const
	{
		Real* ret;
		_allocateClean(ret, N);
		for (int j=0; j<N; j++) ret[j] = ini;
		return ret;
	}

	void update(Real* const dest, Real* const grad, Real* const _1stMom,
			const int N, const int batchsize) const;

 public:
	const Real eta, lambda, alpha;
	long unsigned nepoch;

	Optimizer(Network* const _net, Profiler* const _prof, Settings & settings);

	virtual ~Optimizer()
	{
		_myfree(_1stMomW);
		_myfree(_1stMomB);
	}
	virtual void update(Grads* const G, const int batchsize);

	virtual void stackGrads(Grads* const G, const Grads* const g) const;
	virtual void stackGrads(Grads* const G, const vector<Grads*> g) const;

	virtual void save(const string fname);
	virtual bool restart(const string fname);
	virtual void moveFrozenWeights(const Real _alpha);

	inline void applyL1(Real* const dest, const int N, const Real lambda_)
	{
		#pragma omp parallel for
		for (int i=0; i<N; i++)
		dest[i] += (dest[i]<0 ? lambda_ : -lambda_);
	}

	inline void applyL2(Real* const dest, const int N, const Real lambda_)
	{
		#pragma omp parallel for
		for (int i=0; i<N; i++)
		dest[i] -= dest[i]*lambda_;
	}

	void save_recurrent_connections(const string fname)
	{
		const int nNeurons(net->getnNeurons()), nLayers(net->getnLayers());
		const int nAgents(net->getnAgents()), nStates(net->getnStates());
		string nameBackup = fname + "_mems_tmp";
		ofstream out(nameBackup.c_str());
		if (!out.good())
			die("Unable to open save into file %s\n", nameBackup.c_str());

		for(int agentID=0; agentID<nAgents; agentID++) {
			for (int j=0; j<nNeurons; j++)
			out << net->mem[agentID]->outvals[j] << "\n";
			for (int j=0; j<nStates;  j++)
			out << net->mem[agentID]->ostates[j] << "\n";
		}
		out.flush();
		out.close();
		string command = "cp " + nameBackup + " " + fname + "_mems";
		system(command.c_str());
	}

	bool restart_recurrent_connections(const string fname)
	{
		const int nNeurons(net->getnNeurons()), nLayers(net->getnLayers());
		const int nAgents(net->getnAgents()), nStates(net->getnStates());

		string nameBackup = fname + "_mems";
		ifstream in(nameBackup.c_str());
		debug1("Reading from %s\n", nameBackup.c_str());
		if (!in.good()) {
			error("Couldnt open file %s \n", nameBackup.c_str());
			return false;
		}

		Real tmp;
		for(int agentID=0; agentID<nAgents; agentID++) {
			for (int j=0; j<nNeurons; j++) {
				in >> tmp;
				if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
				net->mem[agentID]->outvals[j] = tmp;
			}
			for (int j=0; j<nStates; j++) {
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
	Real* const _2ndMomW;
	Real* const _2ndMomB;

	void update(Real* const dest, Real* const grad, Real* const _1stMom,
			Real* const _2ndMom, const int N, const int batchsize, const Real _eta);

 public:
	AdamOptimizer(Network* const _net,Profiler* const _prof,Settings& settings,
			const Real B1 = 0.9, const Real B2 = 0.999);

	~AdamOptimizer()
	{
		_myfree(_2ndMomW);
		_myfree(_2ndMomB);
	}
	void update(Grads* const G, const int batchsize) override;

	void save(const string fname) override;
	bool restart(const string fname) override;
};

class EntropySGD: public AdamOptimizer
{
 protected:
	const Real alpha_eSGD, gamma_eSGD, eta_eSGD, eps_eSGD;
	const int L_eSGD;
	Real* const _muW_eSGD;
	Real* const _muB_eSGD;

	void update(Real* const dest, const Real* const target, Real* const grad,
			Real* const _1stMom, Real* const _2ndMom, Real* const _mu, const int N,
			const int batchsize, const Real _eta);
 public:

	EntropySGD(Network* const _net,Profiler* const _prof,Settings& settings);

	~EntropySGD()
	{
		_myfree(_muW_eSGD);
		_myfree(_muB_eSGD);
	}
	void update(Grads* const G, const int batchsize) override;
	bool restart(const string fname) override;
	void moveFrozenWeights(const Real _alpha) override;
};
