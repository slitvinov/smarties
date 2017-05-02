/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "PolicyAlgorithm.h"

using namespace std;

class RACER : public PolicyAlgorithm
{
	const Real truncation;
	mutable vector<vector<Real>> stdGrad, avgGrad;
	mutable vector<Real> cntGrad;
	#ifdef __ACER_SAFE
	//Real stdev = 0.1;
	Real variance = 0.01;
	Real precision = 100;
	#endif

	void Train_BPTT(const int seq, const int thrID=0) const override;
	void Train(const int seq, const int samp, const int thrID=0) const override;
	void processStats(vector<trainData*> _stats, const Real avgTime) override;

	vector<Real> basicNetOut(const int agentId, State& s, Action& a,
		State& sOld, Action& aOld, const int info, Real r)
	{
		if (info == 2) { //no need for action, just pass terminal s & r
			data->passData(agentId, info, sOld, a, vector<Real>(), s, r);
			return vector<Real>(0);
		}

		Activation* currActivation = net->allocateActivation();

		vector<Real> output(nOutputs);
		vector<Real> input = s.copy_observed();
		//if required, chain together nAppended obs to compose state
		if (nAppended>0) {
			const int sApp = nAppended*sInfo.dimUsed;
			if(info==1)
				input.insert(input.end(),sApp, 0);
			else {
				assert(data->Tmp[agentId]->tuples.size()!=0);
				const Tuple * const last = data->Tmp[agentId]->tuples.back();
				input.insert(input.end(),last->s.begin(),last->s.begin()+sApp);
				assert(last->s.size()==input.size());
			}
		}

		if(info==1) {
			net->predict(data->standardize(input), output, currActivation
			#ifdef __EntropySGD //then we sample from target weights
			      , net->tgt_weights, net->tgt_biases
			#endif
	      );
		} else { //then if i'm using RNN i need to load recurrent connections (else no effect)
			Activation* prevActivation = net->allocateActivation();
			net->loadMemory(net->mem[agentId], prevActivation);
			net->predict(data->standardize(input), output, prevActivation, currActivation
			#ifdef __EntropySGD //then we sample from target weights
			      , net->tgt_weights, net->tgt_biases
			#endif
	      );
			_dispose_object(prevActivation);
		}

		//save network transition
		net->loadMemory(net->mem[agentId], currActivation);
		_dispose_object(currActivation);

		return output;
	}

	vector<Real> basicNetOut(Action& a, const vector<Real> mu, const vector<Real> var)
	{
		assert(mu.size()==nA);
		assert(var.size()==nA);
		vector<Real> beta(2*nA, 0);
		const Real eps = annealingFactor();

		if(bTrain && positive(eps)) {
			for(int i=0; i<nA; i++) {
				const Real varscale = aInfo.addedVariance(i);
				const Real policy_std = std::sqrt(var[i]); //output: 1/S^2
				Real anneal_std = eps*varscale*greedyEps + (1-eps)*policy_std;
				const Real annealed_mean = (1-eps*eps)*mu[i];
				//const Real annealed_mean = output[1+nL+i];
				std::normal_distribution<Real> dist_cur(annealed_mean, anneal_std);
				beta[i] = annealed_mean; //to save correct mu
				beta[nA+i] = 1./std::pow(anneal_std, 2); //to save correct mu
				a.vals[i] = dist_cur(*gen);
			}
		}
		else if (positive(greedyEps) || bTrain) { //still want to sample policy.
			for(int i=0; i<nA; i++) {
				std::normal_distribution<Real> dist_cur(mu[i], std::sqrt(var[i]));
				a.vals[i] = dist_cur(*gen);
				beta[i] = mu[i]; //to save correct mu
				beta[nA+i] = 1./var[i]; //to save correct mu
			}
		}
		else {//load computed policy into a
			for(int i=0; i<nA; i++) {
				a.vals[i] = mu[i];
				beta[i] = mu[i]; //to save correct mu
				beta[nA+i] = 1./var[i]; //to save correct mu
			}
		}

		finalizePolicy(a); //if bounded action space: scale
		return beta;
	}

public:
	RACER(MPI_Comm comm, Environment*const env, Settings & settings);
	void select(const int agentId, State& s,Action& a, State& sOld,
			Action& aOld, const int info, Real r) override;

	static int getnOutputs(const int NL, const int NA)
	{
		#if defined __ACER_RELAX
			// I output V(s), P(s), pol(s), prec(s) (and variate)
			#ifdef __ACER_VARIATE
				return 1+NL+NA+NA+1;
			#else
				return 1+NL+NA+NA;
			#endif
		#elif defined __ACER_SAFE
			// I output V(s), P(s), pol(s), mu(s) (and variate)
			#ifdef __ACER_VARIATE
				return 1+NL+NA+NA+1;
			#else
				return 1+NL+NA+NA;
			#endif
		#else //full formulation
			// I output V(s), P(s), pol(s), prec(s), mu(s) (and variate)
			#ifdef __ACER_VARIATE
				return 1+NL+NA+NA+NA+1;
			#else
				return 1+NL+NA+NA+NA;
			#endif
		#endif
	}

private:

	inline vector<Real> finalizeGradient(const Real Qerror, const Real Verror,
			const vector<Real>& gradCritic, const vector<Real>& gradPolicy,
			const vector<Real>& out, const Real err_Cov) const
	{
		assert(out.size() == nOutputs);
		assert(gradPolicy.size() == 2*nA); //no matter what
		vector<Real> grad(nOutputs);
		#ifdef __ACER_RELAX
			assert(gradCritic.size() == 1+nL);
		#else
			assert(gradCritic.size() == 1+nL+nA);
		#endif

		grad[0] = Qerror+Verror;
		for (int j=1; j<nL+1; j++)
			grad[j] = Qerror*gradCritic[j];

		for (int j=0; j<nA; j++)
			grad[1+nL+j] = gradPolicy[j];

		#ifndef __ACER_SAFE
			const vector<Real> gradVar = finalizeVarianceGrad(gradPolicy, out);
			for (int j=0; j<nA; j++)
				grad[1+nL+nA+j] = gradVar[j];
		#endif

		#ifndef __ACER_RELAX
			#ifndef __ACER_SAFE
				for (int j=nL+1; j<nA+nL+1; j++)
					grad[j+nA*2] = Qerror*gradCritic[j];
			#else
				for (int j=nL+1; j<nA+nL+1; j++)
					grad[j+nA] = Qerror*gradCritic[j];
			#endif
		#else
			//no gradient for mean of critic, ofc
		#endif

		#ifdef __ACER_VARIATE
			grad[nOutputs-1] = err_Cov;
		#endif

		return grad;
	}

	void buildNetwork(const vector<int> nouts, Settings & settings)
	{
		string lType = bRecurrent ? "LSTM" : "Normal";
		vector<int> lsize;
		assert(nouts.size()>0);

		lsize.push_back(settings.nnLayer1);
		if (settings.nnLayer2>1) {
			lsize.push_back(settings.nnLayer2);
			if (settings.nnLayer3>1) {
				lsize.push_back(settings.nnLayer3);
				if (settings.nnLayer4>1) {
					lsize.push_back(settings.nnLayer4);
					if (settings.nnLayer5>1) {
						lsize.push_back(settings.nnLayer5);
					}
				}
			}
		}

		net = new Network(settings);
		//check if environment wants a particular network structure
		if (not env->predefinedNetwork(net))
		{
			//if that was true, environment created the layers it wanted
			// else we read the settings:
			net->addInput(nInputs);
			//const int nsplit = lsize.size()>3 ? 2 : 1;
			const int nsplit = lsize.size();
			for (int i=0; i<lsize.size()-nsplit; i++)
				net->addLayer(lsize[i], lType);

			const int firstSplit = lsize.size()-nsplit;
			const vector<int> lastJointLayer(1,net->getLastLayerID());

			for (int i=0; i<nouts.size(); i++)
			{
				net->addLayer(lsize[firstSplit], lType, lastJointLayer);

				for (int j=firstSplit+1; j<lsize.size(); j++)
					net->addLayer(lsize[j], lType);

				net->addOutput(nouts[i], "Normal");
			}
		}
		net->build();

		#ifndef __EntropySGD
			opt = new AdamOptimizer(net, profiler, settings);
		#else
			opt = new EntropySGD(net, profiler, settings);
		#endif
		data->bRecurrent = bRecurrent = true;
	}

	vector<vector<Real>> prepareBins(const vector<Real> lower, const vector<Real>& upper,
			const vector<int>& nbins)
	{
		if(nAppended || nA!=1)
			die("TODO missing features\n");
		assert(lower.size() == upper.size());
		assert(nbins.size() == upper.size());
		assert(nbins.size() == nInputs);
		vector<vector<Real>> bins(nbins.size());
		int nDumpPoints = 1;
		for (int i=0; i<nbins.size(); i++) {
			nDumpPoints *= nbins[i];
			bins[i] = vector<Real>(nbins[i]);
			for (int j=0; j<nbins[i]; j++) {
				const Real l = j/(Real)(nbins[i]-1);
				bins[i][j] = lower[i] + (upper[i]-lower[i])*l;
			}
		}
		return bins;
	}

	void dumpPolicy(const vector<Real> lower, const vector<Real>& upper,
	 		const vector<int>& nbins) override;
	 /*
	 void dumpNetworkInfo(const int agentId)
	 {
	 	net->dump(agentId);
	 	vector<Real> output(nOutputs);

	 	const int ndata = data->Tmp[agentId]->tuples.size(); //last one already placed
	 	if (ndata == 0) return;

	 	vector<Activation*> timeSeries_base = net->allocateUnrolledActivations(ndata);
	 	net->clearErrors(timeSeries_base);

	 	for (int k=0; k<ndata; k++) {
	 		const Tuple * const _t = data->Tmp[agentId]->tuples[k];
	 		vector<Real> scaledSnew = data->standardize(_t->s);
	 		net->predict(scaledSnew, output, timeSeries_base, k);
	 	}

	 	//sensitivity of value for this action in this state wrt all previous inputs
	 	for (int ii=0; ii<ndata; ii++)
	 	for (int i=0; i<nInputs; i++) {
	 		vector<Activation*> series =net->allocateUnrolledActivations(ndata);

	 		for (int k=0; k<ndata; k++) {
	 			const Tuple* const _t = data->Tmp[agentId]->tuples[k];
	 			vector<Real> scaledSnew = data->standardize(_t->s);
	 			if (k==ii) scaledSnew[i] = 0;
	 			net->predict(scaledSnew, output, series, k);
	 		}
	 		vector<Real> oDiff = net->getOutputs(series.back());
	 		vector<Real> oBase = net->getOutputs(timeSeries_base.back());
	 		const Tuple* const _t = data->Tmp[agentId]->tuples[ii];
	 		vector<Real> sSnew = data->standardize(_t->s);

	 		//assert(nA==1); //TODO ask Sid for muldi-dim actions?
	 		double dAct = 0;
	 		for (int j=0; j<nA; j++)
	 			dAct+=pow(oDiff[1+nL+j]-oBase[1+nL+j],2);

	 		timeSeries_base[ii]->errvals[i]=sqrt(dAct)/sSnew[i];
	 		net->deallocateUnrolledActivations(&series);
	 	}

	 	string fname="gradInputs_"+to_string(agentId)+"_"+to_string(ndata)+".dat";
	 	ofstream out(fname.c_str());
	 	if (!out.good()) die("Unable to open save into file %s\n", fname.c_str());
	 	for (int k=0; k<ndata; k++) {
	 		for (int j=0; j<nInputs; j++)
	 		out << timeSeries_base[k]->errvals[j] << " ";
	 		out << "\n";
	 	}
	 	out.close();

	 	net->deallocateUnrolledActivations(&timeSeries_base);
	 }
	 */
};
