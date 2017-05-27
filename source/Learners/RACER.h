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

class RACER : public PolicyAlgorithm
{
	const Real truncation;
	mutable vector<vector<Real>> stdGrad, avgGrad;
	mutable vector<Real> cntGrad;
	#ifdef ACER_SAFE
	//Real stdev = 0.1;
	Real variance = 0.01;
	Real precision = 100;
	#endif

	void Train_BPTT(const Uint seq, const Uint thrID=0) const override;
	void Train(const Uint seq, const Uint samp, const Uint thrID=0) const override;
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
			const Uint sApp = nAppended*sInfo.dimUsed;
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
			prevActivation->loadMemory(net->mem[agentId]);
			net->predict(data->standardize(input), output, prevActivation, currActivation
			#ifdef __EntropySGD //then we sample from target weights
			      , net->tgt_weights, net->tgt_biases
			#endif
	      );
			_dispose_object(prevActivation);
		}

		//save network transition
		currActivation->storeMemory(net->mem[agentId]);
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
			for(Uint i=0; i<nA; i++) {
				const Real varscale = aInfo.addedVariance(i);
				const Real policy_std = std::sqrt(var[i]); //output: 1/S^2
				const Real anneal_std = eps*varscale*greedyEps + (1-eps)*policy_std;
				const Real annealed_mean = (1-eps*eps)*mu[i];
				//const Real annealed_mean = output[1+nL+i];
				std::normal_distribution<Real> dist_cur(annealed_mean, anneal_std);
				beta[i] = annealed_mean; //to save correct mu
				beta[nA+i] = 1./std::pow(anneal_std, 2); //to save correct mu
				a.vals[i] = dist_cur(*gen);
			}
		}
		else if (positive(greedyEps) || bTrain) { //still want to sample policy.
			for(Uint i=0; i<nA; i++) {
				std::normal_distribution<Real> dist_cur(mu[i], std::sqrt(var[i]));
				a.vals[i] = dist_cur(*gen);
				beta[i] = mu[i]; //to save correct mu
				beta[nA+i] = 1./var[i]; //to save correct mu
			}
		}
		else {//load computed policy into a
			for(Uint i=0; i<nA; i++) {
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

	static Uint getnOutputs(const Uint NA)
	{
		#if defined ACER_RELAX
			// I output V(s), P(s), pol(s), prec(s) (and variate)
				return 1+(NA*NA+NA)/2+NA+NA;
		#elif defined ACER_SAFE
			// I output V(s), P(s), pol(s), mu(s) (and variate)
				return 1+(NA*NA+NA)/2+NA+NA;
		#else //full formulation
			// I output V(s), P(s), pol(s), prec(s), mu(s) (and variate)
				return 1+(NA*NA+NA)/2+NA+NA+NA;
		#endif
	}

private:

	inline vector<Real> finalizeGradient(const Real Verror,
			const vector<Real>& gradCritic, const vector<Real>& gradPolicy,
			const vector<Real>& out, const Uint thrID, const Real gain1, const Real eta) const
	{
      const Real anneal = std::pow(annealingFactor(), 2);
		assert(out.size() == nOutputs);
		assert(gradPolicy.size() == 2*nA); //no matter what
		vector<Real> grad(nOutputs);
		#ifdef ACER_RELAX
			assert(gradCritic.size() == 1+nL);
		#else
			assert(gradCritic.size() == 1+nL+nA);
		#endif

		grad[0] = gradCritic[0]+Verror;
		for (Uint j=1; j<nL+1; j++)
			grad[j] = gradCritic[j];

		for (Uint j=0; j<nA; j++)
			//grad[1+nL+j] = gradPolicy[j];
			grad[1+nL+j] = gradPolicy[j] -anneal*out[1+nL+j];

		#ifndef ACER_SAFE
			const vector<Real> gradVar = finalizeVarianceGrad(gradPolicy, out);
			for (Uint j=0; j<nA; j++) grad[1+nL+nA+j] = gradVar[j];
		#endif

		#ifndef ACER_RELAX
			for (Uint j=nL+1; j<nA+nL+1; j++) {
			   #ifndef ACER_SAFE
         		//grad[j+nA*2] = gradCritic[j];
		   	grad[j+nA*2] = gradCritic[j] -anneal*out[j+nA*2];
			   #else
         		//grad[j+nA] = gradCritic[j];
		   	grad[j+nA] = gradCritic[j] -anneal*out[j+nA];
			   #endif
         }
		#else
			//no gradient for mean of critic, ofc
		#endif

		//gradient clipping
		//1) update stats about the gradient
		vector<Real> _dump = grad;
		_dump.push_back(gain1);
		_dump.push_back(eta);
		statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], _dump);
		//2) clip the gradient wrt previous epoch to ACER_GRAD_CUT sigma
		for (Uint i=0; i<grad.size(); i++)
		{
			if(grad[i] >  ACER_GRAD_CUT*stdGrad[0][i] && stdGrad[0][i]>2.2e-16)
				grad[i] =  ACER_GRAD_CUT*stdGrad[0][i];
			else
			if(grad[i] < -ACER_GRAD_CUT*stdGrad[0][i] && stdGrad[0][i]>2.2e-16)
				grad[i] = -ACER_GRAD_CUT*stdGrad[0][i];
		}

		return grad;
	}

	inline Real stateValue(const Real v, const Real w) const
	{
		return std::fabs(v)<std::fabs(w) ? v : w;
	}

	vector<vector<Real>> prepareBins(const vector<Real> lower, const vector<Real>& upper,
			const vector<Uint>& nbins)
	{
		if(nAppended || nA!=1)
			die("TODO missing features\n");
		assert(lower.size() == upper.size());
		assert(nbins.size() == upper.size());
		assert(nbins.size() == nInputs);
		vector<vector<Real>> bins(nbins.size());
		Uint nDumpPoints = 1;
		for (Uint i=0; i<nbins.size(); i++) {
			nDumpPoints *= nbins[i];
			bins[i] = vector<Real>(nbins[i]);
			for (Uint j=0; j<nbins[i]; j++) {
				const Real l = j/(Real)(nbins[i]-1);
				bins[i][j] = lower[i] + (upper[i]-lower[i])*l;
			}
		}
		return bins;
	}

	void dumpPolicy(const vector<Real> lower, const vector<Real>& upper,
	 		const vector<Uint>& nbins) override;
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
