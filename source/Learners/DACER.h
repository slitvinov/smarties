/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "DiscreteAlgorithm.h"

class DACER : public DiscreteAlgorithm
{
	const Real truncation;
	mutable vector<vector<Real>> stdGrad, avgGrad;
	mutable vector<Real> cntGrad;

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

	vector<Real> basicNetOut(Action& a, const vector<Real> pol)
	{
		assert(pol.size()==nA);
		vector<Real> beta = pol;
		const Real eps = annealingFactor();
		Uint iAct = 0;
		const Real addedVar = greedyEps*eps/nA, trunc = (1-greedyEps*eps);

		if(bTrain && positive(eps))
			for(Uint i=0; i<nA; i++) beta[i] = trunc*beta[i] + addedVar;

		std::discrete_distribution<Uint> dist(beta.begin(),beta.end());

		if (positive(greedyEps) || bTrain)
			iAct = dist(*gen);
		else
			iAct = maxInd(pol);
		assert(iAct<nA);
		a.set(iAct);
		return beta;
	}

public:
	DACER(MPI_Comm comm, Environment*const env, Settings & settings);
	void select(const int agentId, State& s,Action& a, State& sOld,
			Action& aOld, const int info, Real r) override;

	static Uint getnOutputs(const Uint NA)
	{
		return 1+NA+NA;
	}

private:

	inline vector<Real> finalizeGradient(const Real Verror,
			const vector<Real>& gradCritic, const vector<Real>& gradPolicy) const
	{
		assert(gradPolicy.size() == nA);
		assert(gradCritic.size() == 1+nA);
		vector<Real> grad(nOutputs);

		grad[0] = gradCritic[0]+Verror;
		for (Uint j=0; j<nA; j++)
			grad[j+1] = gradPolicy[j];
		for (Uint j=1; j<nA+1; j++)
			grad[j+nA] = gradCritic[j];

		//gradient clipping
		//for (unsigned int i=0; i<grad.size(); i++)
		//	grad[i] = std::max(-10.,std::min(10.,grad[i]));

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
		int nDumpPoints = 1;
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
	 	if (!out.good()) _die("Unable to open save into file %s\n", fname.c_str());
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
