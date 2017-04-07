/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "ACER.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <cmath>



ACER::ACER(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner(comm,_env,settings), nA(_env->aI.dim),
nL((_env->aI.dim*_env->aI.dim+_env->aI.dim)/2), generators(settings.generators)
{
	printf("Running (R)ACER! Fancy banner here\n");
	string lType = bRecurrent ? "LSTM" : "Normal";
	vector<int> lsize;
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
	{ //if that was true, environment created the layers it wanted, else we read the settings:
		net->addInput(nInputs);
		for (int i=0; i<lsize.size()-1; i++) net->addLayer(lsize[i], lType);
		const int splitLayer = lsize.size()-1;
		const vector<int> lastJointLayer(1,net->getLastLayerID());
		net->addLayer(lsize[splitLayer], lType, lastJointLayer);
		net->addOutput(1, "Normal");
		net->addLayer(lsize[splitLayer], lType, lastJointLayer);
		net->addOutput(nL, "Normal");
		net->addLayer(lsize[splitLayer], lType, lastJointLayer);
		net->addOutput(nA, "Normal");
		net->addLayer(lsize[splitLayer], lType, lastJointLayer);
		net->addOutput(nA, "Normal");
	}
	net->build();
	assert(1+nL+2*nA == net->getnOutputs() && nInputs == net->getnInputs());

	opt = new AdamOptimizer(net, profiler, settings);
	data->bRecurrent = bRecurrent = true;

	#if 1//ndef NDEBUG
   vector<Real> out_0(nOutputs, 0.1);
   for(int i = 0; i<nOutputs; i++) {
      uniform_real_distribution<Real> dis(-5,5);
      out_0[i] = dis(*gen);
   }
   vector<Real> act(nA,0.25);
   for(int i = 0; i<nA; i++) {
      uniform_real_distribution<Real> dis(-10,10);
      act[i] = dis(*gen);
   }
	 prepareVariance(out_0); 

	 const vector<Real> polGrad = policyGradient(out_0, act, 1.0);
	 for(int i = 0; i<2*nA; i++) {
      vector<Real> grad_1(nOutputs), grad_2(nOutputs);
      vector<Real> out_1 = out_0;
      vector<Real> out_2 = out_0;
      out_1[1+nL+i] -= 0.0001;
      out_2[1+nL+i] += 0.0001;
 			const Real p1 = evaluateLogProbability(act, out_1);
 			const Real p2 = evaluateLogProbability(act, out_2);

      const Real gradi = polGrad[i];
      const Real diffi = (p2-p1)/0.0002;
      printf("LogPol Gradient %d: finite differences %g analytic %g \n", i, diffi, gradi);
   }

	 vector<Real> grad_0 = computeGradient(1., 0., out_0, out_0, act, polGrad);
	 for(int i = 0; i<1+nL; i++) {
      vector<Real> out_1 = out_0;
      vector<Real> out_2 = out_0;
      out_1[i] -= 0.0001;
      out_2[i] += 0.0001;
      const Real Q_1 = computeQ(act, out_0, out_1);
      const Real Q_2 = computeQ(act, out_0, out_2);
      const Real gradi = grad_0[i];
      const Real diffi = (Q_2-Q_1)/0.0002;
      printf("Value Gradient %d: finite differences %g analytic %g \n", i, diffi, gradi);
   }

  #endif

}

static void printselection(const int iA,const int nA,const int i,vector<Real> s)
{
	printf("%d/%d s=%d : ", iA, nA, i);
	for(int k=0; k<s.size(); k++) printf("%g ", s[k]);
	printf("\n"); fflush(0);
}

void ACER::select(const int agentId, State& s, Action& a, State& sOld,
								 Action& aOld, const int info, Real r)
{
		if (info == 2) { //no need for action, just pass terminal s & r
			data->passData(agentId, info, sOld, a, vector<Real>(), s, r);
			return;
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
			net->predict(data->standardize(input), output, currActivation);
		} else { //then if i'm using RNN i need to load recurrent connections (else no effect)
			Activation* prevActivation = net->allocateActivation();
			net->loadMemory(net->mem[agentId], prevActivation);
			net->predict(data->standardize(input), output, prevActivation, currActivation);
      _dispose_object(prevActivation);
		}

    //save network transition
    net->loadMemory(net->mem[agentId], currActivation);
    _dispose_object(currActivation);
		//variance is pos def: transform linear output layer with softplus
		prepareVariance(output);

		const Real annealedEps = bTrain ? std::max(annealingFactor()+greedyEps,1.) : 0;

		if(bTrain) {
			for(int i=0; i<nA; i++) {
				const Real policy_var = 1./std::sqrt(output[1+nL+nA+i]); //output: 1/S^2
				const Real anneal_var = annealedEps*aInfo.addedVariance(i) + policy_var;
				std::normal_distribution<Real> dist_cur(output[1+nL+i], anneal_var);
				output[1+nL+nA+i] = 1./std::pow(anneal_var, 2); //to save correct mu
				a.vals[i] = dist_cur(*gen);
			}
		}
		else if (greedyEps) { //not training but still want to sample policy.
			for(int i=0; i<nA; i++) {
				const Real policy_var = 1./std::sqrt(output[1+nL+nA+i]); //output: 1/S^2
				std::normal_distribution<Real> dist_cur(output[1+nL+i], policy_var);
				a.vals[i] = dist_cur(*gen);
			}
		}
		else {//load computed policy into a
			const vector<Real> pi(&output[1+nL], &output[1+nL]+nA);
			a.set(pi);
		}

		const vector<Real> mu(&output[1+nL], &output[1+nL]+2*nA);
		finalizePolicy(a); //if bounded action space: scale
		data->passData(agentId, info, sOld, a, mu, s, r);

		/*
			#ifdef _dumpNet_
		   if (!bTrain)
				dumpNetworkInfo(agentId);
			#endif
		*/
}

/*
void ACER::dumpNetworkInfo(const int agentId)
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

void ACER::Train(const int seq, const int samp, const int thrID) const
{
    die("ACER only works by sampling entire trajectories.\n");
}

void ACER::Train_BPTT(const int seq, const int thrID) const
{
		//this should go to gamma rather quick:
		const Real rGamma=std::min(1.,Real(opt->nepoch)/epsAnneal)*gamma;
		//const Real rGamma = gamma;
    assert(net->allocatedFrozenWeights && bTrain);
    const int ndata = data->Set[seq]->tuples.size();
		vector<vector<Real>> out_cur(ndata-1, vector<Real>(1+nL+nA*2,0));
		vector<vector<Real>> out_hat(ndata-1, vector<Real>(1+nL+nA*2,0));
		vector<Real> rho_cur(ndata-1), rho_pol(ndata-1);
		//vector<Real> rho_hat(ndata-1), c_hat(ndata-1);
		vector<Real> c_cur(ndata-1);
		vector<vector<Real>> act(ndata-1, vector<Real>(nA,0)), pol(ndata-1, vector<Real>(nA,0));
    vector<Activation*> series_cur = net->allocateUnrolledActivations(ndata-1);
    vector<Activation*> series_hat = net->allocateUnrolledActivations(ndata);

		for (int k=0; k<ndata-1; k++)
		{
      const Tuple * const _t = data->Set[seq]->tuples[k]; //this tuple contains s, a, mu
      vector<Real> scaledSold = data->standardize(_t->s);
      net->predict(scaledSold, out_cur[k], series_cur, k);
      net->predict(scaledSold, out_hat[k], series_hat, k, net->tgt_weights, net->tgt_biases);
			prepareVariance(out_cur[k]); //pass through softplus to make it pos def
			prepareVariance(out_hat[k]); //grad correction in computeGradient
			//also works in unbounded action spce
			act[k] = aInfo.getInvScaled(_t->a);
	    for(int i=0; i<nA; i++) { //sample current policy
				const Real pol_var = 1./std::sqrt(out_cur[k][1+nL+nA+i]);
				std::normal_distribution<Real> dist_cur(out_cur[k][1+nL+i], pol_var);
				pol[k][i] = dist_cur(generators[thrID]);
			}
			const Real actProbOnPolicy = evaluateLogProbability(act[k], out_cur[k]);
			const Real polProbOnPolicy = evaluateLogProbability(pol[k], out_cur[k]);
			//const Real actProbOnTarget = evaluateLogProbability(act[k], out_hat[k]);
			const Real actProbBehavior = evaluateLogBehavioralPolicy(act[k], _t->mu);
			const Real polProbBehavior = evaluateLogBehavioralPolicy(pol[k], _t->mu);

			rho_cur[k] = std::exp(std::min(10.,std::max(-10.,actProbOnPolicy-actProbBehavior)));
			rho_pol[k] = std::exp(std::min(10.,std::max(-10.,polProbOnPolicy-polProbBehavior)));
			//rho_hat[k] = std::exp(std::min(10.,std::max(-10.,actProbOnTarget-actProbBehavior)));
			//rho_hat[k] = rho_cur[k];
			c_cur[k] = std::min((Real)1.,std::pow(rho_cur[k],1./nA));
		}

		Real Q_RET = 0, Q_OPC = 0;
		if(not data->Set[seq]->ended)
		{
			const Tuple * const _t = data->Set[seq]->tuples[ndata-1];
      vector<Real> S_T = data->standardize(_t->s); //last state
			vector<Real> out_T(1+nL+nA*2, 0);
      //net->predict(S_T, out_T, series_hat, ndata-1, net->tgt_weights, net->tgt_biases);
      net->predict(S_T, out_T, series_cur.back(), series_hat.back());
			Q_RET = out_T[0]; //V(s_T) computed with tgt weights
			Q_OPC = out_T[0]; //V(s_T) computed with tgt weights
		}
		#ifndef NDEBUG
		else {
			const Tuple * const _t = data->Set[seq]->tuples[ndata-1];
			assert(_t->mu.size() == 0);
		}
		#endif

		for (int k=ndata-2; k>=0; k--)
		{
      //const Tuple * const _t = data->Set[seq]->tuples[k]; //this tuple contains sOld
			const Tuple * const t_ = data->Set[seq]->tuples[k+1]; //this contains a, r, sNew
			Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
			Q_OPC = t_->r + rGamma*Q_OPC;
			//compute Q using target net for pi and C, for consistency of derivatives
			//Q(s,a)                     v a		 v policy    v quadratic Q parameters
			const Real Q_cur = computeQ(act[k], out_cur[k], out_cur[k]);
			//const Real Q_hat = computeQ(act[k], out_hat[k], out_hat[k]);
			const Real Q_pol = computeQ(pol[k], out_cur[k], out_cur[k]);
			//compute quantities needed for trunc import sampl with bias correction
			const Real importance = std::min(rho_cur[k], truncation);
			const Real correction = std::max(0., 1.-truncation/rho_pol[k]);
			const Real gain1 = (Q_OPC - out_cur[k][0]) * importance;
			const Real gain2 = (Q_pol - out_cur[k][0]) * correction;
			//derivative wrt to statistics
			const vector<Real> gradAcer_1 = policyGradient(out_cur[k], act[k], gain1);
			const vector<Real> gradAcer_2 = policyGradient(out_cur[k], pol[k], gain2);
			//trust region updating
      const vector<Real> gradDivKL= gradDKL(out_cur[k], out_hat[k]);
			const vector<Real> gradAcer = gradAcerTrpo(gradAcer_1,gradAcer_2,gradDivKL);

			const Real Qerror = (Q_RET - Q_cur);
			const Real Verror = (Q_RET - Q_cur) * std::min(1.,rho_cur[k]); //unclear usefulness
			//prepare rolled Q with off policy corrections for next step:
			Q_RET = c_cur[k] *(Q_RET - Q_cur) + out_cur[k][0];
			Q_OPC = 					(Q_OPC - Q_cur) + out_cur[k][0];

			const vector<Real> grad = computeGradient(Qerror, Verror, out_cur[k], out_hat[k],
				act[k], gradAcer);
			net->setOutputDeltas(grad, series_cur[k]);
			//bookkeeping:
			vector<Real> fake{Q_cur, 100};
	    dumpStats(Vstats[thrID], Q_cur, Qerror, fake);
			if(thrID == 1) net->updateRunning(series_cur[k]);
	    data->Set[seq]->tuples[k]->SquaredError = Qerror*Qerror;
		}

		if (thrID==0) net->backProp(series_cur, net->grad);
    else net->backProp(series_cur, net->Vgrad[thrID]);
    net->deallocateUnrolledActivations(&series_cur);
    net->deallocateUnrolledActivations(&series_hat);
}
