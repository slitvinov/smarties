/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "NAF.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <cmath>



NAF::NAF(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner(comm,_env,settings), nA(_env->aI.dim),
nL((_env->aI.dim*_env->aI.dim+_env->aI.dim)/2)
{
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
	}
	net->build();
	assert(1+nL+nA == net->getnOutputs() && nInputs == net->getnInputs());

	opt = new AdamOptimizer(net, profiler, settings);

	#if 1//ndef NDEBUG
   vector<Real> out_0(nOutputs, 0.1), grad_0(nOutputs);
   for(int i = 0; i<nOutputs; i++) {
      uniform_real_distribution<Real> dis(-10,10);
      out_0[i] = dis(*gen);
   }
   vector<Real> act(nA,0.25);
   for(int i = 0; i<nA; i++) {
      uniform_real_distribution<Real> dis(-0.5,0.5);
      act[i] = dis(*gen);
   }
   uniform_real_distribution<Real> dis(-10,10);
   const Real V = dis(*gen);

   vector<Real> Q_0 = computeQandGrad(grad_0, act, out_0, V);
   for(int i = 0; i<1+nL+nA; i++) {
      vector<Real> grad_1(nOutputs), grad_2(nOutputs);
      vector<Real> out_1 = out_0;
      vector<Real> out_2 = out_0;
      out_1[i] -= 0.0001;
      out_2[i] += 0.0001;
      vector<Real> Q_1 = computeQandGrad(grad_1, act, out_1, V);
      vector<Real> Q_2 = computeQandGrad(grad_2, act, out_2, V);
      const Real gradi = grad_0[i]/(V - Q_0[0]);
      const Real diffi = (Q_2[0]-Q_1[0])/0.0002;
      printf("Gradient %d: finite differences %g analytic %g \n", i, diffi, gradi);
   }
  #endif
}

/*
static void printselection(const int iA,const int nA,const int i,vector<Real> s)
{
	printf("%d/%d s=%d : ", iA, nA, i);
	for(int k=0; k<s.size(); k++) printf("%g ", s[k]);
	printf("\n"); fflush(0);
}
*/

void NAF::select(const int agentId, State& s, Action& a, State& sOld,
								 Action& aOld, const int info, Real r)
{
		if (info!=1)
		data->passData(agentId, info, sOld, aOld, s, r);  //store sOld, aOld -> sNew, r
		if (info == 2) return;
		assert(info==1 || data->Tmp[agentId]->tuples.size());

    Activation* currActivation = net->allocateActivation();
    vector<Real> output(nOutputs);

    if (info==1) {
			// if new sequence, sold, aold and reward are meaningless
			vector<Real> inputs(nInputs,0);
			s.copy_observed(inputs);
			vector<Real> scaledSold = data->standardize(inputs);
			//printselection(agentId,nAgents,info,scaledSold);
      net->predict(scaledSold, output, currActivation);
    } else {
			//then if i'm using RNN i need to load recurrent connections
			const Tuple* const last = data->Tmp[agentId]->tuples.back();
			vector<Real> scaledSold = data->standardize(last->s);
			Activation* prevActivation = net->allocateActivation();
			net->loadMemory(net->mem[agentId], prevActivation);
			//printselection(agentId,nAgents,info,scaledSold);
			net->predict(scaledSold, output, prevActivation, currActivation);
      _dispose_object(prevActivation);
    }

    //save network transition
    net->loadMemory(net->mem[agentId], currActivation);
    _dispose_object(currActivation);

    //load computed policy into a
    vector<Real> scaledAct(nA);
		for (int j=0; j<nA; j++) scaledAct[j] = output[1+nL+j];
    a.set(aInfo.getScaled(scaledAct));

    //random action?
    const Real annealedEps = bTrain ? annealingFactor() + greedyEps : greedyEps;
    uniform_real_distribution<Real> dis(0.,1.);
    if(dis(*gen) < annealedEps) a.getRandom();

	#ifdef _dumpNet_
   if (!bTrain)
		dumpNetworkInfo(agentId);
	#endif
}

void NAF::dumpNetworkInfo(const int agentId)
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

void NAF::Train_BPTT(const int seq, const int thrID) const
{
    assert(net->allocatedFrozenWeights && bTrain);
    vector<Real> target(nOutputs), output(nOutputs), gradient(nOutputs);
    const int ndata = data->Set[seq]->tuples.size();
    vector<Activation*> timeSeries = net->allocateUnrolledActivations(ndata-1);
    Activation* tgtActivation = net->allocateActivation();
    net->clearErrors(timeSeries);

    for (int k=0; k<ndata-1; k++) { //state in k=[0:N-2], act&rew in k+1, last state (N-1) not used for Q update
      const Tuple * const _t    = data->Set[seq]->tuples[k+1]; //this tuple contains a, sNew, reward
      const Tuple * const _tOld = data->Set[seq]->tuples[k]; //this tuple contains sOld
      vector<Real> scaledSold = data->standardize(_tOld->s);
      net->predict(scaledSold, output, timeSeries, k);

      const bool terminal = k+2==ndata && data->Set[seq]->ended;
      if (not terminal) {
          //vector<Real> scaledSnew = data->standardize(_t->s, __NOISE, thrID);
          vector<Real> scaledSnew = data->standardize(_t->s);
          net->predict(scaledSnew, target, timeSeries[k], tgtActivation,
									net->tgt_weights, net->tgt_biases);
      }
			#if 0
			const Real realxedGamma = gamma * (1. - annealingFactor());
      const Real Vnext = (terminal) ? _t->r : _t->r + realxedGamma*target[0];
			#else
			const Real anneal = annealingFactor(), seqRew = sequenceR(k, seq);
			const Real Vnext = (terminal) ? _t->r :
												anneal*seqRew + (1-anneal)*( _t->r + gamma*target[0]);
			#endif

			const vector<Real> Q = computeQandGrad(gradient, _t->a, output, Vnext);
			const Real err = Vnext - Q[0];

      data->Set[seq]->tuples[k]->SquaredError = err*err;
      net->setOutputDeltas(gradient, timeSeries[k]);
      dumpStats(Vstats[thrID], Q[0], err, Q);
      if(thrID==1) net->updateRunning(timeSeries[k]);
    }

    if (thrID==0) net->backProp(timeSeries, net->grad);
    else net->backProp(timeSeries, net->Vgrad[thrID]);
    net->deallocateUnrolledActivations(&timeSeries);
    _dispose_object(tgtActivation);
}

void NAF::Train(const int seq, const int samp, const int thrID) const
{
    assert(net->allocatedFrozenWeights && bTrain);
    const int ndata = data->Set[seq]->tuples.size();
    vector<Real> target(nOutputs), output(nOutputs), gradient(nOutputs);

    const Tuple* const _tOld = data->Set[seq]->tuples[samp]; //this tuple contains a, sNew, reward:
    const Tuple* const _t = data->Set[seq]->tuples[samp+1]; //this tuple contains a, sNew, reward:
    Activation* sOldActivation = net->allocateActivation();
    sOldActivation->clearErrors();

    vector<Real> scaledSold =data->standardize(_tOld->s);
    net->predict(scaledSold, output, sOldActivation); //sOld in previous tuple

    const bool terminal = samp+2==ndata && data->Set[seq]->ended;
    if (not terminal) {
        Activation* sNewActivation = net->allocateActivation();
        //vector<Real> scaledSnew = data->standardize(_t->s, __NOISE, thrID);
        vector<Real> scaledSnew = data->standardize(_t->s);
        net->predict(scaledSnew, target, sNewActivation,
														net->tgt_weights, net->tgt_biases);
        _dispose_object(sNewActivation);
    }

		const Real realxedGamma = gamma * (1. - annealingFactor());
		const Real Vnext = (terminal) ? _t->r : _t->r + realxedGamma*target[0];

    const vector<Real> Q = computeQandGrad(gradient, _t->a, output, Vnext);
    const Real err = Vnext - Q[0];
    dumpStats(Vstats[thrID], Q[0], err, Q);
    if(thrID == 1) net->updateRunning(sOldActivation);
    data->Set[seq]->tuples[samp]->SquaredError = err*err;

    if (thrID==0) net->backProp(gradient, sOldActivation, net->grad);
	else net->backProp(gradient, sOldActivation, net->Vgrad[thrID]);

    _dispose_object(sOldActivation);
}


vector<Real> NAF::computeQandGrad(vector<Real>& grad, const vector<Real>& act,
                                      const vector<Real>& out, const Real Vnext) const
    {
        vector<Real> Q(3,0), _L(nA*nA,0), _A(nA*nA,0), _dLdl(nA*nA);
        vector<Real> _dPdl(nA*nA), _u(nA), _uL(nA), _uU(nA);

        {
            int kL = 1;
            for (int j=0; j<nA; j++)
                for (int i=0; i<nA; i++) if (i<=j) _L[nA*j + i] = out[kL++];
            assert(kL==1+nL);
        }

        for (int j=0; j<nA; j++) { //compute u = act-pi and matrix L
            const Real min_a = aInfo.getActMinVal(j);
            const Real max_a = aInfo.getActMaxVal(j);
            const Real pi = aInfo.getScaled(out[1+nL+j], j);
            _u[j]  = act[j] - pi;
            _uL[j] = min_a - pi;
            _uU[j] = max_a - pi;
        }
        for (int j=0; j<nA; j++)
            for (int i=0; i<nA; i++) { //A = L * L'
                const int ind = nA*j + i;
                for (int k=0; k<nA; k++) {
                    const int k1 = nA*j + k;
                    const int k2 = nA*i + k;
                    _A[ind] += _L[k1] * _L[k2];
                }
                Q[0] += _A[ind]*_u[i]*_u[j]; //Advantage = u' * A * u
                Q[1] += _A[ind]*_uL[i]*_uL[j];
                Q[2] += _A[ind]*_uU[i]*_uU[j];
            }
        assert(*std::min_element(std::begin(Q), std::end(Q)) > 0.);

#if 1
        //original formulation of advantage = 0.5 (a - pi)' * A * (a - pi)
        const Real dQdA = -0.5;
        Q[2] = out[0] - .5*std::max(Q[1],Q[2]);
        Q[0] = out[0] - .5*Q[0];
        Q[1] = out[0];
#else
        const Real dQdA = -.5*pow(Q[0],-.75);
        Q[2] = out[0] -2.*std::pow(std::max(Q[1],Q[2]),0.25);
        Q[0] = out[0] -2.*std::pow(Q[0],0.25);  //Q = V - 2*Adv^.25
        Q[1] = out[0];
#endif

        const Real error = Vnext - Q[0];
        grad[0] = error;

        for (int il=0; il<nL; il++) {
            int kD=0;
            for (int j=0; j<nA; j++)
            for (int i=0; i<nA; i++) {
                const int ind = nA*j + i;
                _dLdl[ind] = 0;
                if(i<=j) { if(kD++==il) _dLdl[ind]=1; }
            }
            assert(kD==nL);

            for (int j=0; j<nA; j++)
            for (int i=0; i<nA; i++) {
                const int ind = nA*j + i;
                _dPdl[ind] = 0;
                for (int k=0; k<nA; k++) {
                    const int k1 = nA*j + k;
                    const int k2 = nA*i + k;
                    _dPdl[ind] += _dLdl[k1]*_L[k2] + _L[k1]*_dLdl[k2];
                }
            }

            grad[1+il] = 0.;
            for (int j=0; j<nA; j++)
            for (int i=0; i<nA; i++) {
                const int ind = nA*j + i;
                grad[1+il] += _dPdl[ind]*_u[i]*_u[j];
            }

            grad[1+il] *= dQdA*error;
        }

        for (int ia=0; ia<nA; ia++) {
            grad[1+nL+ia] = 0.;
            for (int i=0; i<nA; i++) {
                const int ind = nA*ia + i;
                grad[1+nL+ia] -= 2.*_A[ind]*_u[i];
            }
            grad[1+nL+ia] *= dQdA*error;
        }

        for (int j=0; j<nA; j++)
            grad[1+nL+j] *= aInfo.getDactDscale(out[1+nL+j], j);

        Real meangrad = 0;
        for (int i=0; i<nA+nL+1; i++)
            meangrad += std::fabs(grad[i]);
        meangrad/=(1+nA+nL);

        if(grad[0]>0 && grad[0]> meangrad)
            grad[0] = meangrad;
        if(grad[0]<0 && grad[0]<-meangrad)
            grad[0] = -meangrad;

        //1 action dim dump:
        //printf("act %9.9e, err %9.9e, out %9.9e %9.9e %9.9e, u %9.9e, Q %9.9e, grad %9.9e %9.9e %9.9e\n",
            //act[0],error,out[0],out[1],out[2],_u[0],Q[0],grad[0],grad[1],grad[2]);
        //2 actions dim dump
        //printf("act %f %f, err %f, out %f %f %f %f %f %f, u %f %f, Q %f, grad %f %f %f %f %f %f\n", act[0], act[1], error,
        //out[0], out[1], out[2], out[3], out[4], out[5], _u[0], _u[1], Q[0], grad[0], grad[1], grad[2], grad[3], grad[4], grad[5]);
        return Q;
    }
