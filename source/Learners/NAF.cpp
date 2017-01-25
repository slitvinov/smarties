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



NAF::NAF(MPI_Comm comm, Environment*const env, Settings & settings) :
Learner(comm,env,settings), nA(env->aI.dim),
nL((env->aI.dim*env->aI.dim+env->aI.dim)/2)
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
}

void NAF::select(const int agentId, State& s, Action& a, State& sOld,
								 Action& aOld, const int info, Real r)
{
    Activation* currActivation = net->allocateActivation();
    vector<Real> output(nOutputs), inputs(nInputs);

    s.copy_observed(inputs);
    vector<Real> scaledSold = data->standardize(inputs);

    if (info==1) // if new sequence, sold, aold and reward are meaningless
        net->predict(scaledSold, output, currActivation);
    else {   //then if i'm using RNN i need to load recurrent connections
    		Activation* prevActivation = net->allocateActivation();
				net->loadMemory(net->mem[agentId], prevActivation);
        net->predict(scaledSold, output, prevActivation, currActivation);
        //also, store sOld, aOld -> sNew, r
        data->passData(agentId, info, sOld, aOld, s, r);
        _dispose_object(prevActivation);
    }

    //save network transition
    net->loadMemory(net->mem[agentId], currActivation);
    _dispose_object(currActivation);

    #ifdef _dumpNet_
    net->dump(agentId);
    #endif

    //load computed policy into a
    vector<Real> act(nA);
    for (int j(0); j<nA; j++) act[j] = output[1+nL+j];
    a.set(act);

    //random action?
    Real newEps(greedyEps);
    if (bTrain) { //if training: anneal random chance if i'm just starting to learn
        const int handicap = min(static_cast<int>(data->Set.size())/500.,
                              (bRecurrent ? opt->nepoch/100. : opt->nepoch/100.));
        newEps = exp(-handicap) + greedyEps;//*agentId/Real(agentId+1);
    }

    uniform_real_distribution<Real> dis(0.,1.);
    if(dis(*gen) < newEps) a.getRandom();
      /*
    if(dis(*gen) < newEps) {
        a.getRandom();
        printf("Random action %f  for state %s\n",a.vals[0], s.print().c_str());fflush(0);
    } else {
        printf("Net selected %f for state %s\n", a.vals[0], s.print().c_str()); fflush(0);
    }
      */
}

void NAF::Train_BPTT(const int seq, const int thrID) const
{
    assert(net->allocatedFrozenWeights);
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
            vector<Real> scaledSnew = data->standardize(_t->s);
            net->predict(scaledSnew, target, timeSeries[k], tgtActivation,
																						net->tgt_weights, net->tgt_biases);
        }

        Real err = (terminal) ? _t->r : _t->r + gamma*target[0];
				Real temp = err;
        const vector<Real> Q(computeQandGrad(gradient, _t->a, output, err));
				/* // to ignore
					if (std::fabs(output[1+nL])>0.7) {
					printf("in %e %e %e %e %e %e, out %e %e %e, a %e, r %e, y %e, err %e (%e), g %e %e %e\n",
					_tOld->s[0], _tOld->s[1], _tOld->s[2], _tOld->s[3], _tOld->s[4], _tOld->s[5],
					output[0],output[1],output[2],_t->a[0],_t->r,target[0], err,temp,gradient[0],gradient[1],gradient[2]);
					}

					if (output[0] > 1./(1.-gamma) && gradient[0]>0) gradient[0] = 0;
					//if (std::fabs(output[1+nL]) > 0.75 && gradient[1+nL]*output[1+nL] > 0) {
					//		//gradient[1] = 0;
					//		gradient[1+nL] = 0;
					//}
				*/
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
    assert(net->allocatedFrozenWeights);
    const int ndata = data->Set[seq]->tuples.size();
    vector<Real> target(nOutputs), output(nOutputs), gradient(nOutputs);

    vector<Real> scaledSold =data->standardize(data->Set[seq]->tuples[samp]->s);
    const Tuple* const _tOld = data->Set[seq]->tuples[samp]; //this tuple contains a, sNew, reward:
    const Tuple* const _t = data->Set[seq]->tuples[samp+1]; //this tuple contains a, sNew, reward:
    Activation* sOldActivation = net->allocateActivation();
    sOldActivation->clearErrors();

    net->predict(scaledSold, output, sOldActivation); //sOld in previous tuple

    const bool terminal = samp+2==ndata && data->Set[seq]->ended;
    if (not terminal) {
        Activation* sNewActivation = net->allocateActivation();
        vector<Real> scaledSnew = data->standardize(_t->s);
        net->predict(scaledSnew, target, sNewActivation,
														net->tgt_weights, net->tgt_biases);
        _dispose_object(sNewActivation);
    }

    Real err = (terminal) ? _t->r : _t->r + gamma*target[0];
	 	Real temp = err;
    const vector<Real> Q(computeQandGrad(gradient, _t->a, output, err));
		/* // to ignore
			if (std::fabs(output[1+nL])>0.7) {
			printf("in %e %e %e %e %e %e, out %e %e %e, a %e, r %e, y %e, err %e (%e), g %e %e %e\n",
			_tOld->s[0], _tOld->s[1], _tOld->s[2], _tOld->s[3], _tOld->s[4], _tOld->s[5],
			output[0],output[1],output[2],_t->a[0],_t->r,target[0], err,temp,gradient[0],gradient[1],gradient[2]);
			}

			if (output[0] > 1./(1.-gamma) && gradient[0]>0) gradient[0] = 0;
	  */
    dumpStats(Vstats[thrID], Q[0], err, Q);
    if(thrID == 1) net->updateRunning(sOldActivation);
    data->Set[seq]->tuples[samp]->SquaredError = err*err;

    if (thrID==0) net->backProp(gradient, sOldActivation, net->grad);
	else net->backProp(gradient, sOldActivation, net->Vgrad[thrID]);

    _dispose_object(sOldActivation);
}

#if 0 //original formulation of advantage = 0.5 (a - pi)' * A * (a - pi), does not work: why?

vector<Real> NAF::computeQandGrad(vector<Real>& grad, const vector<Real>& act,
																	const vector<Real>& out, Real& error) const
{
    vector<Real> Q(3,0), _L(nA*nA,0), _A(nA*nA,0), _dLdl(nA*nA), _dPdl(nA*nA);
		vector<Real> _u(nA), _uL(nA), _uU(nA);

    int kL = 1; //skip out[0] == V(state)
    for (int j=0; j<nA; j++)
    {
        //compute u = act-pi
        _u[j]  = act[j] - out[1+nL+j];
        //to compute the relative error: Q for the max and min actions
        _uL[j] = *std::min_element(std::begin(aInfo.values[j]),
																	 std::end(aInfo.values[j])   ) - out[1+nL+j];
        _uU[j] = *std::max_element(std::begin(aInfo.values[j]),
																	 std::end(aInfo.values[j])   ) - out[1+nL+j];

        //put in place elements of lower diag matrix L
        for (int i=0; i<nA; i++) if (i<=j) _L[nA*j + i] = out[kL++];
    }

   /*
   std::stringstream ooo;
		ooo << "[";
		for (int i=0; i<nA*nA; i++)
	      ooo << _L[i] << " ";
		ooo << "]";
   printf("%s\n", ooo.str().c_str());
   fflush(0);
   */
    assert(kL==1+nL);

    //A = L * L'
    for (int j=0; j<nA; j++)
    for (int i=0; i<nA; i++) {
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

    //Q(s,a) = V(s) - .5 * Advantage(s,a)
    Q[2] = out[0] - .5*std::max(Q[1],Q[2]);
    Q[0] = out[0] - .5*Q[0];
    Q[1] = out[0];
    error -= Q[0];

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
            grad[1+il] += -0.5*_dPdl[ind]*_u[i]*_u[j];
        }

        grad[1+il] *= error;
    }

    for (int ia=0; ia<nA; ia++) {
        grad[1+nL+ia] = 0.;
        for (int i=0; i<nA; i++) {
            const int ind = nA*ia + i;
            grad[1+nL+ia] += _A[ind]*_u[i];
        }
        grad[1+nL+ia] *= error;
    }
    //1 action dim dump:
    //printf("act %9.9e, err %9.9e, out %9.9e %9.9e %9.9e, u %9.9e, Q %9.9e, grad %9.9e %9.9e %9.9e\n",
		//act[0],error,out[0],out[1],out[2],_u[0],Q[0],grad[0],grad[1],grad[2]);
    //2 actions dim dump
    ///printf("act %f %f, err %f, out %f %f %f %f %f %f, u %f %f, Q %f, grad %f %f %f %f %f %f\n", act[0], act[1], error,
    ///out[0], out[1], out[2], out[3], out[4], out[5], _u[0], _u[1], Q[0], grad[0], grad[1], grad[2], grad[3], grad[4], grad[5]);
    return Q;
}

#else  //my formulation of advantage = 2 pow((a - pi)' * A * (a - pi), 0.25), works

vector<Real> NAF::computeQandGrad(vector<Real>& grad, const vector<Real>& act,
																	const vector<Real>& out, Real& error) const
{
    vector<Real> Q(3,0), _L(nA*nA,0), _A(nA*nA,0), _dLdl(nA*nA), _dPdl(nA*nA), _u(nA), _uL(nA), _uU(nA);

    int kL(1);
    for (int j=0; j<nA; j++) { //compute u = act-pi and matrix L
        _u[j]  = act[j] - out[1+nL+j];
        _uL[j] = *std::min_element(std::begin(aInfo.values[j]), std::end(aInfo.values[j])) - out[1+nL+j];
        _uU[j] = *std::max_element(std::begin(aInfo.values[j]), std::end(aInfo.values[j])) - out[1+nL+j];

        for (int i=0; i<nA; i++) if (i<=j) _L[nA*j + i] = out[kL++];
    }
    assert(kL==1+nL);

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

    const Real dQdA = -.5*pow(Q[0],-.75);
    Q[2] = out[0] -2.*std::pow(std::max(Q[1],Q[2]),0.25);
    Q[0] = out[0] -2.*std::pow(Q[0],0.25);  //Q = V - 2*Adv^.25
    Q[1] = out[0];
    error -= Q[0];

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
            for (int k(0); k<nA; k++) {
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
    //1 action dim dump:
		//printf("act %9.9e, err %9.9e, out %9.9e %9.9e %9.9e, u %9.9e, Q %9.9e, grad %9.9e %9.9e %9.9e\n",
		//act[0], error, out[0], out[1], out[2], _u[0], Q[0], grad[0], grad[1], grad[2]);
    //2 actions dim dump
    ///printf("act %f %f, err %f, out %f %f %f %f %f %f, u %f %f, Q %f, grad %f %f %f %f %f %f\n", act[0], act[1], error, out[0],  out[1], out[2], out[3],  out[4], out[5], _u[0], _u[1], Q[0], grad[0], grad[1], grad[2], grad[3], grad[4], grad[5]);
    return Q;
}
#endif

/*
#ifdef _scaleR_
vector<Real> NAF::computeQandGrad(vector<Real>& grad, const vector<Real>& act, const vector<Real>& out, Real& error) const
{
    vector<Real> Q(3, out[0]), _u(nA), _uL(nA), _uU(nA);

    for (int j(0); j<nA; j++) {
        _u[j]  = act[j] - out[1+nA+j];
        _uL[j] = -1.    - out[1+nA+j];
        _uU[j] =  1.    - out[1+nA+j];
        const Real A = -std::log(.5-.5*out[1+j]);
        //Q[0] -= A*fabs(_u[j] );
        //Q[1] -= A*fabs(_uL[j]);
        //Q[2] -= A*fabs(_uU[j]);
        Q[0] -= sqrt(A*fabs(_u[j] ));
        Q[1] -= sqrt(A*fabs(_uL[j]));
        Q[2] -= sqrt(A*fabs(_uU[j]));
    }

    error -= Q[0];
    Q[2] = std::min(Q[1],Q[2]);
    Q[1] = out[0];

    grad[0] = error;
    for (int j(0); j<nA; j++) {
        const Real A = -std::log(.5-.5*out[1+j]);
        grad[1+j]    = -error*fabs(_u[j])/(1-out[1+j]);
        grad[1+nA+j] = (_u[j]>0.) ? error*A : -error*A;
        const Real fac = 0.5 / sqrt( A * fabs(_u[j] ));
        grad[1+j]    = -error*fac*fabs(_u[j])/(1-out[1+j]);
        grad[1+nA+j] = (_u[j]>0.) ? error*A*fac : -error*A*fac;
    }

    //printf("act %f, err %f, out %f %f %f, u %f, Q %f, grad %f %f %f\n", act[0], error, out[0],  out[1], out[2], _u[0], Q[0], grad[0], grad[1], grad[2]);
    return Q;
}
#else

 //THE FOLLOWING HAS ADV = -A*ABS(ACT-PI) WHERE A IS A POSITIVE COEFFICIENT
vector<Real> NAF::computeQandGrad(vector<Real>& grad, const vector<Real>& act, const vector<Real>& out, Real& error) const
{
    vector<Real> Q(3, out[0]), _u(nA), _uL(nA), _uU(nA);

    for (int j(0); j<nA; j++) {
        _u[j]  = act[j] - out[1+nA+j];
        _uL[j] = -1.    - out[1+nA+j];
        _uU[j] =  1.    - out[1+nA+j];
        const Real A = (out[1+j]<0) ? 0 : out[1+j];
        Q[0] -= sqrt(A*fabs(_u[j] ));
        Q[1] -= sqrt(A*fabs(_uL[j]));
        Q[2] -= sqrt(A*fabs(_uU[j]));
    }

    error -= Q[0];
    Q[2] = std::min(Q[1],Q[2]);
    Q[1] = out[0];

    grad[0] = error;
    for (int j(0); j<nA; j++) {
        if (out[1+j]<=0) {
            grad[1+j] = 10;
            grad[1+nA+j] = 0;
        } else {
            const Real fac = .5/sqrt(out[1+j]*fabs(_u[j]));
            grad[1+j]    = -error*fac*fabs(_u[j]);
            grad[1+nA+j] = (_u[j]>0.) ? error*fac*out[1+j] : -error*fac*out[1+j];
        }
    }

    //printf("act %f, err %f, out %f %f %f, u %f, Q %f, grad %f %f %f\n", act[0], error, out[0],  out[1], out[2], _u[0], Q[0], grad[0], grad[1], grad[2]);
    return Q;
}
#endif

 //THE FOLLOWING HAS ADV = -A*ABS(ACT-PI) WHERE A IS A POSITIVE COEFFICIENT
 // A = A1 IF ACT-PI>0, ELSE A=A2
vector<Real> NAF::computeQandGrad(vector<Real>& grad, const vector<Real>& act, const vector<Real>& out, Real& error) const
{
    vector<Real> Q(3, out[0]), _u(nA), _uL(nA), _uU(nA);
    //printf("out %f %f %f %f \n", out[0], out[1], out[2], out[3]);

    for (int j(0); j<nA; j++) {
        _u[j]  = act[j]- out[1+2*nA+j];
        _uL[j] = -1.   - out[1+2*nA+j];
        _uU[j] =  1.   - out[1+2*nA+j];

        const Real A1 = (out[1   +j]<0) ? 0 : out[1+j];
        const Real A2 = (out[1+nA+j]<0) ? 0 : out[1+nA+j];
        Q[0] += (_u[j]>0. ) ? -A1*_u[j]  : A2*_u[j];
        Q[1] += (_uL[j]>0.) ? -A1*_uL[j] : A2*_uL[j];
        Q[2] += (_uU[j]>0.) ? -A1*_uU[j] : A2*_uU[j];
    }

    error -= Q[0];
    Q[2] = std::min(Q[1],Q[2]);
    Q[1] = out[0];

    grad[0] = error;
    for (int j(0); j<nA; j++) {
        const Real A1 = (out[1   +j]<0) ? 0 : out[1+j];
        const Real A2 = (out[1+nA+j]<0) ? 0 : out[1+nA+j];
        grad[1+2*nA+j] = (_u[j] > 0.) ? error*A1 : -error*A2;
        grad[1+j]    = (out[1+j]<0)   ? 10. : ((_u[j]>0.)? -error*_u[j] : 0);
        grad[1+nA+j] = (out[1+nA+j]<0)? 10. : ((_u[j]<0.)?  error*_u[j] : 0);
    }
    //printf("act %f, err %f, out %f %f %f %f , u %f, Q %f, grad %f %f %f %f \n", act[0], error, out[0], out[1], out[2], out[3], _u[0], Q[0], grad[0], grad[1], grad[2], grad[3]);
    return Q;
}

Real NAF::computeQandGrad(vector<Real>& grad, const vector<Real>& act, const vector<Real>& out, const Real error) const
{
    Real Q(out[0]);
    vector<Real> _L(nA*nA,0), _A(nA*nA,0), _dLdl(nA*nA), _dPdl(nA*nA), _u(nA);
    grad[0] = 1.;

    int kL(1);
    for (int j(0); j<nA; j++) {
        _u[j] = act[j] - out[1+nL+j];
        for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            if (i<=j) _L[ind] = out[kL++];
        }
    }
    assert(kL==1+nL);

    for (int j(0); j<nA; j++) for (int i(0); i<nA; i++) {
        const int ind = nA*j + i;
        for (int k(0); k<nA; k++) {
            const int k1 = nA*j + k;
            const int k2 = nA*i + k;
            _A[ind] += _L[k1] * _L[k2];
        }
        Q -= .5*_A[ind]*_u[i]*_u[j];
    }

    for (int il(0); il<nL; il++) {
        int kD(0);
        for (int j(0); j<nA; j++) for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            _dLdl[ind] = 0;
            if(i<=j) { if(kD++==il) _dLdl[ind]=1; }
        }
        assert(kD==nL);

        for (int j(0); j<nA; j++) for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            _dPdl[ind] = 0;
            for (int k(0); k<nA; k++) {
                const int k1 = nA*j + k;
                const int k2 = nA*i + k;
                _dPdl[ind] += _dLdl[k1]*_L[k2]+_L[k1]*_dLdl[k2];
            }
        }

        grad[1+il] = 0.;
        for (int j(0); j<nA; j++) for (int i(0); i<nA; i++) {
            const int ind = nA*j + i;
            grad[1+il] += -0.5*_dPdl[ind]*_u[i]*_u[j];
        }
    }

    for (int ia(0); ia<nA; ia++) {
        grad[1+nL+ia] = 0.;
        for (int i(0); i<nA; i++) {
            const int ind = nA*ia + i;
            grad[1+nL+ia] += _A[ind]*_u[i];
        }
    }

    return Q;
}*/
