/*
 *  NFQ.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "DPG.h"

DPG::DPG(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner(comm,_env,settings), nS(_env->sI.dimUsed*(1+settings.dqnAppendS)), nA(_env->aI.dim)
{
	buildNetwork(net, opt, vector<int>(1,1), settings);
	buildNetwork(net_policy, opt_policy, vector<int>(1,nA), settings, vector<int>(1,nA));
}

void DPG::select(const int agentId, State& s, Action& a,
								 State& sOld, Action& aOld, const int info, Real r)
{
		if (info!=1)
			data->passData(agentId, info, sOld, aOld, s, r);  //store sOld, aOld -> sNew, r
		if (info == 2) return;
		assert(info==1 || data->Tmp[agentId]->tuples.size());

    Activation* currActivation = net_policy->allocateActivation();
    vector<Real> output(nA);

    if (info==1) {// if new sequence, sold, aold and reward are meaningless
				vector<Real> inputs(nInputs,0);
		    s.copy_observed(inputs);
    		vector<Real> scaledSold = data->standardize(inputs);
        net_policy->predict(scaledSold, output, currActivation);
    } else { //then if i'm using RNN i need to load recurrent connections
				const Tuple* const last = data->Tmp[agentId]->tuples.back();
				vector<Real> scaledSold = data->standardize(last->s);
        Activation* prevActivation = net_policy->allocateActivation();
        net_policy->loadMemory(net_policy->mem[agentId], prevActivation);
        net_policy->predict(scaledSold, output, prevActivation, currActivation);
        _dispose_object(prevActivation);
    }

    //save network transition
    net_policy->loadMemory(net_policy->mem[agentId], currActivation);
    _dispose_object(currActivation);

		#ifdef _dumpNet_
    net_policy->dump(agentId);
		#endif

    //load computed policy into a
		a.set(aInfo.getScaled(output));

    const Real annealedEps = bTrain ? annealingFactor() + greedyEps : greedyEps;
    uniform_real_distribution<Real> dis(0.,1.);
    if(dis(*gen) < annealedEps)  a.getRandom();
}

void DPG::Train_BPTT(const int seq, const int thrID) const
{
  assert(net->allocatedFrozenWeights && net_policy->allocatedFrozenWeights);
	const int ndata = data->Set[seq]->tuples.size();
	vector<Activation*>valSeries=       net->allocateUnrolledActivations(ndata-1);
	vector<Activation*>polSeries=net_policy->allocateUnrolledActivations(ndata);
	net->clearErrors(valSeries);
	Grads* tmp_grad = new Grads(net->getnWeights(), net->getnBiases());
	Activation* tgtQAct = net->allocateActivation();
	//now update value network:
	{
		vector<Real> scaledSnew = data->standardize(data->Set[seq]->tuples[0]->s);
		vector<Real> scaledSold, policy(nA);
		net_policy->predict(scaledSnew, policy, polSeries, 0,
												net_policy->tgt_weights, net_policy->tgt_biases);

		for (int k=0; k<ndata-1; k++)
		{ //state in k=[0:N-2], act&rew in k+1, last state (N-1) not used for Q update
			const Tuple*const _t   =data->Set[seq]->tuples[k+1]; //contains a, sNew, r
			scaledSold = scaledSnew;
			scaledSnew = data->standardize(_t->s);

			vector<Real> vSnew(1), Q(1), gradient(1);
	    { //join state and rescaled action to predict Q
	        vector<Real> input = scaledSold;
					vector<Real> scaledA = aInfo.getInvScaled(_t->a);
	        input.insert(input.end(), scaledA.begin(), scaledA.end());
	        net->predict(input, Q, valSeries, k);
	    }

			const bool terminal = k+2==ndata && data->Set[seq]->ended;
			if (not terminal) {
				//first predict best action with policy NN w/ target weights
				net_policy->predict(scaledSnew, policy, polSeries, k+1
					//, net_policy->tgt_weights, net_policy->tgt_biases
						   );
				//then predict target value for V(s_new)
				vector<Real> input = scaledSnew;
				input.insert(input.end(),policy.begin(),policy.end());
				net->predict(input, vSnew, valSeries[k], tgtQAct,
									net->tgt_weights, net->tgt_biases);
			}
			{
				#if 0
				const Real realxedGamma = gamma * (1. - annealingFactor());
	      const Real target = (terminal) ? _t->r : _t->r + realxedGamma*vSnew[0];
				#else
				const Real anneal = annealingFactor(), seqRew = sequenceR(k, seq);
	      const Real target = (terminal) ? _t->r :
													anneal*seqRew + (1-anneal)*( _t->r + gamma*vSnew[0]);
				#endif
				gradient[0] = target - Q[0];
				data->Set[seq]->tuples[k]->SquaredError = gradient[0]*gradient[0];
			}

			net->setOutputDeltas(gradient, valSeries[k]);
			vector<Real> dumQ(2, 100); dumQ[0] = Q[0]; //avoid nans in dumpStats
			dumpStats(Vstats[thrID], Q[0], gradient[0], dumQ);
		}

		if (thrID==0) net->backProp(valSeries, net->grad);
		else net->backProp(valSeries, net->Vgrad[thrID]);
	}
	//now update policy network:
	delete polSeries.back();
	polSeries.pop_back(); //im done using the term state for the policy, and i want to bptt
	net->clearErrors(valSeries); net_policy->clearErrors(polSeries);
	{
		for (int k=0; k<ndata-1; k++) {
		  //state in k=[0:N-2], act&rew in k+1, last state (N-1) not used for Q update
			const Tuple*const _tOld=data->Set[seq]->tuples[k]; //this tuple contains sOld

			vector<Real> pol(nA), Q(1), scaledSold = data->standardize(_tOld->s);
			net_policy->predict(scaledSold, pol, polSeries, k);

			vector<Real> input = scaledSold;
			input.insert(input.end(), pol.begin(), pol.end());
			net->predict(input, Q, valSeries, k
									//, net->tgt_weights, net->tgt_biases
									);
			Q[0] = 1.; //grad
			net->setOutputDeltas(Q, valSeries[k]);
		}

		net->backProp(valSeries,
									//net->tgt_weights, net->tgt_biases,
									tmp_grad);

		for (int k=0; k<ndata-1; k++) {
			vector<Real> pol_gradient(nA);
			for(int i=0;i<nA;i++)
				pol_gradient[i] = valSeries[k]->errvals[nS+i];

			net_policy->setOutputDeltas(pol_gradient, polSeries[k]);
		}

		if (thrID==0) net_policy->backProp(polSeries, net_policy->grad);
		else net_policy->backProp(polSeries, net_policy->Vgrad[thrID]);
	}

	net->deallocateUnrolledActivations(&valSeries);
	net_policy->deallocateUnrolledActivations(&polSeries);
	_dispose_object(tgtQAct);
	_dispose_object(tmp_grad);
}

void DPG::Train(const int seq, const int samp, const int thrID) const
{
    assert(net->allocatedFrozenWeights && net_policy->allocatedFrozenWeights);

    //this tuple contains a, sNew, reward:
    const Tuple * const _t = data->Set[seq]->tuples[samp+1];
    //sOld contained in previous tuple
    const Tuple * const _tOld = data->Set[seq]->tuples[samp];
    vector<Real> scaledSnew = data->standardize(_t->s);
    vector<Real> scaledSold = data->standardize(_tOld->s);
    Activation* sOldAAct = net_policy->allocateActivation();
    Activation* sOldQAct = net->allocateActivation();
    Activation* sNewQAct = net->allocateActivation();
    sOldAAct->clearErrors();
    sOldQAct->clearErrors();
    sNewQAct->clearErrors();

    //update Q network:
    vector<Real> vSnew(1), Q(1), gradient(1);
    { //join state and rescaled action to predict Q
        vector<Real> input = scaledSold, scaledA = aInfo.getInvScaled(_t->a);
        input.insert(input.end(),scaledA.begin(),scaledA.end());
        net->predict(input, Q, sOldQAct);
    }

    const bool terminal = samp+2==data->Set[seq]->tuples.size()
												 && data->Set[seq]->ended;
    if (not terminal) {
        Activation* sNewAAct = net_policy->allocateActivation();
        //first predict best action with policy NN w/ target weights
        vector<Real> policy(nA);
        net_policy->predict(scaledSnew, policy, sNewAAct,
														net_policy->tgt_weights, net_policy->tgt_biases);
        //then predict target value for V(s_new)
        vector<Real> input = scaledSnew;
        input.insert(input.end(),policy.begin(),policy.end());
        net->predict(input, vSnew, sNewQAct, net->tgt_weights, net->tgt_biases);
        _dispose_object(sNewAAct);
    }

		{
			const Real realxedGamma = gamma * (1. - annealingFactor());
			const Real target = (terminal) ? _t->r : _t->r + realxedGamma*vSnew[0];
		  gradient[0] = target - Q[0];
		  data->Set[seq]->tuples[samp]->SquaredError = gradient[0]*gradient[0];
		}

    if (thrID==0) net->backProp(gradient, sOldQAct, net->grad);
    else net->backProp(gradient, sOldQAct, net->Vgrad[thrID]);
    dumpStats(Vstats[thrID], Q[0], gradient[0], Q);

    //now update policy network:
    { //predict policy for sOld
        vector<Real> policy(nA), pol_gradient(nA);
        net_policy->predict(scaledSold, policy, sOldAAct);

        //use it to compute activation with frozen weitghts for Q net
        vector<Real> input = scaledSold;
        input.insert(input.end(), policy.begin(), policy.end());
				net->predict(input, Q, sNewQAct
											, net->tgt_weights, net->tgt_biases
											);

        //now i need to compute dQ/dA, for Q net use tgt weight throughout
        gradient[0] = 1.; //who to increase Q?
	    	Grads* tmp_grad = new Grads(net->getnWeights(), net->getnBiases());
				net->backProp(gradient, sNewQAct,
											net->tgt_weights, net->tgt_biases,
											tmp_grad);
        _dispose_object(tmp_grad);

    		if (thrID==0)
					net_policy->backProp(pol_gradient, sOldAAct, net_policy->grad);
    		else
					net_policy->backProp(pol_gradient, sOldAAct, net_policy->Vgrad[thrID]);
    }

    _dispose_object(sOldAAct);
    _dispose_object(sOldQAct);
    _dispose_object(sNewQAct);
}

void DPG::updateTargetNetwork()
{
		assert(bTrain);
    if (cntUpdateDelay <= 0) { //DQN-style frozen weight
        cntUpdateDelay = tgtUpdateDelay;

        //2 options: either move tgt_wght = (1-a)*tgt_wght + a*wght
	/*
        if (tgtUpdateDelay==0) {
            net->moveFrozenWeights(tgtUpdateAlpha);
            net_policy->moveFrozenWeights(tgtUpdateAlpha);
        } else {
            net->updateFrozenWeights(); //or copy tgt_wghts = wghts
            net_policy->updateFrozenWeights(); //or copy tgt_wghts = wghts
        }
	*/
	opt->moveFrozenWeights(tgtUpdateAlpha);
	opt_policy->moveFrozenWeights(tgtUpdateAlpha);
    }

    cntUpdateDelay--;
}

void DPG::stackAndUpdateNNWeights(const int nAddedGradients)
{
		assert(nAddedGradients>0 && bTrain);
    opt->nepoch ++;
    opt->stackGrads(net->grad, net->Vgrad); //add up gradients across threads
    opt->update(net->grad,nAddedGradients); //update

    opt_policy->nepoch ++;
    opt_policy->stackGrads(net_policy->grad, net_policy->Vgrad); //add up gradients across threads
    opt_policy->update(net_policy->grad,nAddedGradients); //update
}

void DPG::updateNNWeights(const int nAddedGradients)
{
		assert(nAddedGradients>0 && bTrain);
    opt->nepoch ++;
    opt->update(net->grad,nAddedGradients);

    opt_policy->nepoch ++;
    opt_policy->update(net_policy->grad,nAddedGradients);
}
