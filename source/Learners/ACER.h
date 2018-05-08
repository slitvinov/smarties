/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "../Network/Builder.h"
#include "Learner_offPolicy.h"
#include "../Math/Quadratic_advantage.h"
#include "../Math/Discrete_advantage.h"

class ACER : public Learner_offPolicy
{
 protected:
  using Policy_t = Gaussian_policy;
  using Action_t = Rvec;
  const Uint nA = Policy_t::compute_nA(&aInfo);
  const Real acerTrickPow = 1. / std::sqrt(nA);
  //const Real acerTrickPow = 1. / nA;
  const Uint nAexpectation = 5;
  const Real facExpect = 1./nAexpectation;
  const Real alpha = 1.0;
  //const Real alpha = 0.1;
  Aggregator* relay = nullptr;

  inline Policy_t prepare_policy(const Rvec& out,
    const Tuple*const t = nullptr) const {
    Policy_t pol({0, nA}, &aInfo, out);
    if(t not_eq nullptr) pol.prepare(t->a, t->mu);
    return pol;
  }

  void TrainBySequences(const Uint seq, const Uint thrID) const override
  {
    Sequence* const traj = data->Set[seq];
    const int ndata = traj->tuples.size()-1;
    //policy : we need just 2 calls: pi pi_tilde
     F[0]->prepare_seq(traj, thrID);
     F[1]->prepare_seq(traj, thrID);
    relay->prepare_seq(traj, thrID, ACT);
    //advantage : 1+nAexpect [A(s,a)] + 1 [A(s,a'), same normalization] calls
     F[2]->prepare_seq(traj, thrID, 1+nAexpectation);

    Rvec Vstates(ndata, 0);
    vector<Action_t> policy_samples(ndata);
    vector<Policy_t> policies, policies_tgt;
    policies_tgt.reserve(ndata); policies.reserve(ndata);
    vector<Rvec> advantages(ndata, Rvec(2+nAexpectation, 0));

    if(thrID==1) profiler->stop_start("FWD");
    for(Uint k=0; k<(Uint)ndata; k++) {
      const Rvec outPc = F[0]->forward<CUR>(traj, k, thrID);
      policies.push_back(prepare_policy(outPc, traj->tuples[k]));
      assert(policies.size() == k+1);
      const Rvec outPt = F[0]->forward<TGT>(traj, k, thrID);
      policies_tgt.push_back(prepare_policy(outPt));
      const Rvec outVs = F[1]->forward(traj, k, thrID);

      relay->set(policies[k].sampAct, k, thrID);
      //if(thrID) cout << "Action: " << print(policies[k].sampAct) << endl;
      const Rvec At = F[2]->forward<CUR>    (traj, k, thrID);
      policy_samples[k] = policies[k].sample(&generators[thrID]);
      //if(thrID) cout << "Sample: " << print(policy_samples[k]) << endl;
      relay->set(policy_samples[k], k, thrID);
      const Rvec Ap = F[2]->forward<CUR,TGT>(traj, k, thrID);
      advantages[k][0] = At[0]; advantages[k][1] = Ap[0]; Vstates[k] = outVs[0];
      for(Uint i = 0; i < nAexpectation; i++) {
        relay->set(policies[k].sample(&generators[thrID]), k, thrID);
        const Rvec A = F[2]->forward(traj, k, thrID, 1+i);
        advantages[k][2+i] = A[0];
      }
      //cout << print(advantages[k]) << endl; fflush(0);
    }
    assert(traj->ended);
    Real Q_RET = data->scaledReward(traj, ndata);
    Real Q_OPC = data->scaledReward(traj, ndata);
    if(thrID==1)  profiler->stop_start("POL");
    for(int k=ndata-1; k>=0; k--)
    {
      Real QTheta = Vstates[k]+advantages[k][0], APol = advantages[k][1];
      for(Uint i = 0; i < nAexpectation; i++) {
        QTheta -= facExpect*advantages[k][2+i];
        APol -= facExpect*advantages[k][2+i];
      }
      const Real A_OPC = Q_OPC - Vstates[k], Q_err = Q_RET - QTheta;

      const Real W = std::min((Real)1, policies[k].sampImpWeight);
      const Real C = std::pow(W, acerTrickPow); //as in paper, but might be bad
      const Real R = data->scaledReward(traj, k), V_err = Q_err*W;
      const Rvec pGrad = policyGradient(traj->tuples[k], policies[k],
        policies_tgt[k], A_OPC, APol, policy_samples[k]);
      F[0]->backward(pGrad,   k, thrID);
      F[1]->backward({alpha*(V_err+Q_err)}, k, thrID);
      F[2]->backward({alpha*Q_err}, k, thrID);
      for(Uint i = 0; i < nAexpectation; i++)
        F[2]->backward({-alpha*facExpect*Q_err}, k, thrID, i+1);
      //prepare Q with off policy corrections for next step:
      Q_RET = R +gamma*( C*(Q_RET-QTheta) +Vstates[k]);
      Q_OPC = R +gamma*((Q_OPC-QTheta)+Vstates[k]); //as paper, but might be bad
      //Q_OPC = R +gamma*( C*(Q_OPC-QTheta) +Vstates[k]);
      //traj->SquaredError[k] = std::min(1/policies[k].sampImpWeight, policies[k].sampImpWeight);
      traj->offPolicImpW[k] = policies[k].sampImpWeight;
      Vstats[thrID].dumpStats(QTheta, Q_err);
    }

    if(thrID==1)  profiler->stop_start("BCK");
     F[0]->gradient(thrID);
     F[1]->gradient(thrID);
     F[2]->gradient(thrID);
  }

  void Train(const Uint seq, const Uint obs, const Uint thrID) const override
  {
    die("not allowed");
  }

  inline Rvec policyGradient(const Tuple*const _t,
    const Policy_t& POL, const Policy_t& TGT,
    const Real ARET, const Real APol,
    const Action_t& pol_samp) const {
    //compute quantities needed for trunc import sampl with bias correction
    const Real polProbBehavior = POL.evalBehavior(pol_samp, _t->mu);
    const Real polProbOnPolicy = POL.evalProbability(pol_samp);
    const Real rho_pol = polProbOnPolicy / polProbBehavior;
    const Real gain1 = ARET*std::min((Real) 5, POL.sampImpWeight);
    const Real gain2 = APol*std::max((Real) 0, 1-5/rho_pol);
    const Rvec gradAcer_1 = POL.policy_grad(POL.sampAct, gain1);
    const Rvec gradAcer_2 = POL.policy_grad(pol_samp,    gain2);
    const Rvec penal = POL.div_kl_grad(&TGT, 1);
    const Rvec grad = sum2Grads(gradAcer_1, gradAcer_2);
    const Rvec trust = trust_region_update(grad, penal, 2*nA, 1);
    return POL.finalize_grad(trust);
  }

 public:
  void select(const Agent& agent) override
  {
    const int thrID= omp_get_thread_num();
    Sequence* const traj = data->inProgress[agent.ID];
    data->add_state(agent);

    if( agent.Status < TERM_COMM ) {
      //Compute policy and value on most recent element of the sequence. If RNN
      // recurrent connection from last call from same agent will be reused
      Rvec output = F[0]->forward_agent(traj, agent, thrID);
      Policy_t pol = prepare_policy(output);
      Rvec beta = pol.getVector();
      const Action_t act = pol.finalize(greedyEps>0, &generators[thrID], beta);
      agent.a->set(act);
      data->add_action(agent, beta);
    }
    else data->terminate_seq(agent);
  }

  static Uint getnDimPolicy(const ActionInfo*const aI)
  {
    return 2*aI->dim;
  }

  ACER(Environment*const _env, Settings&_set): Learner_offPolicy(_env,_set)
  {
    _set.splitLayers = 0;
    #if 1
      if(input->net not_eq nullptr) {
        delete input->opt; input->opt = nullptr;
        delete input->net; input->net = nullptr;
      }
      Builder input_build(_set);
      bool bInputNet = false;
      input_build.addInput( input->nOutputs() );
      bInputNet = bInputNet || env->predefinedNetwork(input_build);
      bInputNet = bInputNet || predefinedNetwork(input_build);
      if(bInputNet) {
        Network* net = input_build.build(true);
        input->initializeNetwork(net, input_build.opt);
      }
    #endif

    relay = new Aggregator(_set, data, _env->aI.dim);
    F.push_back(new Approximator("policy", _set, input, data));
    F.push_back(new Approximator("value",  _set, input, data));
    F.push_back(new Approximator("advntg", _set, input, data, relay));

    Builder build_pol = F[0]->buildFromSettings(_set, nA);
    const Real initParam = noiseMap_inverse(greedyEps);
    build_pol.addParamLayer(nA, "Linear", initParam);
    Builder build_val = F[1]->buildFromSettings(_set, 1 ); // V
    Builder build_adv = F[2]->buildFromSettings(_set, 1 ); // A

    F[0]->initializeNetwork(build_pol);
    //_set.learnrate *= 10;
    //const Real backup = _set.nnLambda;
    //_set.nnLambda = 0.01;
    F[1]->initializeNetwork(build_val);
    F[2]->initializeNetwork(build_adv);
    //_set.nnLambda = backup;
    //_set.learnrate /= 10;
    F[2]->allocMorePerThread(nAexpectation);
    printf("ACER\n");

    {  // TEST FINITE DIFFERENCES:
      Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
      std::normal_distribution<Real> dist(0, 1);
      for(Uint i=0; i<output.size(); i++) output[i] = dist(generators[0]);
      for(Uint i=0;  i<mu.size(); i++) mu[i] = dist(generators[0]);
      for(Uint i=nA; i<mu.size(); i++) mu[i] = std::exp(mu[i]);

      Policy_t pol = prepare_policy(output);
      Rvec act = pol.finalize(1, &generators[0], mu);
      pol.prepare(act, mu);
      pol.test(act, mu);
    }
  }
  ~ACER() { }
};
