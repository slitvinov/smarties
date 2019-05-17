//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

namespace smarties
{

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::TrainBySequences(const Uint seq,
  const Uint wID, const Uint bID, const Uint thrID) const
{
  Sequence* const traj = data->get(seq);
  const int ndata = traj->ndata();
  if(thrID==0) profiler->stop_start("FWD");

  F[0]->prepare_seq(traj, thrID, wID);
  for (int k=0; k<ndata; k++) F[0]->forward(k, thrID);

  //if partial sequence then compute value of last state (!= R_end)
  Real Q_RET = data->scaledReward(traj, ndata);
  if( traj->isTruncated(ndata) ) {
    const Rvec nxt = F[0]->forward(ndata, thrID);
    Q_RET = gamma * nxt[VsID];
  }

  if(thrID==0)  profiler->stop_start("POL");
  for(int k=ndata-1; k>=0; k--)
  {
    const Rvec out_cur = F[0]->get(k, thrID);
    const Rvec & ACT = traj->actions[k], & MU = traj->policies[k];
    const Policy_t pol = prepare_policy<Policy_t>(out_cur, ACT, MU);
    // far policy definition depends on rho (as in paper)
    // in case rho outside bounds, do not compute gradient

    const Real W = pol.sampImpWeight, R = data->scaledReward(traj, k);
    const Real V_Sk = out_cur[0], A_RET = Q_RET - V_Sk;
    const Real D_RET = std::min((Real)1, W) * A_RET;
      // check whether importance weight is in 1/CmaxRet < c < CmaxRet
    const bool isOff = traj->isFarPolicy(k, W, CmaxRet, CinvRet);
    traj->setMseDklImpw(k, D_RET*D_RET, pol.sampKLdiv, W, CmaxRet, CinvRet);
    trainInfo->log(V_Sk, D_RET, { std::pow(0,2), W }, thrID);

    // compute the gradient
    Rvec G = Rvec(F[0]->nOutputs(), 0);
    if (isOff) {
      pol.finalize_grad(pol.div_kl_grad(MU, alpha*(beta-1)), G);
    }  else {
      const Rvec policyG = pol.policy_grad(pol.sampAct, alpha * A_RET * W);
      const Rvec penaltG = pol.div_kl_grad(MU, -alpha);
      pol.finalize_grad(weightSum2Grads(policyG, penaltG, beta), G);
      trainInfo->trackPolicy(policyG, penaltG, thrID);
    }

    // value gradient:
    assert(std::fabs(G[0])<1e-16); // make sure it was untouched
    G[0] = (1-alpha) * beta * D_RET;
    F[0]->backward(G, k, thrID);

    // update retrace for the previous step k-1:
    Q_RET = R + gamma*(V_Sk + std::min((Real)1,W)*(Q_RET - V_Sk) );
  }

  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::Train(const MiniBatch&MB, const Uint wID,const Uint bID) const
{
  Sequence* const S = data->get(seq);
  assert(t+1 < S->ndata());

  if(thrID==0) profiler->stop_start("FWD");
  F[0]->prepare_one(S, t, thrID, wID); // prepare thread workspace
  const Rvec out = F[0]->forward(t, thrID); // network compute
  const Rvec & MU = S->policies[t];
  #ifdef DACER_singleNet
    static constexpr int valNetID = 0;
    const Rvec& val = out;
  #else
    static constexpr int valNetID = 1;
    F[1]->prepare_one(S, t, thrID, wID); // prepare thread workspace
    const Rvec val = F[1]->forward(t, thrID); // network compute
  #endif

  if ( wID == 0 and S->isTruncated(t+1) ) {
    assert( t+1 == S->ndata() );
    const Rvec nxt = F[valNetID]->forward(t+1, thrID);
    updateRetrace(S, t+1, 0, nxt[VsID], 0);
  }

  if(thrID==0)  profiler->stop_start("CMP");
  const auto P = prepare_policy<Policy_t>(out, S->actions[t], MU);
  const Real W = P.sampImpWeight; // \rho = \pi / \mu
  const Real A_RET = S->Q_RET[t] - val[0], D_RET = std::min((Real)1, W) * A_RET;
    // check whether importance weight is in 1/CmaxRet < c < CmaxRet
  const bool isOff = dropRule==1? false : S->isFarPolicy(t, W, CmaxRet,CinvRet);

  if( wID == 0 )
  {
    const Real dAdv = updateRetrace(S, t, 0, val[0], W);
    S->setMseDklImpw(t, D_RET*D_RET, P.sampKLdiv, W, CmaxRet, CinvRet);
    trainInfo->log(val[0], D_RET, { std::pow(dAdv,2), W }, thrID);
  }

  if(ESpopSize>1)
  {
    advs[bID][wID] = A_RET;
    dkls[bID][wID] = P.sampKLdiv;
    rhos[bID][wID] = P.sampImpWeight;
  }
  else
  {
    const Real BETA = dropRule==2? 1 : beta;
    assert(wID == 0);
    Rvec G = Rvec(F[0]->nOutputs(), 0);
    if(isOff)
    {
      #ifdef DACER_useAlpha
        P.finalize_grad(P.div_kl_grad(MU, alpha*(BETA-1)), G);
      #else //DACER_useAlpha
        P.finalize_grad(P.div_kl_grad(MU,        BETA-1 ), G);
      #endif //DACER_useAlpha
    }
    else
    {
      #ifdef DACER_useAlpha
        const Rvec G1 = P.policy_grad(P.sampAct, alpha * A_RET * W);
        const Rvec G2 = P.div_kl_grad(MU, -alpha);
      #else //DACER_useAlpha
        const Rvec G1 = P.policy_grad(P.sampAct, A_RET * W);
        const Rvec G2 = P.div_kl_grad(MU, -1);
      #endif //DACER_useAlpha
      P.finalize_grad(weightSum2Grads(G1, G2, BETA), G);
      trainInfo->trackPolicy(G1, G2, thrID);
      #ifdef DACER_singleNet
        assert(std::fabs(G[0])<1e-16); // make sure it was untouched
        #ifdef DACER_useAlpha
          G[0] = (1-alpha) * BETA * D_RET;
        #else
          G[0] = BETA * D_RET;
        #endif
      #endif
    }

    if(thrID==0) profiler->stop_start("BCK");
    #ifndef DACER_singleNet
      F[1]->backward( Rvec(1, D_RET), t, thrID);
      F[1]->gradient(thrID);  // backprop
    #endif
    F[0]->backward(G, t, thrID);
    F[0]->gradient(thrID);  // backprop
  }
}

}
