/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#include "../AllLearners.h"

void Gaussian_policy::test(const vector<Real>& act,
  const Gaussian_policy*const pol_hat) const //, const Quadratic_term*const a
{
  vector<Real> _grad(netOutputs.size());
   //const vector<Real> cntrolgrad = control_grad(a, -1);
   const vector<Real> div_klgrad = div_kl_grad(pol_hat);
   const vector<Real> policygrad = policy_grad(act, 1);
   for(Uint i = 0; i<2*nA; i++)
   {
     vector<Real> out_1 = netOutputs, out_2 = netOutputs;
     if(i>=nA && !start_prec) continue;
     const Uint index = i>=nA ? start_prec+i-nA : start_mean+i;
     out_1[index] -= 0.0001;
     out_2[index] += 0.0001;
     Gaussian_policy p1(vector<Uint>{start_mean, start_prec}, aInfo, out_1);
     Gaussian_policy p2(vector<Uint>{start_mean, start_prec}, aInfo, out_2);

    // Quadratic_advantage a1 =
    //  !a->start_mean ?
    //Quadratic_advantage(a->start_matrix,a->nA,a->nL,out_1,&p1)
    //  :
    //Quadratic_advantage(a->start_matrix,a->start_mean,a->nA,a->nL,out_1,&p1);

    //Quadratic_advantage a2 =
    //  !a->start_mean ?
    //Quadratic_advantage(a->start_matrix,a->nA,a->nL,out_2,&p2)
    //  :
    //Quadratic_advantage(a->start_matrix,a->start_mean,a->nA,a->nL,out_2,&p2);

    const Real p_1 = p1.evalLogProbability(act);
    const Real p_2 = p2.evalLogProbability(act);
    const Real d_1 = p1.kl_divergence(pol_hat);
    const Real d_2 = p2.kl_divergence(pol_hat);
    finalize_grad_unb(policygrad, _grad);
    if(fabs(_grad[index]-(p_2-p_1)/.0002)>1e-7)
      _die("LogPol var grad %d: finite differences %g analytic %g error %g \n",
       i,(p_2-p_1)/.0002,_grad[index],fabs(_grad[index]-(p_2-p_1)/.0002));
    printf("LogPol var grad %d: finite differences %g analytic %g error %g \n",
     i,(p_2-p_1)/.0002,_grad[index],fabs(_grad[index]-(p_2-p_1)/.0002));
    //#ifndef ACER_RELAX
    // const Real A_1 = a1.computeAdvantage(act);
    // const Real A_2 = a2.computeAdvantage(act);
    // finalize_grad_unb(cntrolgrad, _grad);
    //if(fabs(_grad[index]-(A_2-A_1)/.0002)>1e-7)
    //_die("Control var grad %d: finite differences %g analytic %g error %g \n",
    //  i,(A_2-A_1)/.0002,_grad[index],fabs(_grad[index]-(A_2-A_1)/.0002));
    //#endif

    finalize_grad_unb(div_klgrad, _grad);
    if(fabs(_grad[index]-(d_2-d_1)/.0002)>1e-7)
      _die("DivKL var grad %d: finite differences %g analytic %g error %g \n",
      i,(d_2-d_1)/.0002,_grad[index],fabs(_grad[index]-(d_2-d_1)/.0002));
    printf("DivKL var grad %d: finite differences %g analytic %g error %g \n",
    i,(d_2-d_1)/.0002,_grad[index],fabs(_grad[index]-(d_2-d_1)/.0002));
   }
}

void Discrete_policy::test(const Uint act, const Discrete_policy*const pol_hat) const
{
  vector<Real> _grad(netOutputs.size());
  //const vector<Real> cntrolgrad = control_grad(-1);
  const vector<Real> div_klgrad = div_kl_grad(pol_hat);
  const vector<Real> policygrad = policy_grad(act, 1);
  //values_grad(act, 1, _grad);
  for(Uint i = 0; i<nA; i++)
  {
    vector<Real> out_1 = netOutputs, out_2 = netOutputs;
    const Uint index = start_prob+i;
    out_1[index] -= 0.0001;
    out_2[index] += 0.0001;
    Discrete_policy p1(vector<Uint>{start_prob},aInfo,out_1);
    Discrete_policy p2(vector<Uint>{start_prob},aInfo,out_2);
    //const Real A_1 = p1.computeAdvantage(act);
    //const Real A_2 = p2.computeAdvantage(act);
    const Real p_1 = p1.evalLogProbability(act);
    const Real p_2 = p2.evalLogProbability(act);
    const Real d_1 = p1.kl_divergence(pol_hat);
    const Real d_2 = p2.kl_divergence(pol_hat);

    finalize_grad(div_klgrad, _grad);
    if(fabs(_grad[index]-(d_2-d_1)/.0002)>1e-7 && i>=nA) //not for values
     _die("DivKL grad %u %u: finite differences %g analytic %g error %g \n",
    i,act,(d_2-d_1)/.0002,_grad[index],fabs(_grad[index]-(d_2-d_1)/.0002));

    // finalize_grad(cntrolgrad, _grad);
    //if(fabs(_grad[index]-(A_2-A_1)/.0002)>1e-7)
    // _die("Control grad %u %u: finite differences %g analytic %g error %g \n",
    //i,act,(A_2-A_1)/.0002,_grad[index],fabs(_grad[index]-(A_2-A_1)/.0002));

    finalize_grad(policygrad, _grad);
    if(fabs(_grad[index]-(p_2-p_1)/.0002)>1e-7 && i>=nA) //not for values
    _die("LogPol grad %u %u: finite differences %g analytic %g error %g \n",
    i,act,(p_2-p_1)/.0002,_grad[index],fabs(_grad[index]-(p_2-p_1)/.0002));
  }
}

void Quadratic_advantage::test(const vector<Real>& act, mt19937*const gen) const
{
  vector<Real> _grad(netOutputs.size());
   grad_unb(act, 1, _grad);
   for(Uint i = 0; i<nL+nA; i++)
   {
     vector<Real> out_1 = netOutputs, out_2 = netOutputs;
     if(i>=nL && !start_mean) continue;
     const Uint index = i>=nL ? start_mean+i-nL : start_matrix+i;
     out_1[index] -= 0.0001;
     out_2[index] += 0.0001;

    Quadratic_advantage a1 = Quadratic_advantage(vector<Uint>{start_matrix, start_mean}, aInfo, out_1, policy);

    Quadratic_advantage a2 = Quadratic_advantage(vector<Uint>{start_matrix, start_mean}, aInfo, out_2, policy);

     const Real A_1 = a1.computeAdvantage(act);
     const Real A_2 = a2.computeAdvantage(act);
    if(fabs(_grad[index]-(A_2-A_1)/.0002)>1e-7)
     _die("Advantage grad %d: finite differences %g analytic %g error %g \n",
       i,(A_2-A_1)/.0002,_grad[index],fabs(_grad[index]-(A_2-A_1)/.0002));
   }
}

void Diagonal_advantage::test(const vector<Real>& act, mt19937*const gen) const
{
  vector<Real> _grad(netOutputs.size());
  grad(act, 1, _grad);
  for(Uint i = 0; i<4*nA; i++)
  {
    vector<Real> out_1 = netOutputs, out_2 = netOutputs;
    const Uint index = start_matrix+i;
    out_1[index] -= 0.0001;
    out_2[index] += 0.0001;

    Diagonal_advantage a1= Diagonal_advantage(vector<Uint>{start_matrix}, aInfo, out_1, policy);

    Diagonal_advantage a2= Diagonal_advantage(vector<Uint>{start_matrix}, aInfo, out_1, policy);

    const Real A_1 = a1.computeAdvantage(act);
    const Real A_2 = a2.computeAdvantage(act);
    if(fabs(_grad[index]-(A_2-A_1)/.0002)>1e-7)
     _die("Advantage grad %d: finite differences %g analytic %g error %g \n",
       i,(A_2-A_1)/.0002,_grad[index],fabs(_grad[index]-(A_2-A_1)/.0002));
    printf("Advantage grad %d: finite differences %g analytic %g error %g \n",
      i,(A_2-A_1)/.0002,_grad[index],fabs(_grad[index]-(A_2-A_1)/.0002));
  }
}

void NAF::test()
{
  vector<Real> out(nOutputs), act(nA);
  uniform_real_distribution<Real> out_dis(-.5,.5);
  uniform_real_distribution<Real> act_dis(-.5,.5);
  for(Uint i = 0; i<nA; i++) act[i] = act_dis(*gen);
  for(Uint i = 0; i<nOutputs; i++) out[i] = out_dis(*gen);
  Quadratic_advantage A = prepare_advantage(out);
  A.test(act, gen);
}

/*
void CACER::test()
{
  vector<Real> hat(nOutputs), out(nOutputs), act(nA);
  uniform_real_distribution<Real> out_dis(-.5,.5);
  uniform_real_distribution<Real> act_dis(-.5,.5);
  for(Uint i = 0; i<nOutputs; i++) out[i] = out_dis(*gen);
  for(Uint i = 0; i<nOutputs; i++) hat[i] = out_dis(*gen);
  for(Uint i = 0; i<nA; i++) act[i] = act_dis(*gen);
  Gaussian_policy pol_hat = prepare_policy(hat);
  Gaussian_policy pol_cur = prepare_policy(out);
  Quadratic_advantage adv = prepare_advantage(out, &pol_cur);
  pol_cur.test(act, &pol_hat); //,&adv
  adv.test(act);
}
*/

/*
void POAC::test()
{
  vector<Real> hat(nOutputs), out(nOutputs), act(nA);
  uniform_real_distribution<Real> out_dis(-.5,.5);
  uniform_real_distribution<Real> act_dis(-.5,.5);
  for(Uint i = 0; i<nOutputs; i++) out[i] = out_dis(*gen);
  for(Uint i = 0; i<nOutputs; i++) hat[i] = out_dis(*gen);
  for(Uint i = 0; i<nA; i++) act[i] = act_dis(*gen);
  Gaussian_policy pol_hat = prepare_policy(hat);
  Gaussian_policy pol_cur = prepare_policy(out);
  Advantage adv = prepare_advantage(out, &pol_cur);
  pol_cur.test(act, &pol_hat); //,&adv
  adv.test(act);
}
*/
/*
void DACER::test()
{
  uniform_real_distribution<Real> out_dis(-.5,.5);
  uniform_real_distribution<Real> act_dis(0, 1.);
  vector<Real> hat(nOutputs), out(nOutputs);
  const Uint act = act_dis(*gen)*nA;
  for(Uint i = 0; i<nOutputs; i++) out[i] = out_dis(*gen);
  for(Uint i = 0; i<nOutputs; i++) hat[i] = out_dis(*gen);
  Discrete_policy pol_hat = prepare_policy(hat);
  Discrete_policy pol_cur = prepare_policy(out);
  pol_cur.test(act, &pol_hat);
}
*/
