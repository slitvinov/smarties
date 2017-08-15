/*
 *  Layers.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include "Optimizer.h"
#include <iomanip>
#include <iostream>
#include <cassert>
#include "saruprng.h"

Optimizer::Optimizer(Network* const _net, Profiler* const _prof, Settings& _s) :
nWeights(_net->getnWeights()), nBiases(_net->getnBiases()), bTrain(_s.bTrain),
net(_net), profiler(_prof),
_1stMomW(initClean(nWeights)), _1stMomB(initClean(nBiases)),
eta(_s.learnrate), lambda(_s.nnLambda), epsAnneal(_s.epsAnneal) { }

AdamOptimizer::AdamOptimizer(Network*const _net, Profiler*const _prof,
    Settings& _s, const Real B1, const Real B2) : Optimizer(_net, _prof, _s),
    beta_1(B1), beta_2(B2), epsilon(1e-8), beta_t_1(B1), beta_t_2(B2),
    _2ndMomW(initClean(nWeights)), _2ndMomB(initClean(nBiases)) { }
//beta_1(0.9), beta_2(0.999), epsilon(1e-8), beta_t_1(0.9), beta_t_2(0.99)

EntropySGD::EntropySGD(Network*const _net, Profiler*const _prof, Settings&_s) :
    AdamOptimizer(_net, _prof, _s, 0.5), alpha_eSGD(0.75), gamma_eSGD(10.),
    eta_eSGD(.1/_s.targetDelay), eps_eSGD(1e-3), L_eSGD(_s.targetDelay),
    _muW_eSGD(initClean(nWeights)), _muB_eSGD(initClean(nBiases))
{
  assert(L_eSGD>0);
  for (Uint i=0; i<nWeights; i++) _muW_eSGD[i] = net->weights_back[i];
  for (Uint i=0; i<nBiases; i++)  _muB_eSGD[i] = net->biases[i];
}

void Optimizer::moveFrozenWeights(const Real _alpha)
{
  if (net->allocatedFrozenWeights==false || _alpha>=1)
    return net->updateFrozenWeights();

#pragma omp parallel
  {
#pragma omp for nowait
    for (Uint j=0; j<nWeights; j++)
      net->tgt_weights[j] += _alpha*(net->weights[j] - net->tgt_weights[j]);

#pragma omp for nowait
    for (Uint j=0; j<nBiases; j++)
      net->tgt_biases[j] += _alpha*(net->biases[j] - net->tgt_biases[j]);
  }
  net->sort_fwd_to_bck(net->tgt_weights, net->tgt_weights_back);
}

void EntropySGD::moveFrozenWeights(const Real _alpha)
{
  assert(_alpha>1);

#pragma omp parallel
  {
    const nnReal fac = eta_eSGD * gamma_eSGD;

#pragma omp for nowait
    for (Uint j=0; j<nWeights; j++) {
      net->tgt_weights_back[j] += fac*(_muW_eSGD[j] - net->tgt_weights_back[j]);
      net->weights_back[j] = net->tgt_weights_back[j];
      _muW_eSGD[j] = net->tgt_weights_back[j];
    }

#pragma omp for nowait
    for (Uint j=0; j<nBiases; j++){
      net->tgt_biases[j] += fac * (_muB_eSGD[j] - net->tgt_biases[j]);
      net->biases[j] = net->tgt_biases[j];
      _muB_eSGD[j] = net->tgt_biases[j];
    }
  }
  net->sort_bck_to_fwd(net->tgt_weights_back, net->tgt_weights);
  net->sort_bck_to_fwd(net->weights_back, net->weights);
}

void EntropySGD::update(Grads* const G, const Uint batchsize)
{
  //const Real _eta = eta/(1.+std::log(1. + (double)nepoch));
  update(net->weights_back,net->tgt_weights_back,G->_W,_1stMomW,_2ndMomW,_muW_eSGD,nWeights,batchsize,eta);
  update(net->biases, net->tgt_biases, G->_B,_1stMomB,_2ndMomB,_muB_eSGD,nBiases, batchsize,eta);

  beta_t_1 *= beta_1;
  if (beta_t_1<nnEPS) beta_t_1 = 0;

  beta_t_2 *= beta_2;
  if (beta_t_2<nnEPS) beta_t_2 = 0;

  if(lambda>nnEPS) net->regularize(lambda*eta);
  net->sortWeights_bck_to_fwd();
}

void Optimizer::stackGrads(Grads* const G, const Grads* const g) const
{
  for (Uint j=0; j<nWeights; j++) G->_W[j] += g->_W[j];
  for (Uint j=0; j<nBiases; j++)  G->_B[j] += g->_B[j];
}

void Optimizer::stackGrads(Grads* const G, const vector<Grads*> g) const
{
  const Uint nThreads = g.size();

#pragma omp parallel
  {
#pragma omp for nowait
    for (Uint j=0; j<nWeights; j++)
      for (Uint k=1; k<nThreads; k++) {
        G->_W[j] += g[k]->_W[j];
        g[k]->_W[j] = 0.;
      }

#pragma omp for nowait
    for (Uint j=0; j<nBiases; j++)
      for (Uint k=1; k<nThreads; k++) {
        G->_B[j] += g[k]->_B[j];
        g[k]->_B[j] = 0.;
      }
  }
}

void Optimizer::update(Grads* const G, const Uint batchsize)
{
  update(net->weights_back, G->_W, _1stMomW, nWeights, batchsize);
  update(net->biases,       G->_B, _1stMomB, nBiases,  batchsize);
  if(lambda>nnEPS) net->regularize(lambda*eta);
  net->sortWeights_bck_to_fwd();
}

void AdamOptimizer::update(Grads* const G, const Uint batchsize)
{
  const Real _eta = eta*std::max(.1, 1-nepoch/epsAnneal);

  update(net->weights_back,G->_W,_1stMomW,_2ndMomW,nWeights,batchsize,_eta);
  update(net->biases,      G->_B,_1stMomB,_2ndMomB,nBiases, batchsize,_eta);

  beta_t_1 *= beta_1;
  if (beta_t_1<nnEPS) beta_t_1 = 0;
  beta_t_2 *= beta_2;
  if (beta_t_2<nnEPS) beta_t_2 = 0;

  if(lambda>nnEPS) net->regularize(lambda*_eta);
  net->sortWeights_bck_to_fwd();
}

void Optimizer::update(nnOpRet dest, nnOpRet grad, nnOpRet _1stMom, const Uint N, const Uint batchsize) const
{
  assert(batchsize>0);
  const nnReal norm = 1./batchsize;
  //const Real eta_ = eta*norm/std::log((double)nepoch/1.);
  const nnReal eta_ = eta*norm/(1.+std::log(1. + (double)nepoch/1e3));

#pragma omp parallel for
  for (Uint i=0; i<N; i++) {
    const nnReal M1 = alpha * _1stMom[i] + eta_ * grad[i];
    _1stMom[i] = std::max(std::min(M1,eta_),-eta_);
    grad[i] = 0.; //reset grads
    dest[i] += _1stMom[i];
  }
}

void EntropySGD::update(nnOpRet dest,const nnOpRet target, nnOpRet grad, nnOpRet _1stMom, nnOpRet _2ndMom, nnOpRet _mu, const Uint N, const Uint batchsize, const Real _eta) const
{
  //const Real fac_ = std::sqrt(1.-beta_t_2)/(1.-beta_t_1);
  assert(batchsize>0);

#pragma omp parallel
  {
    const Uint thrID = static_cast<Uint>(omp_get_thread_num());
    Saru gen(nepoch, thrID, net->generators[thrID]());
    const nnReal eta_ = _eta*std::sqrt(beta_2-beta_t_2)/(1.-beta_t_1);
    const nnReal norm = 1./batchsize, noise = std::sqrt(eta_) * eps_eSGD;
    const nnReal f11=beta_1, f12=1-beta_1, f21=beta_2;

#pragma omp for
    for (Uint i=0; i<N; i++)
    {
      const nnReal DW  = grad[i]*norm;
      const nnReal M1  = f11* _1stMom[i] +f12* DW;
      const nnReal M2  = std::max(f21*_2ndMom[i], std::fabs(DW));
      //const nnReal M2  = f21* _2ndMom[i] +f22* DW*DW;
      const nnReal M2_ = std::max(M2, nnEPS);
      const nnReal _M2 = std::sqrt(M2_);
      const nnReal M1_ = std::max(std::min(M1, _M2), -_M2); //grad clip

      const nnReal RNG = noise * gen.d_mean0_var1();
      const nnReal DW_ = eta_*(f12*DW + f11*M1_)/_M2; //Nesterov Adam

      _1stMom[i] = M1_;
      _2ndMom[i] = M2_;
      grad[i] = 0.; //reset grads
      assert(!std::isnan(DW));
      assert(!std::isnan(dest[i]));

      dest[i] += DW_ + RNG + eta_*gamma_eSGD*(target[i]-dest[i]);
      _mu[i]  += alpha_eSGD*(dest[i] - _mu[i]);
    }
  }
}

#if 1
void AdamOptimizer::update(nnOpRet dest, nnOpRet grad, nnOpRet _1stMom, nnOpRet _2ndMom, const Uint N, const Uint batchsize, const Real _eta) const
{
  assert(batchsize>0);
  #pragma omp parallel
  {
    const nnReal eta_ = _eta*std::sqrt(1.-beta_t_2)/(1.-beta_t_1);
    const nnReal norm = 1./batchsize;
    const nnReal f11=beta_1, f12=1-beta_1, f21=beta_2, f22=1-beta_2;

    #pragma omp for
    for (Uint i=0; i<N; i++) {
      const nnReal DW  = grad[i]*norm;
      const nnReal M1  = f11* _1stMom[i] +f12* DW;
      const nnReal M2  = f21* _2ndMom[i] +f22* DW*DW;
      const nnReal M2_ = std::max(M2, nnEPS);
      const nnReal _M2 = std::sqrt(M2_);
      //this line makes address-sanitizer cry: i have no clue why.
      //const nnReal M1 = std::max(std::min(M1, _M2), -_M2); //grad clip
      //this is fine tho: (I DON'T GET COMPUTERS)
      //const nnReal tmp = M1 >  _M2 ?  _M2 : M1;
      //const nnReal M1_ = M1 < -_M2 ? -_M2 : tmp;
      //printf("batch %u %f %f %f\n",batchsize,DW,M1,M2); fflush(0);
      const nnReal M1_ = M1;
      _1stMom[i] = M1_;
      _2ndMom[i] = M2_;
      grad[i] = 0.; //reset grads
      assert(!std::isnan(DW));
      assert(!std::isnan(dest[i]));
      //if(DW*M1_>0)
      //dest[i] += eta_*M1_/_M2; //Adam
      //else
      dest[i] += eta_*(f12*DW + f11*M1_)/_M2; //Nesterov Adam
    }
  }
}
#else // Adamax:
void AdamOptimizer::update(nnOpRet dest, nnOpRet grad, nnOpRet _1stMom, nnOpRet _2ndMom, const Uint N, const Uint batchsize, const Real _eta) const
{
  assert(batchsize>0);
  const nnReal eta_ = _eta*std::sqrt(beta_2-beta_t_2)/(1.-beta_t_1);
  const nnReal norm = 1./batchsize;
  const nnReal f11=beta_1, f12=1-beta_1, f21=beta_2;

#pragma omp parallel for
  for (Uint i=0; i<N; i++) {
    const nnReal DW  = grad[i]*norm;
    const nnReal M1  = f11*_1stMom[i] +f12*DW;
    const nnReal M2  = std::max(f21*_2ndMom[i], std::fabs(DW));
    const nnReal M2_ = std::max(M2, nnEPS);
    const nnReal M1_ = M1;
    //const nnReal M1_ = std::max(std::min(M1, M2_), -M2_);
    //dest[i] += eta_*M1_/M2_;
    dest[i] += eta_*(f12*DW + f11*M1_)/M2_; //nesterov
    _1stMom[i] = M1_;
    _2ndMom[i] = M2_;
    grad[i] = 0.; //reset grads
  }
}
#endif

void Optimizer::save(const string fname)
{
  const Uint nNeurons(net->getnNeurons()), nLayers(net->getnLayers());
  //const Uint nAgents(net->getnAgents()), nStates(net->getnStates());
  printf("Saving into %s\n", fname.c_str()); fflush(0);

  vector<nnReal> outWeights, outBiases, outMomW, outMomB;
  outWeights.reserve(nWeights); outMomW.reserve(nWeights);
  outBiases.reserve(nBiases); outMomB.reserve(nBiases);

  net->save(outWeights, outBiases, net->weights, net->biases);
  net->save(outMomW, outMomB, _1stMomW, _1stMomB);
  const Uint NW = static_cast<Uint>(outWeights.size());
  const Uint NB = static_cast<Uint>(outBiases.size());
  assert(outWeights.size() == outMomW.size());
  assert(outBiases.size() == outMomB.size());

  FILE * pFile = fopen ((fname+"_net_tmp.raw").c_str(), "ab");
  Uint buf[] = {NW, NB, nLayers, nNeurons};
  fwrite (buf,                 sizeof(Uint),    4,                pFile);
  fwrite (outWeights.data(),   sizeof(nnReal), outMomW.size(),   pFile);
  fwrite (outBiases.data(),   sizeof(nnReal), outMomB.size(),   pFile);
  fwrite (outMomW.data(),     sizeof(nnReal), outMomW.size(),   pFile);
  fwrite (outMomB.data(),     sizeof(nnReal), outMomB.size(),   pFile);
  fflush(pFile);
  fclose(pFile);
  /*
    ofstream out((fname+"_net_tmp").c_str());
    if (!out.good()) _die("Unable to open save into file %s\n", fname.c_str());
    out.precision(20);
    out<<outWeights.size()<<" "<<outBiases.size()<<" "<<nLayers<<" "<<nNeurons<<endl;
    for(Uint i=0;i<outMomW.size();i++)out<<outWeights[i]<<" "<<outMomW[i]<<"\n";
    for(Uint i=0;i<outMomB.size();i++)out<<outBiases[i] <<" "<<outMomB[i]<<"\n";
    out.flush();
    out.close();
  */
  string command = "cp " + fname+"_net_tmp.raw " + fname+"_net.raw";
  system(command.c_str());

  save_recurrent_connections(fname);
}

bool EntropySGD::restart(const string fname)
{
  const bool ret = AdamOptimizer::restart(fname);
  if (!ret) return ret;
  for (Uint i=0; i<nWeights; i++) _muW_eSGD[i] = net->weights_back[i];
  for (Uint i=0; i<nBiases; i++)  _muB_eSGD[i] = net->biases[i];
  return ret;
}

void AdamOptimizer::save(const string fname)
{
  const Uint nNeurons(net->getnNeurons()), nLayers(net->getnLayers());
  //const Uint nAgents(net->getnAgents()), nStates(net->getnStates());
  printf("Saving into %s\n", fname.c_str()); fflush(0);

  vector<nnReal> outWeights, outBiases, out1MomW, out1MomB, out2MomW, out2MomB;
  outWeights.reserve(nWeights);  outBiases.reserve(nBiases);
  out1MomW.reserve(nWeights);    out1MomB.reserve(nBiases);
  out2MomW.reserve(nWeights);    out2MomB.reserve(nBiases);

  net->save(outWeights, outBiases, net->weights, net->biases);
  net->save(out1MomW,   out1MomB,  _1stMomW,     _1stMomB);
  net->save(out2MomW,   out2MomB,  _2ndMomW,     _2ndMomB);
  const Uint NW = static_cast<Uint>(outWeights.size());
  const Uint NB = static_cast<Uint>(outBiases.size());
  assert(outWeights.size()==out1MomW.size()&&outBiases.size()==out1MomB.size());
  assert(outWeights.size()==out2MomW.size()&&outBiases.size()==out2MomB.size());

  FILE * pFile = fopen ((fname+"_net_tmp.raw").c_str(), "wb");
  Uint buf[] = {NW, NB, nLayers, nNeurons};
  fwrite (buf,               sizeof( Uint ),               4, pFile);
  fwrite (outWeights.data(), sizeof(nnReal), out1MomW.size(), pFile);
  fwrite (outBiases.data(),  sizeof(nnReal), out1MomB.size(), pFile);
  fwrite (out1MomW.data(),   sizeof(nnReal), out1MomW.size(), pFile);
  fwrite (out1MomB.data(),   sizeof(nnReal), out1MomB.size(), pFile);
  fwrite (out2MomW.data(),   sizeof(nnReal), out1MomW.size(), pFile);
  fwrite (out2MomB.data(),   sizeof(nnReal), out1MomB.size(), pFile);
  fflush(pFile);
  fclose(pFile);

  /*
    ofstream out(nameBackup.c_str());
    if (!out.good()) _die("Unable to open save into file %s\n", fname.c_str());
    out.precision(20);
    out<<outWeights.size()<<" "<<outBiases.size()<<" "<<nLayers<<" "<<nNeurons<<endl;
    for(Uint i=0;i<outWeights.size();i++)
      out<<outWeights[i]<<" "<<out1MomW[i]<<" "<<out2MomW[i]<<"\n";
    for(Uint i=0;i<outBiases.size();i++)
      out<<outBiases[i] <<" "<<out1MomB[i]<<" "<<out2MomB[i]<<"\n";
    out.flush();
    out.close();
  */
  string command = "cp " + fname+"_net_tmp.raw " + fname+"_net.raw";
  system(command.c_str());

  save_recurrent_connections(fname);
}

bool Optimizer::restart(const string fname)
{
  const Uint nNeurons(net->getnNeurons()), nLayers(net->getnLayers());
  vector<nnReal> outWeights, outBiases, outMomW, outMomB;
  //const Uint nAgents(net->getnAgents()); // , nStates(net->getnStates()) TODO

  ifstream in((fname + "_net").c_str());
  if (in.good())
  {
    Uint readTotWeights, readTotBiases, readNNeurons, readNLayers;
    in >> readTotWeights  >> readTotBiases >> readNLayers >> readNNeurons;
    if (readNLayers != nLayers || readNNeurons != nNeurons)
      die("Network parameters differ!");
    //readTotWeights != nWeights || readTotBiases != nBiases || TODO
    outWeights.resize(readTotWeights); outBiases.resize(readTotBiases);
    outMomW.resize(readTotWeights);    outMomB.resize(readTotBiases);
    for (Uint i=0;i<readTotWeights;i++) in>>outWeights[i]>>outMomW[i];
    for (Uint i=0;i<readTotBiases; i++) in>>outBiases[i] >>outMomB[i];
    net->restart(outWeights, outBiases, net->weights, net->biases);
    net->restart(outMomW, outMomB, _1stMomW, _1stMomB);
    in.close();
    net->updateFrozenWeights();
    net->sortWeights_fwd_to_bck();
    return restart_recurrent_connections(fname);
  }

  FILE * pFile = fopen ((fname + "_net.raw").c_str(), "rb");
  if (pFile != NULL)
  {
    Uint buf[4];
    size_t ret = fread(buf, sizeof(Uint), 4, pFile); assert(ret == 4);
    if(buf[2]!=nLayers || buf[3]!=nNeurons) die("Network parameters differ!");
    outWeights.resize(buf[0]); outBiases.resize(buf[1]);
    outMomW.resize(buf[0]);    outMomB.resize(buf[1]);
    ret = fread (outWeights.data(),  sizeof(nnReal), outMomW.size(),   pFile);
    if(ret!=outMomB.size()) die("ERROR: Adam::restart W0.\n");
    ret = fread (outBiases.data(),   sizeof(nnReal), outMomB.size(),   pFile);
    if(ret!=outMomB.size()) die("ERROR: Adam::restart B0.\n");
    ret = fread (outMomW.data(),     sizeof(nnReal), outMomW.size(),   pFile);
    if(ret!=outMomB.size()) die("ERROR: Adam::restart W1.\n");
    ret = fread (outMomB.data(),     sizeof(nnReal), outMomB.size(),   pFile);
    if(ret!=outMomB.size()) die("ERROR: Adam::restart B1.\n");
    net->restart(outWeights, outBiases, net->weights, net->biases);
    net->restart(outMomW, outMomB, _1stMomW, _1stMomB);
    fclose(pFile);
    net->updateFrozenWeights();
    net->sortWeights_fwd_to_bck();
    return restart_recurrent_connections(fname);
  }

  error("Couldnt open policy (%s) file", fname.c_str());
  #ifndef NDEBUG //if debug, you might want to do this
    if(!bTrain) {die("...and I'm not training\n");}
  #endif
  return false;
}

bool AdamOptimizer::restart(const string fname)
{
  const Uint nNeurons(net->getnNeurons()), nLayers(net->getnLayers());
  vector<nnReal> outWeights, outBiases, out1MomW, out1MomB, out2MomW, out2MomB;
  //const Uint nAgents(net->getnAgents()); // , nStates(net->getnStates()) TODO

  ifstream in((fname + "_net").c_str());
  if (in.good())
  {
    Uint readTotWeights, readTotBiases, readNNeurons, readNLayers;
    in >> readTotWeights  >> readTotBiases >> readNLayers >> readNNeurons;
    if (readNLayers != nLayers || readNNeurons != nNeurons)
      die("Network parameters differ!");
    //readTotWeights != nWeights || readTotBiases != nBiases || TODO
    outWeights.resize(readTotWeights);  outBiases.resize(readTotBiases);
    out1MomW.resize(readTotWeights);    out1MomB.resize(readTotBiases);
    out2MomW.resize(readTotWeights);    out2MomB.resize(readTotBiases);
    for (Uint i=0;i<readTotWeights;i++)
      in >> outWeights[i] >> out1MomW[i] >> out2MomW[i];
    for (Uint i=0;i<readTotBiases; i++)
      in >> outBiases[i]  >> out1MomB[i] >> out2MomB[i];
    net->restart(outWeights, outBiases, net->weights, net->biases);
    net->restart(out1MomW, out1MomB, _1stMomW, _1stMomB);
    net->restart(out2MomW, out2MomB, _2ndMomW, _2ndMomB);
    in.close();
    net->updateFrozenWeights();
    net->sortWeights_fwd_to_bck();
    return restart_recurrent_connections(fname);
  }

  FILE * pFile = fopen ((fname + "_net.raw").c_str(), "rb");
  if (pFile != NULL)
  {
    Uint buf[4];
    size_t ret = fread(buf, sizeof(Uint), 4, pFile); assert(ret == 4);
    if(buf[2]!=nLayers || buf[3]!=nNeurons) die("Network parameters differ!");
    outWeights.resize(buf[0]);  outBiases.resize(buf[1]);
    out1MomW.resize(buf[0]);    out1MomB.resize(buf[1]);
    out2MomW.resize(buf[0]);    out2MomB.resize(buf[1]);
    ret = fread(outWeights.data(),  sizeof(nnReal), out1MomW.size(), pFile);
    if(ret!=out1MomW.size()) die("Adam::restart W0.\n");
    ret = fread(outBiases.data(),    sizeof(nnReal), out1MomB.size(), pFile);
    if(ret!=out1MomB.size()) die("Adam::restart B0.\n");
    ret = fread(out1MomW.data(),    sizeof(nnReal), out1MomW.size(), pFile);
    if(ret!=out1MomW.size()) die("Adam::restart W1.\n");
    ret = fread(out1MomB.data(),    sizeof(nnReal), out1MomB.size(), pFile);
    if(ret!=out1MomB.size()) die("Adam::restart B1.\n");
    ret = fread(out2MomW.data(),    sizeof(nnReal), out1MomW.size(), pFile);
    if(ret!=out1MomW.size()) die("Adam::restart W2.\n");
    ret = fread(out2MomB.data(),    sizeof(nnReal), out1MomB.size(), pFile);
    if(ret!=out1MomB.size()) die("Adam::restart B2.\n");
    net->restart(outWeights, outBiases, net->weights, net->biases);
    net->restart(out1MomW, out1MomB, _1stMomW, _1stMomB);
    net->restart(out2MomW, out2MomB, _2ndMomW, _2ndMomB);
    fclose(pFile);
    net->updateFrozenWeights();
    net->sortWeights_fwd_to_bck();
    return restart_recurrent_connections(fname);
  }

  error("Couldnt open policy (%s) file", fname.c_str());
  #ifndef NDEBUG //if debug, you might want to do this
    if(!bTrain) {die("...and I'm not training");}
  #endif
  return false;
}
