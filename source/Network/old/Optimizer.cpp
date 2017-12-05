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

Optimizer::Optimizer(Network*const _net, Profiler*const _prof, Settings&_s,
  const Real B1) : nWeights(_net->getnWeights()), nBiases(_net->getnBiases()),
  bTrain(_s.bTrain), net(_net), profiler(_prof), _1stMomW(initClean(nWeights)),
  _1stMomB(initClean(nBiases)), eta(_s.learnrate), beta_1(B1),
  lambda(_s.nnLambda), epsAnneal(_s.epsAnneal) { }

AdamOptimizer::AdamOptimizer(Network*const _net, Profiler*const _prof,
  Settings& _s, const Real B1, const Real B2) : Optimizer(_net, _prof, _s, B1),
  beta_2(B2), epsilon(1e-8), beta_t_1(B1), beta_t_2(B2),
  _2ndMomW(initClean(nWeights)), _2ndMomB(initClean(nBiases)) { }
//beta_1(0.9), beta_2(0.999), epsilon(1e-8), beta_t_1(0.9), beta_t_2(0.99)

EntropySGD::EntropySGD(Network*const _net, Profiler*const _prof, Settings&_s) :
  AdamOptimizer(_net, _prof, _s),
{
  //assert(L_eSGD>0);
  for (Uint i=0; i<nWeights; i++) _muW_eSGD[i] = net->weights_back[i];
  for (Uint i=0; i<nBiases; i++)  _muB_eSGD[i] = net->biases[i];
}

void Optimizer::moveFrozenWeights(const Real _alpha)
{

  net->sort_fwd_to_bck(net->tgt_weights, net->tgt_weights_back);
}

#if 0
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
