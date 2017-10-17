/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Links.h"
#include "Graph.h"
#include <iostream>

class NormalLink: public Link
{
 public:
  /*
     a link here is defined as link layer to layer:
     index iI along the network activation outvals representing the index of the first neuron of input layer
     the number nI of neurons of the input layer
     the index iO of the first neuron of the output layer
     the number of neurons in the output layer nO
     the index of the first weight iW along the weight vector
     the weights are all to all: so this link occupies space iW to (iW + nI*nO) along weight vector
   */
  NormalLink(Uint _nI, Uint _iI, Uint _nO, Uint _iO, Uint _iW, Uint _nO_simd, Uint _nI_simd) :
    Link(_nI, _iI, _nO, _iO, _iW, _nO_simd, _nI_simd, _nI_simd*_nO_simd)
  {
    assert(iW % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(iI % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(iO % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(nO_simd % (VEC_WIDTH/sizeof(nnReal)) == 0);
    print();
    assert(nI>0 && nO>0);
  }

  void print() const
  {
    const string fname = "network_build.log";
    FILE * f = fopen(fname.c_str(), "a");
    if (f == NULL) die("Save fail\n");
    fprintf(f,"Normal link: nInputs:%d IDinput:%d nOutputs:%d IDoutput:%d IDweight:%d nWeights:%d nO_simd:%d nI_simd:%d\n",
      nI,iI,nO,iO,iW,nW,nO_simd,nI_simd);
    fflush(f); fclose(f);
  }

  void save(vector<nnReal> & out, nnOpRet _weights) const override
  {
    _save(out, _weights, iW, nO, nI, nO_simd);
  }

  void restart(vector<nnReal> & buf, nnOpRet _weights) const override
  {
    _restart(buf, _weights, iW, nO, nI, nO_simd);
  }

  void initialize(mt19937*const gen, nnOpRet _weights, const Function*const func, const Real fac) const
  {
    const Real init = func->weightsInitFactor(nI, nO)*fac;
    _initialize(gen, _weights, init, iW, nO, nI, nO_simd);
  }

  //Links are from specific layer to specific layer:
  // propagate inp_i = sum_i,j w_i,j out_j ( Layer.h does out_i = f(inp_i) )
  // netFrom is network state in which out_j of input layer are stored
  // netTo is network state in which inp_i of output layer are to be computed
  inline void propagate(const Activation*const netFrom, Activation*const netTo, nnOpInp weights) const
  {
    nnOpInp inp = netFrom->outvals +iI;
    nnOpRet out = netTo->in_vals +iO;

    for (Uint i = 0; i < nI; i++) {
      nnOpInp w = weights +iW +nO_simd*i;
      #pragma omp simd aligned(inp,out,w : VEC_WIDTH) safelen(VEC_WIDTH)
      for (Uint o = 0; o < nO; o++) out[o] += inp[i] * w[o];
    }
  }

  void orthogonalize(nnOpRet _weights, nnOpInp _biases, const Uint firstBias) const override
  {
    nnOpRet w = _weights +iW; nnOpInp b = _biases +firstBias;
    for (Uint i=1; i<nO; i++) {
      for (Uint j=0; j<i; j++) {
        nnReal u_d_u = 0.0, v_d_u = 0.0;
        for (Uint k=0; k<nI; k++) {
          u_d_u += w[j*nI_simd +k] * w[j*nI_simd +k];
          v_d_u += w[j*nI_simd +k] * w[i*nI_simd +k];
        }
        if( v_d_u < 0) continue;
        const nnReal fac = v_d_u/u_d_u * nnSafeExp(-100*std::pow(b[i]-b[j],2));
        for (Uint k=0; k<nI; k++) w[i*nI_simd +k] -= fac * w[j*nI_simd +k];
      }
    }
  }

  //input: weights sorted for fast back prop, ret: weights for fast fwd prop
  inline void sortWeights_bck_to_fwd(nnOpInp w_bck, nnOpRet w_fwd) const override
  {
    #pragma omp parallel for collapse(2)
    for (Uint i = 0; i < nI; i++)
    for (Uint o = 0; o < nO; o++)
      w_fwd[iW +nO_simd*i +o] = w_bck[iW +nI_simd*o +i];
  }

  //input: weights sorted for fast fwd prop, ret: weights for fast back prop
  inline void sortWeights_fwd_to_bck(nnOpInp w_fwd, nnOpRet w_bck) const override
  {
    #pragma omp parallel for collapse(2)
    for (Uint i = 0; i < nI; i++)
    for (Uint o = 0; o < nO; o++)
      w_bck[iW +nI_simd*o +i] = w_fwd[iW +nO_simd*i +o];
  }

  // assumes that layer.h already computed d Error/d inp_i
  inline void backPropagate(Activation*const netFrom, const Activation*const netTo, nnOpInp weights, nnOpRet gradW) const
  {
    nnOpInp inp = netFrom->outvals + iI;
    //nnOpInp out = netTo->in_vals +iO;
    nnOpInp delta = netTo->errvals + iO; //contains d Error / d inp_i
    nnOpRet err = netFrom->errvals + iI; //to compute: d Error / d out_j
    //const Real lambda = 1e-6;
    for (Uint o = 0; o < nO; o++) {
      nnOpInp w = weights +iW +nI_simd*o;
      nnOpRet g = gradW +iW +nI_simd*o;
      #pragma omp simd aligned(g,inp,delta,err,w : VEC_WIDTH) safelen(VEC_WIDTH)
      for (Uint i = 0; i < nI; i++) {
        g[i] += inp[i] * delta[o];// - lambda*inp[i]*out[o];
        err[i] += delta[o] * w[i];
      }
    }
  }

  /*
  static void build(Graph* const graph, Builder*const build)
  {
    assert(graph->written == true && graph->built == false);
    assert(build->nNeurons%simdWidth==0 && build->nNeurons>0);
    assert(build->nWeights%simdWidth==0);
    assert(build->nBiases%simdWidth==0);
    vector<NormalLink*> input_links;
    NormalLink* recurrent_link = nullptr;

    graph->layerSize_simd = roundUpSimd(graph->layerSize);
    assert(graph->layerSize>0 && graph->layerSize_simd>=graph->layerSize);
    assert(graph->linkedTo.size()>0);

    graph->firstNeuron_ID = build->nNeurons;
    build->nNeurons += graph->layerSize_simd; //move the counter
    graph->firstBias_ID = build->nBiases;
    build->nBiases += graph->layerSize_simd; //one bias per neuron

    for(Uint i = 0; i<graph->linkedTo.size(); i++)
    {
      const Graph* const layerFrom = build->G[graph->linkedTo[i]];
      assert(layerFrom->firstNeuron_ID < graph->firstNeuron_ID);
      assert(layerFrom->written && layerFrom->built);
      assert(layerFrom->layerSize>0);

      NormalLink* tmp = new NormalLink(
          layerFrom->layerSize, layerFrom->firstNeuron_ID,
          graph->layerSize, graph->firstNeuron_ID,
          build->nWeights, graph->layerSize_simd, layerFrom->layerSize_simd
      );

      input_links.push_back(tmp);
      graph->links.push_back(tmp);
      build->nWeights += layerFrom->layerSize_simd*graph->layerSize_simd;
    }

    if (graph->RNN) { //connected  to past realization of current normal layer
      recurrent_link = new NormalLink(
          graph->layerSize, graph->firstNeuron_ID,
          graph->layerSize, graph->firstNeuron_ID,
          build->nWeights, graph->layerSize_simd, graph->layerSize_simd
      );
      graph->links.push_back(recurrent_link);
      build->nWeights += graph->layerSize_simd * graph->layerSize_simd;
    }

    Layer * l = new NormalLayer(
        graph->layerSize, graph->firstNeuron_ID,
        graph->firstBias_ID,
        input_links, recurrent_link,
        graph->func, graph->layerSize_simd
    );

    build->layers.push_back(l);
    graph->built = true;
  }

  static void stack(const int size, const string layerType, const string funcType, vector<int> linkedTo, const bool bOutput, vector<Graph*>& G)
  {
    Graph * g = new Graph(nLayers++);

    assert(!layerType.empty());
    if (layerType == "RNN")  g->RNN = true; //non-LSTM recurrent neural network
    else if (layerType == "LSTM") g->LSTM = true;
    else if (layerType == "IntegrateFire") g->IntegrateFire = true;

    assert(!funcType.empty());
    if(g->LSTM) {
      g->cell = new Linear(); //in original paper is Tanh, but is unnecessary
      g->gate = new SoftSigm(); //in original paper is Sigm (Sigmoid)
    }                            //g->func is TwoTanh (2*Tanh)
    g->func = readFunction(funcType, bOutput);

    g->layerSize = size;
    if(!G.size()) die("Proposed link not available: graph empty\n");
    const Uint nPrevLayers = G.size();
    //default link is to previous layer:
    if(linkedTo.size() == 0)
      linkedTo.push_back(static_cast<int>(nPrevLayers)-1);

    const Uint inputLinks = linkedTo.size();
    assert(g->linkedTo.size() == 0);

    for(Uint i = 0; i<inputLinks; i++) {
      g->linkedTo.push_back(static_cast<Uint>(linkedTo[i]));
      if(g->linkedTo.back() >= nPrevLayers)
        die("Proposed link not available\n");
    }

    g->written = true;
    G.push_back(g);
  }
  */
};
