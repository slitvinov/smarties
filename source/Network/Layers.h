/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "../Settings.h"
#include "Links.h"
//#include <cblas.h>
#include <iostream>

class Layer
{
    public:
    const int nNeurons, n1stNeuron, n1stBias, nNeurons_simd;
    const bool bOutput;
    virtual ~Layer() {}
	  Layer(const int _nNeurons, const int _n1stNeuron,
      const int _n1stBias, const int nn_simd, const bool bOut) :
      nNeurons(_nNeurons), n1stNeuron(_n1stNeuron),
      n1stBias(_n1stBias), nNeurons_simd(nn_simd), bOutput(bOut) {}

    virtual void propagate(const Activation* const prev, Activation* const curr,
                           const Real* const weights, const Real* const biases) const = 0;
    virtual void backPropagate( Activation* const prev,  Activation* const curr,
                                const Activation* const next, Grads* const grad,
                                const Real* const weights,
                                const Real* const biases) const = 0;

    virtual void initialize(mt19937* const gen, Real* const weights, Real* const biases) const = 0;

    virtual void save(std::ostringstream & outWeights, std::ostringstream & outBiases,
  			Real* const _weights, Real* const _biases) const = 0;
    virtual void restart(std::istringstream & bufWeights, std::istringstream & bufBiases,
  			Real* const _weights, Real* const _biases) const = 0;

    void propagate(Activation* const curr, const Real* const weights, const Real* const biases) const
        { return propagate(nullptr, curr, weights, biases); }
    void backPropagate( Activation* const curr, Grads* const grad, const Real* const weights, const Real* const biases) const
        { return backPropagate(nullptr, curr, nullptr, grad, weights, biases); }

    virtual void regularize(Real* const weights, Real* const biases, const Real lambda) const = 0;
};

class NormalLayer: public Layer
{
    const vector<NormalLink*>* const input_links;
    const NormalLink* const recurrent_link;
    const Function* const func;

  public:
    NormalLayer(int _nNeurons, int _n1stNeuron, int _n1stBias,
      const vector<NormalLink*>* const nl_il, const NormalLink* const nl_rl,
      const Function* const f, const int nn_simd, const bool bOut = false) :
    Layer(_nNeurons, _n1stNeuron, _n1stBias, nn_simd, bOut),
    input_links(nl_il), recurrent_link(nl_rl), func(f)
    {
	   printf("Normal Layer of size %d, with first ID %d and first bias ID %d\n",
     nNeurons, n1stNeuron, n1stBias);
    }

    void initialize(mt19937* const gen, Real* const weights, Real* const biases) const override
    {
      uniform_real_distribution<Real> dis(-sqrt(6./nNeurons),sqrt(6./nNeurons));

      for (const auto & link : *input_links)
        if(link not_eq nullptr) link->initialize(gen,weights);

      if(recurrent_link not_eq nullptr)
        recurrent_link->initialize(gen,weights);

      for (int w=n1stBias; w<n1stBias+nNeurons_simd; w++)
				biases[w] = dis(*gen);
    }

    void save(std::ostringstream & outWeights, std::ostringstream & outBiases,
  			Real* const _weights, Real* const _biases) const
  	{
  		for (const auto & l : *input_links)
  			if(l not_eq nullptr) l->save(outWeights, _weights);

      if(recurrent_link not_eq nullptr)
        recurrent_link->save(outWeights, _weights);

  		for (int w=n1stBias; w<n1stBias+nNeurons; w++)
  			outBiases << _biases[w] << "\n";
  	}

    void restart(std::istringstream & bufWeights, std::istringstream & bufBiases,
  			Real* const _weights, Real* const _biases) const
  	{
      for (const auto & l : *input_links)
  			if(l not_eq nullptr) l->restart(bufWeights, _weights);

      if(recurrent_link not_eq nullptr)
        recurrent_link->restart(bufWeights, _weights);

  		Real tmp;
  		for (int w=n1stBias; w<n1stBias+nNeurons; w++) {
  			bufBiases >> tmp;
  			assert(not std::isnan(tmp) & not std::isinf(tmp));
  			_biases[w] = tmp;
  		}
  	}

    void propagate(const Activation* const prev, Activation* const curr,
                   const Real* const weights, const Real* const biases) const override
    {
        Real* __restrict__ const outputs = curr->outvals +n1stNeuron;
        Real* __restrict__ const inputs = curr->in_vals +n1stNeuron;
        const Real* __restrict__ const bias = biases +n1stBias;
        //__builtin_assume_aligned(outputs,  __vec_width__);
        //__builtin_assume_aligned(inputs, __vec_width__);
        //__builtin_assume_aligned(bias, __vec_width__);

        for (int n=0; n<nNeurons; n++) inputs[n] = bias[n];

        for (const auto & link : *input_links)
            link->propagate(curr,curr,weights);
            /*
        	cblas_dgemv(CblasRowMajor, CblasTrans, link->nI, nNeurons_simd,
        				1.0, weights  + link->iW, nNeurons_simd,
						curr->outvals + link->iI, 1,
						1.0, inputs, 1);
            */
        if(recurrent_link not_eq nullptr && prev not_eq nullptr)
            recurrent_link->propagate(prev,curr,weights);
            /*
        	cblas_dgemv(CblasRowMajor, CblasTrans, nNeurons, nNeurons_simd,
        				1.0, weights  +recurrent_link->iW, nNeurons_simd,
						prev->outvals +n1stNeuron, 1,
						1.0, inputs, 1);
            */
        for (int n=0; n<nNeurons; n++) outputs[n] = func->eval(inputs[n]);
    }

    void backPropagate( Activation* const prev,  Activation* const curr, const Activation* const next,
                       Grads* const grad, const Real* const weights, const Real* const biases) const override
    {
        const Real* __restrict__ const inputs = curr->in_vals +n1stNeuron;
        Real* __restrict__ const deltas = curr->errvals +n1stNeuron;
        Real* __restrict__ const gradbias = grad->_B +n1stBias;
        //__builtin_assume_aligned(deltas,  __vec_width__);
        //__builtin_assume_aligned(inputs, __vec_width__);
        //__builtin_assume_aligned(gradbias, __vec_width__);

        for (int n=0; n<nNeurons; n++) deltas[n] *= func->evalDiff(inputs[n]);

        for (const auto & link : *input_links)
                      link->backPropagate(curr,curr,weights,grad->_W);

        if(recurrent_link not_eq nullptr && prev not_eq nullptr)
            recurrent_link->backPropagate(prev,curr,weights,grad->_W);

        for (int n=0; n<nNeurons; n++) gradbias[n] += deltas[n];
    }

    void regularize(Real* const weights, Real* const biases, const Real lambda) const override
  	{
      if(bOutput) return;
  		for (const auto & link : *input_links)
                  link->regularize(weights, lambda);

      if(recurrent_link not_eq nullptr)
        recurrent_link->regularize(weights, lambda);

      Lpenalization(biases, n1stBias, nNeurons, lambda);
  	}
};

class Conv2DLayer : public Layer
{
    const vector<LinkToConv2D*>* const input_links;
    const Function* const func;

 public:
    Conv2DLayer(int _nNeurons, int _n1stNeuron, int _n1stBias,
      const vector<LinkToConv2D*>* const nl_il,
      const Function* const f, const int nn_simd, const bool bOut = false) :
      Layer(_nNeurons, _n1stNeuron, _n1stBias, nn_simd, bOut),
      input_links(nl_il), func(f)
    {
	   printf("Conv2D Layer of size %d, with first ID %d and first bias ID %d\n", nNeurons,n1stNeuron, n1stBias);
    }

    void propagate(const Activation* const prev, Activation* const curr,
                   const Real* const weights, const Real* const biases) const  override
    {
        Real* __restrict__ const inputs  = curr->in_vals +n1stNeuron;
        Real* __restrict__ const outputs = curr->outvals +n1stNeuron;
        const Real* __restrict__ const bias = biases +n1stBias;
        //__builtin_assume_aligned(outputs,  __vec_width__);
        //__builtin_assume_aligned(inputs, __vec_width__);
        //__builtin_assume_aligned(bias, __vec_width__);

        for(int o=0; o<nNeurons;  o++) inputs[o] = bias[o];

        for (const auto & link : *input_links)
                      link->propagate(curr,curr,weights);

        //recurrent con2D? I get them for free with this code, but let's hope it never comes to that
        //if(recurrent_link not_eq nullptr && prev not_eq nullptr)
        //    recurrent_link->propagate(prev,curr,weights);

        for (int n=0; n<nNeurons; n++) outputs[n] = func->eval(inputs[n]);
    }
	   void backPropagate( Activation* const prev,  Activation* const curr, const Activation* const next,
                       Grads* const grad, const Real* const weights, const Real* const biases) const  override
    {
        const Real* __restrict__ const inputs = curr->in_vals +n1stNeuron;
        Real* __restrict__ const errors = curr->errvals +n1stNeuron;
        Real* __restrict__ const gradbias = grad->_B +n1stBias;
        //__builtin_assume_aligned(errors,  __vec_width__);
        //__builtin_assume_aligned(inputs, __vec_width__);
        //__builtin_assume_aligned(gradbias, __vec_width__);

        for (int n=0; n<nNeurons; n++) errors[n] *= func->evalDiff(inputs[n]);

        for (const auto & link : *input_links)
            link->backPropagate(curr,curr,weights,grad->_W);

        //recurrent con2D? I get them for free with this code, but let's hope it never comes to that
        //if(recurrent_link not_eq nullptr && prev not_eq nullptr)
        //    recurrent_link->backPropagate(prev,curr,weights,grad->_W);

        for (int n=0; n<nNeurons; n++) gradbias[n] += errors[n];
    }

    void initialize(mt19937* const gen, Real* const weights, Real* const biases) const override
    {
      uniform_real_distribution<Real> dis(-sqrt(6./nNeurons),sqrt(6./nNeurons));

      for (const auto & link : *input_links)
        if(link not_eq nullptr) link->initialize(gen,weights);

      for (int w=n1stBias; w<n1stBias+nNeurons_simd; w++)
				biases[w] = dis(*gen);
    }

    void save(std::ostringstream & outWeights, std::ostringstream & outBiases,
  			Real* const _weights, Real* const biases) const
  	{
  		for (const auto & l : *input_links)
  			if(l not_eq nullptr) l->save(outWeights, _weights);

      //if(recurrent_link not_eq nullptr)
      //  recurrent_link->save(outWeights, _weights);

  		for (int w=n1stBias; w<n1stBias+nNeurons; w++)
  			outBiases << biases[w] << "\n";
  	}

    void restart(std::istringstream & bufWeights, std::istringstream & bufBiases,
  			Real* const _weights, Real* const biases) const
  	{
      for (const auto & l : *input_links)
  			if(l not_eq nullptr) l->restart(bufWeights, _weights);

      //if(recurrent_link not_eq nullptr)
      //  recurrent_link->restart(bufWeights, _weights);

  		Real tmp;
  		for (int w=n1stBias; w<n1stBias+nNeurons; w++) {
  			bufBiases >> tmp;
  			assert(not std::isnan(tmp) & not std::isinf(tmp));
  			biases[w] = tmp;
  		}
  	}

    void regularize(Real* const weights, Real* const biases, const Real lambda) const override
  	{
      if(bOutput) return;
  		for (const auto & link : *input_links)
                  link->regularize(weights, lambda);

      Lpenalization(biases, n1stBias, nNeurons, lambda);
  	}
};

class LSTMLayer: public Layer
{
    const int n1stCell, n1stBiasIG, n1stBiasFG, n1stBiasOG;
    const vector<LinkToLSTM*>* const input_links;
    const LinkToLSTM* const recurrent_link;
    const Function* const outFunc;
    const Function* const gateFunc;
    const Function* const cellFunc;

 public:

    LSTMLayer(int _nNeurons, int _n1stNeuron, int _indState,
              int _n1stBias, int _n1stBiasIG, int _n1stBiasFG, int _n1stBiasOG,
              const vector<LinkToLSTM*>* const rl_il,
              const LinkToLSTM* const rl_rl, const Function* const f,
              const Function* const g, const Function* const c,
              const int nn_simd, const bool bOut = false) :
    Layer(_nNeurons, _n1stNeuron, _n1stBias, nn_simd, bOut),
    n1stCell(_indState), n1stBiasIG(_n1stBiasIG), n1stBiasFG(_n1stBiasFG),
    n1stBiasOG(_n1stBiasOG), input_links(rl_il), recurrent_link(rl_rl),
    outFunc(f), gateFunc(g), cellFunc(c)
    {
	    printf("LSTM Layer of size %d, with first ID %d, first cell ID %d, and first bias ID %d\n",nNeurons, n1stNeuron, n1stCell, n1stBias);
        assert(n1stBiasIG==n1stBias  +nn_simd);
        assert(n1stBiasFG==n1stBiasIG+nn_simd);
        assert(n1stBiasOG==n1stBiasFG+nn_simd);
    }

    void propagate(const Activation* const prev, Activation* const curr,
                   const Real* const weights, const Real* const biases) const  override
    {
        Real* __restrict__ const outputI = curr->oIGates +n1stCell;
        Real* __restrict__ const outputF = curr->oFGates +n1stCell;
        Real* __restrict__ const outputO = curr->oOGates +n1stCell;
        Real* __restrict__ const outputC = curr->oMCell +n1stCell;
        Real* __restrict__ const inputs = curr->in_vals +n1stNeuron;
        Real* __restrict__ const inputI = curr->iIGates +n1stCell;
        Real* __restrict__ const inputF = curr->iFGates +n1stCell;
        Real* __restrict__ const inputO = curr->iOGates +n1stCell;
        const Real* __restrict__ const biasC = biases +n1stBias;
        const Real* __restrict__ const biasI = biases +n1stBiasIG;
        const Real* __restrict__ const biasF = biases +n1stBiasFG;
        const Real* __restrict__ const biasO = biases +n1stBiasOG;
        //__builtin_assume_aligned(outputI,  __vec_width__);
        //__builtin_assume_aligned(outputF, __vec_width__);
        //__builtin_assume_aligned(outputO, __vec_width__);
        //__builtin_assume_aligned(outputC,  __vec_width__);
        //__builtin_assume_aligned(inputs, __vec_width__);
        //__builtin_assume_aligned(inputI, __vec_width__);
        //__builtin_assume_aligned(inputF,  __vec_width__);
        //__builtin_assume_aligned(inputO, __vec_width__);
        //__builtin_assume_aligned(biasC, __vec_width__);
        //__builtin_assume_aligned(biasI, __vec_width__);
        //__builtin_assume_aligned(biasF, __vec_width__);
        //__builtin_assume_aligned(biasO, __vec_width__);

        for (int n=0; n<nNeurons; n++) {
            inputs[n] = biasC[n];
            inputI[n] = biasI[n];
            inputF[n] = biasF[n];
            inputO[n] = biasO[n];
        }

        for (const auto & link : *input_links)
                      link->propagate(curr,curr,weights);

        if(recurrent_link not_eq nullptr && prev not_eq nullptr)
            recurrent_link->propagate(prev,curr,weights);

        for (int n=0; n<nNeurons; n++) {
          outputC[n] = cellFunc->eval(inputs[n]);
          outputI[n] = gateFunc->eval(inputI[n]);
          outputF[n] = gateFunc->eval(inputF[n]);
          //#ifndef __posDef_layers_
          outputO[n] = gateFunc->eval(inputO[n]);
          //#else
          //outputO[n] = SoftSign::eval(inputO[n]);
          //#endif

          curr->ostates[n1stCell+n] = outputC[n] * outputI[n] +
                  (prev==nullptr ?  0 : prev->ostates[n1stCell+n] * outputF[n]);

          curr->outvals[n1stNeuron+n] =outputO[n] *
                                       outFunc->eval(curr->ostates[n1stCell+n]);
        }
    }
	 void backPropagate( Activation* const prev,  Activation* const curr, const Activation* const next,
                       Grads* const grad, const Real* const weights, const Real* const biases) const  override
    {
        const Real* __restrict__ const inputs = curr->in_vals +n1stNeuron;
        const Real* __restrict__ const inputI = curr->iIGates +n1stCell;
        const Real* __restrict__ const inputF = curr->iFGates +n1stCell;
        const Real* __restrict__ const inputO = curr->iOGates +n1stCell;
        const Real* __restrict__ const outputI = curr->oIGates +n1stCell;
        const Real* __restrict__ const outputF = curr->oFGates +n1stCell;
        const Real* __restrict__ const outputO = curr->oOGates +n1stCell;
        const Real* __restrict__ const outputC = curr->oMCell +n1stCell;
        Real* __restrict__ const deltas = curr->errvals +n1stNeuron;
        Real* __restrict__ const deltaI = curr->eIGates +n1stCell;
        Real* __restrict__ const deltaF = curr->eFGates +n1stCell;
        Real* __restrict__ const deltaO = curr->eOGates +n1stCell;
        Real* __restrict__ const deltaC = curr->eMCell +n1stCell;
        Real* __restrict__ const gradbiasC = grad->_B +n1stBias;
        Real* __restrict__ const gradbiasI = grad->_B +n1stBiasIG;
        Real* __restrict__ const gradbiasF = grad->_B +n1stBiasFG;
        Real* __restrict__ const gradbiasO = grad->_B +n1stBiasOG;
        //__builtin_assume_aligned(outputI,  __vec_width__);
        //__builtin_assume_aligned(outputF, __vec_width__);
        //__builtin_assume_aligned(outputO, __vec_width__);
        //__builtin_assume_aligned(outputC,  __vec_width__);
        //__builtin_assume_aligned(inputs, __vec_width__);
        //__builtin_assume_aligned(inputI, __vec_width__);
        //__builtin_assume_aligned(inputF,  __vec_width__);
        //__builtin_assume_aligned(inputO, __vec_width__);
        //__builtin_assume_aligned(deltas, __vec_width__);
        //__builtin_assume_aligned(deltaI, __vec_width__);
        //__builtin_assume_aligned(deltaF, __vec_width__);
        //__builtin_assume_aligned(deltaO, __vec_width__);
        //__builtin_assume_aligned(deltaC, __vec_width__);
        //__builtin_assume_aligned(gradbiasC, __vec_width__);
        //__builtin_assume_aligned(gradbiasI, __vec_width__);
        //__builtin_assume_aligned(gradbiasF, __vec_width__);
        //__builtin_assume_aligned(gradbiasO, __vec_width__);

        for (int n=0; n<nNeurons; n++) {
          deltaC[n] = outputI[n] * cellFunc->evalDiff(inputs[n]);
          deltaI[n] = outputC[n] * gateFunc->evalDiff(inputI[n]);
          //#ifndef __posDef_layers_
          deltaO[n] =  deltas[n] * gateFunc->evalDiff(inputO[n]);
          //#else
          //deltaO[n] =  deltas[n] * SoftSign::evalDiff(inputO[n]);
          //#endif

          const Real outState = outFunc->eval(curr->ostates[n1stCell+n]);
          const Real diffState = outFunc->evalDiff(curr->ostates[n1stCell+n]);

          deltas[n] = deltas[n] * outputO[n] * diffState +
            (next==nullptr ? 0 : next->errvals[n1stNeuron+n]*next->oFGates[n1stCell+n]);

          deltaF[n] = (prev==nullptr) ? 0
                      : prev->ostates[n1stCell+n] * gateFunc->evalDiff(inputF[n]);

          deltaC[n] *= deltas[n];
          deltaI[n] *= deltas[n];
          deltaF[n] *= outState;
          deltaO[n] *= deltas[n];
        }

    	for (const auto & link : *input_links)
    				  link->backPropagate(curr,curr,weights,grad->_W);

    	if(recurrent_link not_eq nullptr && prev not_eq nullptr)
    		recurrent_link->backPropagate(prev,curr,weights,grad->_W);

    	for (int n=0; n<nNeurons; n++)  { //grad bias == delta
    		gradbiasC[n] += deltaC[n];
    		gradbiasI[n] += deltaI[n];
    		gradbiasF[n] += deltaF[n];
    		gradbiasO[n] += deltaO[n];
    	}
    }

    void initialize(mt19937* const gen, Real* const weights, Real* const biases) const override
    {
      uniform_real_distribution<Real> dis(-sqrt(6./nNeurons),sqrt(6./nNeurons));

      for (const auto & link : *input_links)
        if(link not_eq nullptr) link->initialize(gen,weights);

      if(recurrent_link not_eq nullptr) 
          recurrent_link->initialize(gen,weights);

      for (int w=n1stBias; w<n1stBias+nNeurons_simd; w++)
				biases[w] = dis(*gen);
      for (int w=n1stBiasIG; w<n1stBiasIG+nNeurons_simd; w++)
				biases[w] = dis(*gen) - 1.0;
			for (int w=n1stBiasFG; w<n1stBiasFG+nNeurons_simd; w++)
				biases[w] = dis(*gen) + 1.0;
			for (int w=n1stBiasOG; w<n1stBiasOG+nNeurons_simd; w++)
				biases[w] = dis(*gen) - 1.0;
    }

    void save(std::ostringstream & outWeights, std::ostringstream & outBiases,
  			Real* const _weights, Real* const _biases) const
  	{
  		for (const auto & l : *input_links)
  			if(l not_eq nullptr) l->save(outWeights, _weights);

      if(recurrent_link not_eq nullptr)
        recurrent_link->save(outWeights, _weights);

  		for (int w=n1stBias; w<n1stBias+nNeurons; w++)
  			outBiases << _biases[w] << "\n";
      for (int w=n1stBiasIG; w<n1stBiasIG+nNeurons; w++)
				outBiases << _biases[w] << "\n";
			for (int w=n1stBiasFG; w<n1stBiasFG+nNeurons; w++)
				outBiases << _biases[w] << "\n";
			for (int w=n1stBiasOG; w<n1stBiasOG+nNeurons; w++)
				outBiases << _biases[w] << "\n";
  	}

    void restart(std::istringstream & bufWeights, std::istringstream & bufBiases,
  			Real* const _weights, Real* const _biases) const
  	{
      for (const auto & l : *input_links)
  			if(l not_eq nullptr) l->restart(bufWeights, _weights);

      if(recurrent_link not_eq nullptr)
        recurrent_link->restart(bufWeights, _weights);

  		Real tmp;
  		for (int w=n1stBias; w<n1stBias+nNeurons; w++) {
  			bufBiases >> tmp;
  			assert(not std::isnan(tmp) & not std::isinf(tmp));
  			_biases[w] = tmp;
  		}
      for (int w=n1stBiasIG; w<n1stBiasIG+nNeurons; w++){
				bufBiases >> tmp;
				assert(not std::isnan(tmp) & not std::isinf(tmp));
				_biases[w] = tmp;
			}
			for (int w=n1stBiasFG; w<n1stBiasFG+nNeurons; w++){
				bufBiases >> tmp;
				assert(not std::isnan(tmp) & not std::isinf(tmp));
				_biases[w] = tmp;
			}
			for (int w=n1stBiasOG; w<n1stBiasOG+nNeurons; w++){
				bufBiases >> tmp;
				assert(not std::isnan(tmp) & not std::isinf(tmp));
				_biases[w] = tmp;
			}
  	}

    void regularize(Real* const weights, Real* const biases, const Real lambda) const override
  	{
      if(bOutput) return;
  		for (const auto & link : *input_links)
                  link->regularize(weights, lambda);

      if(recurrent_link not_eq nullptr)
        recurrent_link->regularize(weights, lambda);

      Lpenalization(biases, n1stBias, nNeurons, lambda);
  	}
};
