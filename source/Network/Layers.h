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
#include "../Profiler.h"

#ifndef __CHECK_DIFF
#define LSTM_PRIME_FAC 1 //input/output gates start closed, forget starts open
#else //else we are testing finite diffs
#define LSTM_PRIME_FAC 0 //otherwise finite differences are small
#endif

class Layer
{
public:
	const Uint nNeurons, n1stNeuron, n1stBias, nNeurons_simd;
	const Function* const func;
	//Profiler* profiler;
	const bool bOutput;

	virtual ~Layer() { _dispose_object(func); }
	Layer(const Uint _nNeurons, const Uint _n1stNeuron, const Uint _n1stBias,
			const Function* const f, const Uint nn_simd, const bool bOut) :
				nNeurons(_nNeurons), n1stNeuron(_n1stNeuron), n1stBias(_n1stBias),
				nNeurons_simd(nn_simd), func(f), bOutput(bOut) {}

	virtual void propagate(const Activation* const prev, Activation* const curr,
			const nnReal* const weights, const nnReal* const biases) const = 0;

	virtual void backPropagate(Activation* const prev,  Activation* const curr,
			const Activation* const next, Grads* const grad,
			const nnReal* const weights, const nnReal* const biases) const = 0;

	virtual void initialize(mt19937* const gen, nnReal* const weights,
			nnReal* const biases, Real initializationFac) const = 0;

	virtual void save(vector<nnReal> & outWeights, vector<nnReal> & outBiases,
			nnReal* const _weights, nnReal* const _biases) const = 0;

	virtual void restart(vector<nnReal> & bufWeights, vector<nnReal> & bufBiases,
			nnReal* const _weights, nnReal* const _biases) const = 0;

	virtual void regularize(nnReal* const weights, nnReal* const biases,
			const Real lambda) const = 0;

	void propagate(Activation* const curr, const nnReal* const weights,
			const nnReal* const biases) const
	{
		return propagate(nullptr, curr, weights, biases);
	}
	void backPropagate( Activation* const curr, Grads* const grad,
			const nnReal* const weights, const nnReal* const biases) const
	{
		return backPropagate(nullptr, curr, nullptr, grad, weights, biases);
	}
};

template<typename TLink>
class BaseLayer: public Layer
{
public:
	const vector<TLink*> input_links;
	const TLink* const recurrent_link;

	virtual ~BaseLayer()
	{
		for (auto & trash : input_links) _dispose_object(trash);
		_dispose_object(recurrent_link);
	}

	BaseLayer(const Uint _nNeurons, const Uint _n1stNeuron, const Uint _n1stBias,
			const vector<TLink*> nl_il, const TLink* const nl_rl,
			const Function* const f, const Uint nn_simd, const bool bOut) :
				Layer(_nNeurons, _n1stNeuron, _n1stBias, f, nn_simd, bOut),
				input_links(nl_il), recurrent_link(nl_rl) {}

	virtual void propagate(const Activation* const prev, Activation* const curr,
			const nnReal* const weights, const nnReal* const biases) const override
	{
		nnReal* __restrict__ const outputs = curr->outvals +n1stNeuron;
		nnReal* __restrict__ const inputs = curr->in_vals +n1stNeuron;
		const nnReal* __restrict__ const bias = biases +n1stBias;
		//const int thrID = omp_get_thread_num();
		//if(thrID==1) profiler->push_start("FB");
		#pragma omp simd aligned(inputs,bias : __vec_width__) safelen(simdWidth)
		for (Uint n=0; n<nNeurons; n++) inputs[n] = bias[n];
		//if(thrID==1)  profiler->stop_start("FP");
		for (const auto & link : input_links)
			link->propagate(curr,curr,weights);
		if(recurrent_link not_eq nullptr && prev not_eq nullptr)
			recurrent_link->propagate(prev,curr,weights);

		//if(thrID==1)  profiler->stop_start("FD");
		//for (Uint n=0; n<nNeurons; n++) outputs[n] = func->eval(inputs[n]);
		func->eval(inputs,outputs,nNeurons_simd);
		//if(thrID==1) profiler->pop_stop();
	}

	virtual void backPropagate(Activation*const prev,  Activation*const curr,
			const Activation*const next, Grads*const grad, const nnReal*const weights,
			const nnReal*const biases) const override
	{
		const nnReal* __restrict__ const inputs = curr->in_vals +n1stNeuron;
		nnReal* __restrict__ const deltas = curr->errvals +n1stNeuron;
		nnReal* __restrict__ const gradbias = grad->_B +n1stBias;
		//const int thrID = omp_get_thread_num();
		//if(thrID==1) profiler->push_start("BD");
		for (Uint n=0; n<nNeurons; n++) deltas[n] *= func->evalDiff(inputs[n]);

		//if(thrID==1)  profiler->stop_start("BP");
		for (const auto & link : input_links)
			link->backPropagate(curr,curr,weights,grad->_W);

		if(recurrent_link not_eq nullptr && prev not_eq nullptr)
			recurrent_link->backPropagate(prev,curr,weights,grad->_W);

		//if(thrID==1)  profiler->stop_start("BB");
#pragma omp simd aligned(gradbias,deltas: __vec_width__) safelen(simdWidth)
		for (Uint n=0; n<nNeurons; n++) gradbias[n] += deltas[n];
		//if(thrID==1) profiler->pop_stop();
	}

	virtual void initialize(mt19937* const gen, nnReal* const weights,
			nnReal* const biases, Real initializationFac) const override
	{
		const nnReal prefac = (initializationFac>0) ? initializationFac : 1;
		const nnReal biasesInit = prefac*func->biasesInitFactor(nNeurons);//usually 0
		uniform_real_distribution<nnReal> dis(-biasesInit, biasesInit);

		for (Uint w=n1stBias; w<n1stBias+nNeurons_simd; w++)
			biases[w] = dis(*gen);

		for (const auto & link : input_links)
			if(link not_eq nullptr) link->initialize(gen,weights,func,prefac);

		if(recurrent_link not_eq nullptr)
			recurrent_link->initialize(gen,weights,func,prefac);
	}

	virtual void save(vector<nnReal> & outWeights, vector<nnReal> & outBiases,
			nnReal* const _weights, nnReal* const _biases) const override
	{
		for (const auto & l : input_links)
			if(l not_eq nullptr) l->save(outWeights, _weights);

		if(recurrent_link not_eq nullptr)
			recurrent_link->save(outWeights, _weights);

		for (Uint w=n1stBias; w<n1stBias+nNeurons; w++)
			outBiases.push_back(_biases[w]);
	}

	virtual void restart(vector<nnReal>& bufWeights, vector<nnReal>& bufBiases,
			nnReal* const _weights, nnReal* const _biases) const override
	{
		for (const auto & l : input_links)
			if(l not_eq nullptr) l->restart(bufWeights, _weights);

		if(recurrent_link not_eq nullptr)
			recurrent_link->restart(bufWeights, _weights);

		for (Uint w=n1stBias; w<n1stBias+nNeurons; w++)
		{
			_biases[w] = bufBiases.front();
			bufBiases.erase(bufBiases.begin(),bufBiases.begin()+1);
			assert(!std::isnan(_biases[w]) && !std::isinf(_biases[w]));
		}
	}

	virtual void regularize(nnReal* const weights, nnReal* const biases,
			const Real lambda) const override
	{
		if(bOutput) return;
		for (const auto & link : input_links)
			link->regularize(weights, lambda);

		if(recurrent_link not_eq nullptr)
			recurrent_link->regularize(weights, lambda);

		Lpenalization(biases, n1stBias, nNeurons, lambda);
	}
};

class NormalLayer: public BaseLayer<NormalLink>
{
public:
	NormalLayer(Uint _nNeurons, Uint _n1stNeuron, Uint _n1stBias,
			const vector<NormalLink*> nl_il, const NormalLink* const nl_rl,
			const Function* const f, const Uint nn_simd, const bool bOut = false) :
				BaseLayer(_nNeurons, _n1stNeuron, _n1stBias, nl_il, nl_rl, f, nn_simd, bOut)
	{
		printf("%s layer of size %d, with first ID %d and first bias ID %d\n",
				bOut?"Output":"Normal", nNeurons, n1stNeuron, n1stBias);
	}
};

class Conv2DLayer : public BaseLayer<LinkToConv2D>
{
public:
	Conv2DLayer(Uint _nNeurons, Uint _n1stNeuron, Uint _n1stBias,
			const vector<LinkToConv2D*> nl_il,
			const Function* const f, const Uint nn_simd, const bool bOut = false) :
				BaseLayer(_nNeurons, _n1stNeuron, _n1stBias, nl_il,
						static_cast<LinkToConv2D*>(nullptr), f, nn_simd, bOut)
	{
		printf("Conv2D Layer of size %d, with first ID %d and first bias ID %d\n",
				nNeurons,n1stNeuron, n1stBias);
	}
};

class LSTMLayer: public BaseLayer<LinkToLSTM>
{
	const Uint n1stCell, n1stBiasIG, n1stBiasFG, n1stBiasOG;
	const Function* const gate;
	const Function* const cell;

public:
	~LSTMLayer() {_dispose_object(gate); _dispose_object(cell);}
	LSTMLayer(Uint _nNeurons, Uint _n1stNeuron, Uint _indState, Uint _n1stBias,
			Uint _n1stBiasIG, Uint _n1stBiasFG, Uint _n1stBiasOG,
			const vector<LinkToLSTM*> rl_il, const LinkToLSTM* const rl_rl,
			const Function* const f, const Function* const g,
			const Function* const c, const Uint nn_simd, const bool bOut=false):
				BaseLayer(_nNeurons,_n1stNeuron,_n1stBias,rl_il,rl_rl,f,nn_simd,bOut),
				n1stCell(_indState), n1stBiasIG(_n1stBiasIG), n1stBiasFG(_n1stBiasFG),
				n1stBiasOG(_n1stBiasOG), gate(g), cell(c)
	{
		printf("LSTM Layer of size %d, with first ID %d, first cell ID %d, and first bias ID %d\n",
				nNeurons, n1stNeuron, n1stCell, n1stBias);
		assert(n1stBiasIG==n1stBias  +nn_simd);
		assert(n1stBiasFG==n1stBiasIG+nn_simd);
		assert(n1stBiasOG==n1stBiasFG+nn_simd);
	}

	void propagate(const Activation* const prev, Activation* const curr,
			const nnReal* const weights, const nnReal* const biases) const override
	{
		nnReal* __restrict__ const outputI = curr->oIGates +n1stCell;
		nnReal* __restrict__ const outputF = curr->oFGates +n1stCell;
		nnReal* __restrict__ const outputO = curr->oOGates +n1stCell;
		nnReal* __restrict__ const outputC = curr->oMCell +n1stCell;
		nnReal* __restrict__ const inputs = curr->in_vals +n1stNeuron;
		nnReal* __restrict__ const inputI = curr->iIGates +n1stCell;
		nnReal* __restrict__ const inputF = curr->iFGates +n1stCell;
		nnReal* __restrict__ const inputO = curr->iOGates +n1stCell;
		const nnReal* __restrict__ const biasC = biases +n1stBias;
		const nnReal* __restrict__ const biasI = biases +n1stBiasIG;
		const nnReal* __restrict__ const biasF = biases +n1stBiasFG;
		const nnReal* __restrict__ const biasO = biases +n1stBiasOG;
		const nnReal* __restrict__ const oldState = (prev==nullptr ? curr->ostates : prev->ostates) +n1stCell; //if nullptr then unused, but assigned for safety
		nnReal* __restrict__ const state = curr->ostates +n1stCell;
		nnReal* __restrict__ const output = curr->outvals +n1stNeuron;

#pragma omp simd aligned(inputs, inputI, inputF, inputO, biasC, biasI, biasF, biasO: __vec_width__) safelen(simdWidth)
		for (Uint n=0; n<nNeurons; n++) {
			inputs[n] = biasC[n];
			inputI[n] = biasI[n];
			inputF[n] = biasF[n];
			inputO[n] = biasO[n];
		}

		for (const auto & link : input_links)
			link->propagate(curr,curr,weights);

		if(recurrent_link not_eq nullptr && prev not_eq nullptr)
			recurrent_link->propagate(prev,curr,weights);

		func->eval(inputs,outputC,nNeurons_simd);
		gate->eval(inputI,outputI,nNeurons_simd);
		gate->eval(inputF,outputF,nNeurons_simd);
		gate->eval(inputO,outputO,nNeurons_simd);

#pragma omp simd aligned(state, outputC, outputI, oldState, outputF, output, outputO: __vec_width__) safelen(simdWidth)
		for (Uint n=0; n<nNeurons; n++) {
			state[n]=outputC[n]*outputI[n] +(prev==nullptr?0:oldState[n]*outputF[n]);
			output[n] = outputO[n] * state[n];
		}
	}

	void backPropagate(Activation*const prev, Activation*const curr,
			const Activation*const next, Grads* const grad,
			const nnReal* const weights, const nnReal* const biases) const override
	{
		const nnReal* __restrict__ const inputs = curr->in_vals +n1stNeuron;
		const nnReal* __restrict__ const inputI = curr->iIGates +n1stCell;
		const nnReal* __restrict__ const inputF = curr->iFGates +n1stCell;
		const nnReal* __restrict__ const inputO = curr->iOGates +n1stCell;
		const nnReal* __restrict__ const outputI = curr->oIGates +n1stCell;
		//const nnReal* __restrict__ const outputF = curr->oFGates +n1stCell;
		const nnReal* __restrict__ const outputO = curr->oOGates +n1stCell;
		const nnReal* __restrict__ const outputC = curr->oMCell +n1stCell;
		nnReal* __restrict__ const deltas = curr->errvals +n1stNeuron;
		nnReal* __restrict__ const deltaI = curr->eIGates +n1stCell;
		nnReal* __restrict__ const deltaF = curr->eFGates +n1stCell;
		nnReal* __restrict__ const deltaO = curr->eOGates +n1stCell;
		nnReal* __restrict__ const deltaC = curr->eMCell +n1stCell;
		nnReal* __restrict__ const gradbiasC = grad->_B +n1stBias;
		nnReal* __restrict__ const gradbiasI = grad->_B +n1stBiasIG;
		nnReal* __restrict__ const gradbiasF = grad->_B +n1stBiasFG;
		nnReal* __restrict__ const gradbiasO = grad->_B +n1stBiasOG;

		for (Uint n=0; n<nNeurons; n++)
		{
			const nnReal deltaOut = deltas[n];

			deltaC[n] = func->evalDiff(inputs[n]) * outputI[n];
			deltaI[n] = gate->evalDiff(inputI[n]) * outputC[n];
			deltaF[n] = (prev==nullptr) ? 0 :
					gate->evalDiff(inputF[n]) * prev->ostates[n1stCell+n];
			deltaO[n] = gate->evalDiff(inputO[n]) * deltaOut * curr->ostates[n1stCell+n];

			deltas[n] = deltaOut * outputO[n] +
					(next==nullptr ? 0
							: next->errvals[n1stNeuron+n]*next->oFGates[n1stCell+n]);

			deltaC[n] *= deltas[n];
			deltaI[n] *= deltas[n];
			deltaF[n] *= deltas[n];
			//grad bias == delta:
			gradbiasC[n] += deltaC[n];
			gradbiasI[n] += deltaI[n];
			gradbiasF[n] += deltaF[n];
			gradbiasO[n] += deltaO[n];
		}

		for (const auto & link : input_links)
			link->backPropagate(curr,curr,weights,grad->_W);

		if(recurrent_link not_eq nullptr && prev not_eq nullptr)
			recurrent_link->backPropagate(prev,curr,weights,grad->_W);
	}

	void initialize(mt19937*const gen, nnReal*const weights,
				nnReal* const biases, Real initializationFac) const override
	{
		const Real biasesInit = ((initializationFac>0) ? initializationFac : 1)
								* func->biasesInitFactor(nNeurons); //usually 0
		uniform_real_distribution<nnReal> dis(-biasesInit, biasesInit);
		BaseLayer::initialize(gen, weights, biases, initializationFac);
		for (Uint w=n1stBiasIG; w<n1stBiasIG+nNeurons_simd; w++)
			biases[w] = dis(*gen) - LSTM_PRIME_FAC;
		for (Uint w=n1stBiasFG; w<n1stBiasFG+nNeurons_simd; w++)
			biases[w] = dis(*gen) + LSTM_PRIME_FAC;
		for (Uint w=n1stBiasOG; w<n1stBiasOG+nNeurons_simd; w++)
			biases[w] = dis(*gen) - LSTM_PRIME_FAC;
	}

	void save(std::vector<nnReal> & outWeights, std::vector<nnReal> & outBiases,
			nnReal* const _weights, nnReal* const _biases) const override
	{
		BaseLayer::save(outWeights, outBiases, _weights, _biases);
		for (Uint w=n1stBiasIG; w<n1stBiasIG+nNeurons; w++)
			outBiases.push_back(_biases[w]);
		for (Uint w=n1stBiasFG; w<n1stBiasFG+nNeurons; w++)
			outBiases.push_back(_biases[w]);
		for (Uint w=n1stBiasOG; w<n1stBiasOG+nNeurons; w++)
			outBiases.push_back(_biases[w]);
	}

	void restart(vector<nnReal> & bufWeights, vector<nnReal> & bufBiases,
			nnReal* const _weights, nnReal* const _biases) const override
	{
		BaseLayer::restart(bufWeights, bufBiases, _weights, _biases);
		for (Uint w=n1stBiasIG; w<n1stBiasIG+nNeurons; w++) {
			_biases[w] = bufBiases.front();
			bufBiases.erase(bufBiases.begin(),bufBiases.begin()+1);
			assert(!std::isnan(_biases[w]) && !std::isinf(_biases[w]));
		}
		for (Uint w=n1stBiasFG; w<n1stBiasFG+nNeurons; w++) {
			_biases[w] = bufBiases.front();
			bufBiases.erase(bufBiases.begin(),bufBiases.begin()+1);
			assert(!std::isnan(_biases[w]) && !std::isinf(_biases[w]));
		}
		for (Uint w=n1stBiasOG; w<n1stBiasOG+nNeurons; w++) {
			_biases[w] = bufBiases.front();
			bufBiases.erase(bufBiases.begin(),bufBiases.begin()+1);
			assert(!std::isnan(_biases[w]) && !std::isinf(_biases[w]));
		}
	}

	void regularize(nnReal* const weights, nnReal* const biases, const Real lambda)
	const override
	{
		BaseLayer::regularize(weights, biases, lambda);
	}
};

class IntegrateFireLayer: public BaseLayer<NormalLink>
{
	const Uint n1stNoise;

public:
	IntegrateFireLayer(Uint _nNeurons, Uint _iNeuron, Uint _iNoise, Uint _iBias,
		const vector<NormalLink*> il, const NormalLink* const rl,
		const Function* const f, const Uint nn_simd, const bool bOut=true):
		BaseLayer(_nNeurons,_iNeuron,_iBias,il,rl,f,nn_simd,bOut),n1stNoise(_iNoise)
	{
		printf("IntegrateFireLayer Layer of size %d, with first ID %d, first noise ID %d, and first bias ID %d\n",
				nNeurons, n1stNeuron, n1stNoise, n1stBias);
	}

	void propagate(const Activation* const prev, Activation* const curr,
			const nnReal* const weights, const nnReal* const biases) const override
	{
		nnReal* __restrict__ const inputs = curr->in_vals +n1stNeuron;
		nnReal* __restrict__ const output = curr->outvals +n1stNeuron;
		//if noise as part of input?
		//const nnReal* __restrict__ const sampled_noise = curr->outvals +n1stNoise;

		//Biases array contains, for each output, the bias of the sigmoid,
		// the stdev of the noise inside the sigmoid, the inverse time scale
		// of the exp decay, and the coefficient of the sigmoid
		const nnReal* __restrict__ const bias   = biases +n1stBias +0*nNeurons;
		const nnReal* __restrict__ const invTau = biases +n1stBias +1*nNeurons;
		const nnReal* __restrict__ const excitr = biases +n1stBias +2*nNeurons;
		//const nnReal* __restrict__ const stDev  = biases +n1stBias +3*nNeurons;

		//prepare input to sigmoid: f(weights \cdot inputs + noise + bias )
		for (Uint n=0; n<nNeurons; n++) inputs[n] = bias[n];// + sampled_noise[n];
		//add the dot product using normal fully connected layer
		for (const auto & link : input_links) link->propagate(curr,curr,weights);
		//evaluate sigmoid:
		func->eval(inputs, output, nNeurons_simd);
		//multiply by excitation coefficient
		for (Uint n=0; n<nNeurons; n++) output[n] *= excitr[n];

		//if not first of sequence, add the term depending on previous realization
		if(prev not_eq nullptr) {
			assert(recurrent_link not_eq nullptr);
			const nnReal* __restrict__ const oldOutp = prev->outvals +n1stNeuron;
			for (Uint n=0; n<nNeurons; n++) output[n] += oldOutp[n] * (1-invTau[n]);
		}
	}

	void backPropagate(Activation*const prev, Activation*const curr,
			const Activation*const next, Grads* const grad,
			const nnReal* const weights, const nnReal* const biases) const override
	{
		//const nnReal* __restrict__ const sampled_noise = curr->outvals +n1stNoise;
		const nnReal* __restrict__ const inputs = curr->in_vals +n1stNeuron;
		const nnReal* __restrict__ const output = curr->outvals +n1stNeuron;
		nnReal* __restrict__ const dErr_curr = curr->errvals +n1stNeuron;

		//Biases array contains, for each output, the bias of the sigmoid,
		// the stdev of the noise inside the sigmoid, the inverse time scale
		// of the exp decay, and the coefficient of the sigmoid
		//const nnReal* __restrict__ const bias   = biases +n1stBias +0*nNeurons;
		const nnReal* __restrict__ const invTau = biases +n1stBias +1*nNeurons;
		const nnReal* __restrict__ const excitr = biases +n1stBias +2*nNeurons;
		//const nnReal* __restrict__ const stDev  = biases +n1stBias +3*nNeurons;

		nnReal* __restrict__ const gradBias   = grad->_B+n1stBias +0*nNeurons;
		nnReal* __restrict__ const gradInvTau = grad->_B+n1stBias +1*nNeurons;
		nnReal* __restrict__ const gradExcitr = grad->_B+n1stBias +2*nNeurons;
		//const nnReal* __restrict__ const gradStDev= grad->_B+n1stBias +3*nNeurons;

		for (Uint n=0; n<nNeurons; n++) {
			//gradient of total error wrt to output of neuron depends on
			//current error, plus exponentially decaying dependence on future errors
			if(prev!=nullptr) prev->errvals[n+n1stNeuron]+=(1-invTau[n])*dErr_curr[n];
			// update the gradient to the parameters of the IaF neuron:
			gradInvTau[n]+= -dErr_curr[n]*output[n];
			gradExcitr[n]+=  dErr_curr[n]*func->eval(inputs[n]);
			//fully connected link expects to find in curr->errvals:
			// dErr_t / dInput_t, which is dErr_t / dOut_t (dot) dSigmoid / dInput_t
			// where Input_t is the input to the sigmoid (contained in inputs[n])
			dErr_curr[n] *= excitr[n]*func->evalDiff(inputs[n]);
			//grad of bias: dErr_t / dInput_t * dInput_t /dBias =  dErr_t / dInput_t
			gradBias[n]  += dErr_curr[n];
		}

		for (const auto & link : input_links)
			link->backPropagate(curr,curr,weights,grad->_W);
	}

	void initialize(mt19937*const gen, nnReal*const weights,
				nnReal* const biases, Real initializationFac) const override
	{
		BaseLayer::initialize(gen, weights, biases, initializationFac);
		nnReal* __restrict__ const bias   = biases +n1stBias +0*nNeurons;
		nnReal* __restrict__ const invTau = biases +n1stBias +1*nNeurons;
		nnReal* __restrict__ const excitr = biases +n1stBias +2*nNeurons;
		for(Uint w=0; w<nNeurons_simd; w++) bias[w]   = -1;
		for(Uint w=0; w<nNeurons_simd; w++) invTau[w] = 0.5;
		for(Uint w=0; w<nNeurons_simd; w++) excitr[w] = 0.5;
	}

	void save(std::vector<nnReal> & outWeights, std::vector<nnReal> & outBiases,
			nnReal* const _weights, nnReal* const _biases) const override
	{
		BaseLayer::save(outWeights, outBiases, _weights, _biases);
		const nnReal* __restrict__ const invTau = _biases +n1stBias +1*nNeurons;
		const nnReal* __restrict__ const excitr = _biases +n1stBias +2*nNeurons;
		for(Uint w=0; w<nNeurons_simd; w++)	outBiases.push_back(invTau[w]);
		for(Uint w=0; w<nNeurons_simd; w++)	outBiases.push_back(excitr[w]);
	}

	void restart(vector<nnReal> & bufWeights, vector<nnReal> & bufBiases,
			nnReal* const _weights, nnReal* const _biases) const override
	{
		BaseLayer::restart(bufWeights, bufBiases, _weights, _biases);
		nnReal* __restrict__ const invTau = _biases +n1stBias +1*nNeurons;
		nnReal* __restrict__ const excitr = _biases +n1stBias +2*nNeurons;
		for(Uint w=0; w<nNeurons_simd; w++){
			invTau[w] = bufBiases.front();
			bufBiases.erase(bufBiases.begin(),bufBiases.begin()+1);
			assert(!std::isnan(invTau[w]) && !std::isinf(invTau[w]));
		}
		for(Uint w=0; w<nNeurons_simd; w++){
			excitr[w] = bufBiases.front();
			bufBiases.erase(bufBiases.begin(),bufBiases.begin()+1);
			assert(!std::isnan(excitr[w]) && !std::isinf(excitr[w]));
		}
	}

	void regularize(nnReal*const weights, nnReal*const biases, const Real lambda)
	const override
	{
		BaseLayer::regularize(weights, biases, lambda);
	}
};
