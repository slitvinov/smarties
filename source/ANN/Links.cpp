/*
 *  Layers.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include "Layers.h"
#include <cassert>

using namespace ErrorHandling;

void Link::set(int _nI, int _iI, int _nO, int _iO, int _iW)
{
	this->nI = _nI; this->iI = _iI; this->nO = _nO; this->iO = _iO; this->iW = _iW;
	print();
}

void Link::print() const
{
	cout << nI << " " << iI << " " << nO << " " << iO << " " << iW << endl;
	fflush(0);
}

Real Link::backPropagate(const Activation* const lab, const int ID_NeuronFrom, const Real* const weights) const
{
	Real dEdOutput(0.);
	const Real* const dEdInput_LayerTo = lab->errvals +iO;
	const Real* const my_weights =  weights +iW;
	//error: sum error signals in target layer times weights
	//weights are sorted in row major order, when row is input neuron, column is output neuron
	for (int i=0; i<nO; i++)
		dEdOutput += dEdInput_LayerTo[i] * my_weights[i*nI + ID_NeuronFrom];
	return dEdOutput;
}

Real Link::propagate(const Activation* const lab, const int ID_NeuronTo, const Real* const weights) const
{
	Real input(0);
	const Real* const output_LayerFrom = lab->outvals +iI;
	const Real* const my_weights =  weights +iW;
	//looping all the neurons that input to the neuron oNeuron and are located in the layer
	for (int i=0; i<nI; i++)
		input += output_LayerFrom[i] * my_weights[ID_NeuronTo*nI + i];
	return input;
}

void Link::propagate(Real* const inputs, const Activation* const lab, const int ID_NeuronTo, const Real* const weights) const
{
	die("Forbidden\n");
}

void Link::computeGrad(const Activation* const activation_From, const Activation* const activation_To, Real* const dEdW) const
{
	//in principle, this link could connect two different time-realizations of the network
	//therefore we would have two separate network activations
	const Real* const output_LayerFrom = activation_From->outvals +iI;
	const Real* const dEdInput_LayerTo = activation_To->errvals +iO;
	Real* const my_dEdW = dEdW +iW;

	for (int j=0; j<nO; j++)
		for (int i=0; i<nI; i++)
			my_dEdW[j*nI +i] = output_LayerFrom[i] * dEdInput_LayerTo[j];
}

void Link::addUpGrads(const Activation* const activation_From, const Activation* const activation_To, Real* const dEdW) const
{
	//in principle, this link could connect two different time-realizations of the network
	//therefore we would have two separate network activations
	const Real* const output_LayerFrom = activation_From->outvals +iI;
	const Real* const dEdInput_LayerTo = activation_To->errvals +iO;
	Real* const my_dEdW = dEdW +iW;

	for (int j=0; j<nO; j++)
		for (int i=0; i<nI; i++)
			my_dEdW[j*nI +i] += output_LayerFrom[i] * dEdInput_LayerTo[j];
}

void Link::initialize(uniform_real_distribution<Real>& dis, mt19937* const gen, Real* const _weights) const
{
	for (int w=iW ; w<(iW + nO*nI); w++)
		*(_weights +w) = dis(*gen) / Real(nO + nI);

	orthogonalize(iW, _weights);
}

void Link::orthogonalize(const int n0, Real* const _weights) const
{
	if (nI>=nO)
		for (int i=1; i<nO; i++)
			for (int j=0; j<i;  j++) {
				Real u_d_u = 0.0;
				Real v_d_u = 0.0;
				for (int k=0; k<nI; k++) {
					u_d_u += *(_weights +n0 +j*nI +k)* *(_weights +n0 +j*nI +k);
					v_d_u += *(_weights +n0 +j*nI +k)* *(_weights +n0 +i*nI +k);
				}

				if(u_d_u>0)
					for (int k=0; k<nI; k++)
						*(_weights+n0+i*nI+k) -= (v_d_u/u_d_u) * *(_weights+n0+j*nI+k);
			}
}

void LinkToLSTM::print() const
{
	cout << nI << " " << iI << " " << nO << " " << iO << " " << iW << " " << iC << " " << iWI << " " << iWF << " " << iWO << " " << endl;
	fflush(0);
}

void LinkToLSTM::set(int _nI, int _iI, int _nO, int _iO, int _iC, int _iW, int _iWI, int _iWF, int _iWO)
{
	this->nI = _nI; this->iI = _iI; this->nO = _nO; this->iO = _iO; this->iW = _iW; this->iC = _iC; this->iWI = _iWI; this->iWF = _iWF; this->iWO = _iWO;
	print();
}

Real LinkToLSTM::backPropagate(const Activation* const lab, const int ID_NeuronFrom, const Real* const weights) const
{
	Real dEdOutput(0.);
	const Real* const dEdState_BlocksTo = lab->errvals +iO;
	const Real* const dEdInput_OGatesTo = lab->eOGates +iC;
	const Real* const dSdInput_FGatesTo = lab->eFGates +iC;
	const Real* const dSdInput_IGatesTo = lab->eIGates +iC;
	const Real* const dSdInput_CellsTo = lab->eMCell +iC;

	const Real* const weights_toCell =  weights +iW;
	const Real* const weights_toOgate =  weights +iWO;
	const Real* const weights_toFgate =  weights +iWF;
	const Real* const weights_toIgate =  weights +iWI;

	//error: sum error signals in target layer times weights
	//weights are sorted in row major order, when row is input neuron, column is output neuron
	for (int i=0; i<nO; i++) {
		const int ID_link = i * nI + ID_NeuronFrom;
		dEdOutput += dEdInput_OGatesTo[i] * weights_toOgate[ID_link] +
				dEdState_BlocksTo[i] * (
						dSdInput_CellsTo[i]  * weights_toCell[ID_link]  +
						dSdInput_IGatesTo[i] * weights_toIgate[ID_link] +
						dSdInput_FGatesTo[i] * weights_toFgate[ID_link] );
	}
	return dEdOutput;
}

void LinkToLSTM::propagate(Real* const inputs, const Activation* const lab, const int ID_NeuronTo, const Real* const weights) const
{
	const Real* const output_LayerFrom = lab->outvals +iI;
	const Real* const weights_toCell =  weights +iW;
	const Real* const weights_toOgate =  weights +iWO;
	const Real* const weights_toFgate =  weights +iWF;
	const Real* const weights_toIgate =  weights +iWI;

	//error: sum error signals in target layer times weights
	//weights are sorted in row major order, when row is input neuron, column is output neuron
	for (int i=0; i<nI; i++) {
		const int ID_link = ID_NeuronTo * nI + i;
		inputs[0] += output_LayerFrom[i] * weights_toCell[ID_link];
		inputs[1] += output_LayerFrom[i] * weights_toIgate[ID_link];
		inputs[2] += output_LayerFrom[i] * weights_toFgate[ID_link];
		inputs[3] += output_LayerFrom[i] * weights_toOgate[ID_link];
	}
}

Real LinkToLSTM::propagate(const Activation* const lab, const int ID_NeuronTo, const Real* const weights) const
{
	die("Forbidden\n"); return 0.;
}

void LinkToLSTM::computeGrad(const Activation* const activation_From, const Activation* const activation_To, Real* const dEdW) const
{
	const Real* const dEdState_BlocksTo = activation_To->errvals +iO;
	const Real* const dEdInput_OGatesTo = activation_To->eOGates +iC;
	const Real* const dSdInput_FGatesTo = activation_To->eFGates +iC;
	const Real* const dSdInput_IGatesTo = activation_To->eIGates +iC;
	const Real* const dSdInput_CellsTo = activation_To->eMCell +iC;
	const Real* const output_LayerFrom = activation_From->outvals +iI;

	Real* const dWdE_toCell =  dEdW +iW;
	Real* const dWdE_toOgate =  dEdW +iWO;
	Real* const dWdE_toFgate =  dEdW +iWF;
	Real* const dWdE_toIgate =  dEdW +iWI;

	for (int j=0; j<nO; j++) {
		const Real eC = dSdInput_CellsTo[j]  * dEdState_BlocksTo[j];
		const Real eI = dSdInput_IGatesTo[j] * dEdState_BlocksTo[j];
		const Real eF = dSdInput_FGatesTo[j] * dEdState_BlocksTo[j];
		const Real eO = dEdInput_OGatesTo[j];

		for (int i=0; i<nI; i++) {
			const int ID_link = j * nI + i;
			dWdE_toCell[ID_link] = output_LayerFrom[i] * eC;
			dWdE_toIgate[ID_link] = output_LayerFrom[i] * eI;
			dWdE_toFgate[ID_link] = output_LayerFrom[i] * eF;
			dWdE_toOgate[ID_link] = output_LayerFrom[i] * eO;
		}
	}
}

void LinkToLSTM::addUpGrads(const Activation* const activation_From, const Activation* const activation_To, Real* const dEdW) const
{
	const Real* const dEdState_BlocksTo = activation_To->errvals +iO;
	const Real* const dEdInput_OGatesTo = activation_To->eOGates +iC;
	const Real* const dSdInput_FGatesTo = activation_To->eFGates +iC;
	const Real* const dSdInput_IGatesTo = activation_To->eIGates +iC;
	const Real* const dSdInput_CellsTo = activation_To->eMCell +iC;
	const Real* const output_LayerFrom = activation_From->outvals +iI;

	Real* const dWdE_toCell =  dEdW +iW;
	Real* const dWdE_toOgate =  dEdW +iWO;
	Real* const dWdE_toFgate =  dEdW +iWF;
	Real* const dWdE_toIgate =  dEdW +iWI;

	for (int j=0; j<nO; j++) {
		const Real eC = dSdInput_CellsTo[j]  * dEdState_BlocksTo[j];
		const Real eI = dSdInput_IGatesTo[j] * dEdState_BlocksTo[j];
		const Real eF = dSdInput_FGatesTo[j] * dEdState_BlocksTo[j];
		const Real eO = dEdInput_OGatesTo[j];

		for (int i=0; i<nI; i++) {
			const int ID_link = j * nI + i;
			dWdE_toCell[ID_link] += output_LayerFrom[i] * eC;
			dWdE_toIgate[ID_link] += output_LayerFrom[i] * eI;
			dWdE_toFgate[ID_link] += output_LayerFrom[i] * eF;
			dWdE_toOgate[ID_link] += output_LayerFrom[i] * eO;
		}
	}
}

void LinkToLSTM::initialize(uniform_real_distribution<Real>& dis, mt19937* const gen, Real* const _weights) const
{
	for (int w=iW ; w<(iW + nO*nI); w++)
		*(_weights +w) = dis(*gen) / Real(nO + nI);
	orthogonalize(iW, _weights);

	for (int w=iWI; w<(iWI+ nO*nI); w++)
		*(_weights +w) = dis(*gen) / Real(nO + nI);
	orthogonalize(iWI, _weights);

	for (int w=iWF; w<(iWF+ nO*nI); w++)
		*(_weights +w) = dis(*gen) / Real(nO + nI);
	orthogonalize(iWF, _weights);

	for (int w=iWO; w<(iWO+ nO*nI); w++)
		*(_weights +w) = dis(*gen) / Real(nO + nI);
	orthogonalize(iWO, _weights);
}

void Graph::initializeWeights(mt19937* const gen, Real* const _weights, Real* const _biases) const
{
	uniform_real_distribution<Real> dis(-sqrt(6.),sqrt(6.));

	for (const auto & l : *(nl_inputs_vec))
		l->initialize(dis, gen, _weights);

	if(nl_recurrent not_eq nullptr)
		nl_recurrent->initialize(dis, gen, _weights);

	for (const auto & l : *(rl_inputs_vec))
		l->initialize(dis, gen, _weights);

	if(rl_recurrent not_eq nullptr)
		rl_recurrent->initialize(dis, gen, _weights);

	if (not last) //no bias on output layer
			for (int w=biasHL; w<biasHL+normalSize; w++)
				*(_biases +w) = dis(*gen) / Real(normalSize);

	if (not last)
		for (int w=biasIN; w<biasIN+recurrSize; w++)
			*(_biases +w) = dis(*gen) / Real(recurrSize);

	for (int w=biasIG; w<biasIG+recurrSize; w++)
		*(_biases +w) = dis(*gen) / Real(recurrSize) + 1.0;

	for (int w=biasFG; w<biasFG+recurrSize; w++)
		*(_biases +w) = dis(*gen) / Real(recurrSize) + 1.0;

	for (int w=biasOG; w<biasOG+recurrSize; w++)
		*(_biases +w) = dis(*gen) / Real(recurrSize) + 1.0;
}
