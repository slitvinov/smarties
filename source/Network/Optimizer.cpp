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
eta(_s.learnrate), lambda(_s.nnLambda) { }

AdamOptimizer::AdamOptimizer(Network*const _net, Profiler*const _prof,
		Settings& _s, const Real B1, const Real B2) : Optimizer(_net, _prof, _s),
		beta_1(B1), beta_2(B2), epsilon(1e-8), beta_t_1(B1), beta_t_2(B2),
		_2ndMomW(initClean(nWeights)), _2ndMomB(initClean(nBiases)) { }
//beta_1(0.9), beta_2(0.999), epsilon(1e-8), beta_t_1(0.9), beta_t_2(0.99)

EntropySGD::EntropySGD(Network*const _net, Profiler*const _prof, Settings&_s) :
		AdamOptimizer(_net, _prof, _s, 0.8), alpha_eSGD(0.75), gamma_eSGD(1.),
		eta_eSGD(1./_s.targetDelay), eps_eSGD(1e-6), L_eSGD(_s.targetDelay),
		_muW_eSGD(initClean(nWeights)), _muB_eSGD(initClean(nBiases))
{
	assert(L_eSGD>0);
	for (Uint i=0; i<nWeights; i++) _muW_eSGD[i] = net->weights[i];
	for (Uint i=0; i<nBiases; i++)  _muB_eSGD[i] = net->biases[i];
}

void Optimizer::moveFrozenWeights(const Real _alpha)
{
	if (net->allocatedFrozenWeights==false || _alpha>1)
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
}

void EntropySGD::moveFrozenWeights(const Real _alpha)
{
	assert(_alpha>1);

#pragma omp parallel
	{
		const nnReal fac = eta_eSGD * gamma_eSGD;

#pragma omp for nowait
		for (Uint j=0; j<nWeights; j++) {
			net->tgt_weights[j] += fac * (_muW_eSGD[j] - net->tgt_weights[j]);
			net->weights[j] = net->tgt_weights[j];
			_muW_eSGD[j] = net->tgt_weights[j];
		}

#pragma omp for nowait
		for (Uint j=0; j<nBiases; j++){
			net->tgt_biases[j] += fac * (_muB_eSGD[j] - net->tgt_biases[j]);
			net->biases[j] = net->tgt_biases[j];
			_muB_eSGD[j] = net->tgt_biases[j];
		}
	}
}

void EntropySGD::update(nnReal*const dest,const nnReal*const target,
	nnReal*const grad, nnReal*const _1stMom, nnReal*const _2ndMom,
	nnReal*const _mu, const Uint N, const Uint batchsize, const Real _eta)
{
	//const Real fac_ = std::sqrt(1.-beta_t_2)/(1.-beta_t_1);
	assert(batchsize>0);
	const nnReal eta_ = _eta*std::sqrt(1.-beta_t_2)/(1.-beta_t_1);
	const nnReal norm = 1./batchsize;
	// TODO const Real lambda_ = _lambda*eta_;
	const nnReal noise = std::sqrt(eta_) * eps_eSGD;

#pragma omp parallel
	{
		const Uint thrID = static_cast<Uint>(omp_get_thread_num());
		Saru gen(nepoch, thrID, net->generators[thrID]());

#pragma omp for
		for (Uint i=0; i<N; i++)
		{
			const nnReal DW  = grad[i]*norm;
			const nnReal M1_ = beta_1* _1stMom[i] +(1.-beta_1) *DW;
			const nnReal M2  = beta_2* _2ndMom[i] +(1.-beta_2) *DW*DW;
			const nnReal M2_ = std::max(M2, (nnReal)epsilon);

			const nnReal RNG = noise * gen.d_mean0_var1();
			const nnReal DW_ = eta_*M1_/std::sqrt(M2_);

			_1stMom[i] = M1_;
			_2ndMom[i] = M2_;
			grad[i] = 0.; //reset grads

			dest[i] += DW_ + RNG + eta_*gamma_eSGD*(target[i]-dest[i]);
			_mu[i]  += alpha_eSGD*(dest[i] - _mu[i]);
		}
	}

}

void EntropySGD::update(Grads* const G, const Uint batchsize)
{
	//const Real _eta = eta/(1.+std::log(1. + (double)nepoch));
	update(net->weights,net->tgt_weights,G->_W,_1stMomW,_2ndMomW,_muW_eSGD,nWeights,batchsize,eta);
	update(net->biases, net->tgt_biases, G->_B,_1stMomB,_2ndMomB,_muB_eSGD,nBiases, batchsize,eta);

	beta_t_1 *= beta_1;
	if (beta_t_1<2.2e-16) beta_t_1 = 0;

	beta_t_2 *= beta_2;
	if (beta_t_2<2.2e-16) beta_t_2 = 0;

	if(lambda>2.2e-16) net->regularize(lambda*eta);
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
			for (Uint k=0; k<nThreads; k++) {
				G->_W[j] += g[k]->_W[j];
				g[k]->_W[j] = 0.;
			}

#pragma omp for nowait
		for (Uint j=0; j<nBiases; j++)
			for (Uint k=0; k<nThreads; k++) {
				G->_B[j] += g[k]->_B[j];
				g[k]->_B[j] = 0.;
			}
	}
}

void Optimizer::update(Grads* const G, const Uint batchsize)
{
	update(net->weights, G->_W, _1stMomW, nWeights, batchsize);
	update(net->biases,  G->_B, _1stMomB, nBiases, batchsize);
	if(lambda>2.2e-16) net->regularize(lambda*eta);
}

void AdamOptimizer::update(Grads* const G, const Uint batchsize)
{
	const Real _eta = eta/(1.+(Real)nepoch/1e5);

	update(net->weights,G->_W,_1stMomW,_2ndMomW,nWeights,batchsize,_eta);
	update(net->biases, G->_B,_1stMomB,_2ndMomB,nBiases, batchsize,_eta);

	beta_t_1 *= beta_1;
	if (beta_t_1<2.2e-16) beta_t_1 = 0;
	beta_t_2 *= beta_2;
	if (beta_t_2<2.2e-16) beta_t_2 = 0;

	if(lambda>2.2e-16) net->regularize(lambda*_eta);
}

void Optimizer::update(nnReal*const dest,nnReal*const grad,nnReal*const _1stMom,
		const Uint N, const Uint batchsize) const
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

#if 1
void AdamOptimizer::update(nnReal*const dest, nnReal*const grad,
		nnReal*const _1stMom, nnReal*const _2ndMom,
		const Uint N, const Uint batchsize, const Real _eta)
{
	assert(batchsize>0);
	const nnReal eta_ = _eta*std::sqrt(beta_2-beta_t_2)/(1.-beta_t_1);
	const nnReal eps = std::numeric_limits<nnReal>::epsilon();
	const nnReal norm = 1./batchsize;
	const nnReal f11=beta_1, f12=1-beta_1, f21=beta_2, f22=1-beta_2;
#pragma omp parallel for
	for (Uint i=0; i<N; i++) {
		const nnReal DW  = grad[i]*norm;
		const nnReal M1  = f11* _1stMom[i] +f12* DW;
		const nnReal M2  = f21* _2ndMom[i] +f22* DW*DW;
		const nnReal M2_ = std::max(M2, eps);
		const nnReal _M2 = std::sqrt(M2_);
		const nnReal M1_ = std::max(std::min(M1, _M2), -_M2); //grad clip -1:1
		_1stMom[i] = M1_;
		_2ndMom[i] = M2_;
		grad[i] = 0.; //reset grads

		//dest[i] += eta_*M1_/_M2; //Adam
		dest[i] += eta_*(f12*DW + f11*M1_)/_M2; //Nesterov Adam
	}
}
#else // Adamax:
void AdamOptimizer::update(Real* const dest, Real* const grad,
		Real* const _1stMom, Real* const _2ndMom,
		const Uint N, const Uint batchsize, const Real eta_)
{
	assert(batchsize>0);
	const Real eps = std::numeric_limits<Real>::epsilon();
	const Real norm = 1./batchsize;
#pragma omp parallel for
	for (Uint i=0; i<N; i++) {
		const Real DW  = grad[i]*norm;
		const Real M1  = beta_1* _1stMom[i] +(1.-beta_1) *DW;
		const Real M2  = std::max(beta_2*_2ndMom[i], std::fabs(DW));
		const Real M2_ = std::max(M2,eps);
		const Real M1_ = M1;
		//dest[i] += eta_*M1_/M2_;
		dest[i] += eta_*((1-beta_1)*DW + beta_1*M1_)/M2_; //nesterov
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

	printf("Saving into %s\n", fname.c_str());
	fflush(0);
	string nameBackup = fname + "_net_tmp";
	ofstream out(nameBackup.c_str());
	if (!out.good()) _die("Unable to open save into file %s\n", fname.c_str());

	vector<nnReal> outWeights, outBiases, outMomW, outMomB;
	outWeights.reserve(nWeights); outMomW.reserve(nWeights);
	outBiases.reserve(nBiases); outMomB.reserve(nBiases);
	out.precision(20);

	net->save(outWeights, outBiases, net->weights, net->biases);
	net->save(outMomW, outMomB, _1stMomW, _1stMomB);

	out<<outWeights.size()<<" "<<outBiases.size()<<" "<<nLayers<<" "<<nNeurons<<endl;
	assert(outWeights.size() == outMomW.size());
	assert(outBiases.size() == outMomB.size());
	for(Uint i=0;i<outMomW.size();i++) out<<outWeights[i]<<" "<<outMomW[i]<<"\n";
	for(Uint i=0;i<outMomB.size();i++) out<<outBiases[i] <<" "<<outMomB[i]<<"\n";
	out.flush();
	out.close();
	string command = "cp " + nameBackup + " " + fname + "_net";
	system(command.c_str());

	save_recurrent_connections(fname);
}

bool EntropySGD::restart(const string fname)
{
	const bool ret = AdamOptimizer::restart(fname);
	if (!ret) return ret;
	for (Uint i=0; i<nWeights; i++) _muW_eSGD[i] = net->weights[i];
	for (Uint i=0; i<nBiases; i++)  _muB_eSGD[i] = net->biases[i];
	return ret;
}

void AdamOptimizer::save(const string fname)
{
	const Uint nNeurons(net->getnNeurons()), nLayers(net->getnLayers());
	//const Uint nAgents(net->getnAgents()), nStates(net->getnStates());

	printf("Saving into %s\n", fname.c_str());
	fflush(0);
	string nameBackup = fname + "_net_tmp";
	ofstream out(nameBackup.c_str());
	if (!out.good()) _die("Unable to open save into file %s\n", fname.c_str());

	vector<nnReal> outWeights, outBiases, out1MomW, out1MomB, out2MomW, out2MomB;
	outWeights.reserve(nWeights); out1MomW.reserve(nWeights); out2MomW.reserve(nWeights);
	outBiases.reserve(nBiases);   out1MomB.reserve(nBiases);  out2MomB.reserve(nBiases);
	out.precision(20);

	net->save(outWeights, outBiases, net->weights, net->biases);
	net->save(out1MomW,   out1MomB,  _1stMomW,     _1stMomB);
	net->save(out2MomW,   out2MomB,  _2ndMomW,     _2ndMomB);

	out<<outWeights.size()<<" "<<outBiases.size()<<" "<<nLayers<<" "<<nNeurons<<endl;
	assert(outWeights.size() == out1MomW.size() && outWeights.size() == out2MomW.size());
	assert(outBiases.size() == out1MomB.size() && outBiases.size() == out2MomB.size());

	for(Uint i=0;i<outWeights.size();i++)
		out<<outWeights[i]<<" "<<out1MomW[i]<<" "<<out2MomW[i]<<"\n";
	for(Uint i=0;i<outBiases.size();i++)
		out<<outBiases[i] <<" "<<out1MomB[i]<<" "<<out2MomB[i]<<"\n";
	out.flush();
	out.close();
	string command = "cp " + nameBackup + " " + fname + "_net";
	system(command.c_str());

	save_recurrent_connections(fname);
}

bool Optimizer::restart(const string fname)
{
	const Uint nNeurons(net->getnNeurons()), nLayers(net->getnLayers());
	//const Uint nAgents(net->getnAgents()); // , nStates(net->getnStates()) TODO

	string nameBackup = fname + "_net";
	ifstream in(nameBackup.c_str());
	debugN("Reading from %s\n", nameBackup.c_str());
	if (!in.good())
	{
		error("Couldnt open file %s \n", nameBackup.c_str());
#ifndef NDEBUG //if debug, you might want to do this
		if(!bTrain) {die("...and I'm not training\n");}
#endif
		return false;
	}

	Uint readTotWeights, readTotBiases, readNNeurons, readNLayers;
	in >> readTotWeights  >> readTotBiases >> readNLayers >> readNNeurons;
	if (readNLayers != nLayers || readNNeurons != nNeurons)
		die("Network parameters differ!");
	//readTotWeights != nWeights || readTotBiases != nBiases || TODO

	vector<nnReal> outWeights, outBiases, outMomW, outMomB;
	outWeights.resize(nWeights); outMomW.resize(nWeights);
	outBiases.resize(nBiases); outMomB.resize(nBiases);
	for (Uint i=0;i<readTotWeights;i++)
		in >> outWeights[i] >> outMomW[i];
	for (Uint i=0;i<readTotBiases; i++)
		in >> outBiases[i]  >> outMomB[i];

	net->restart(outWeights, outBiases, net->weights, net->biases);
	net->restart(outMomW, outMomB, _1stMomW, _1stMomB);
	in.close();
	net->updateFrozenWeights();
	return restart_recurrent_connections(fname);
}

bool AdamOptimizer::restart(const string fname)
{
	//const Uint nNeurons(net->getnNeurons()), nLayers(net->getnLayers());

	string nameBackup = fname + "_net";
	ifstream in(nameBackup.c_str());
	debugN("Reading from %s\n", nameBackup.c_str());
	if (!in.good())
	{
		error("Couldnt open file %s \n", nameBackup.c_str());
#ifndef NDEBUG //if debug, you might want to do this
		if(!bTrain) {die("...and I'm not training\n");}
#endif
		return false;
	}

	Uint readTotWeights, readTotBiases, readNNeurons, readNLayers;
	in >> readTotWeights  >> readTotBiases >> readNLayers >> readNNeurons;
//	if (readNLayers != nLayers || readNNeurons != nNeurons)
//		die("Network parameters differ!");
	//readTotWeights != nWeights || readTotBiases != nBiases || TODO

	vector<nnReal> outWeights, outBiases, out1MomW, out1MomB, out2MomW, out2MomB;
	outWeights.resize(nWeights); out1MomW.resize(nWeights); out2MomW.resize(nWeights);
	outBiases.resize(nBiases);   out1MomB.resize(nBiases);  out2MomB.resize(nBiases);

	for (Uint i=0;i<readTotWeights;i++)
		in >> outWeights[i] >> out1MomW[i] >> out2MomW[i];
	for (Uint i=0;i<readTotBiases; i++)
		in >> outBiases[i]  >> out1MomB[i] >> out2MomB[i];

	net->restart(outWeights, outBiases, net->weights, net->biases);
	net->restart(out1MomW,   out1MomB,  _1stMomW,     _1stMomB);
	net->restart(out2MomW,   out2MomB,  _2ndMomW,     _2ndMomB);
	in.close();
	net->updateFrozenWeights();
	return restart_recurrent_connections(fname);
}
