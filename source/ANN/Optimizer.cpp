/*
 *  Layers.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include "Optimizer.h"
#include <iomanip>      // std::setprecision
#include <iostream>     // std::cout, std::fixed
#include <cassert>
#include "saruprng.h"

using namespace ErrorHandling;

Optimizer::Optimizer(Network* const _net, Profiler* const _prof,
  Settings& settings) : nWeights(_net->getnWeights()),
  nBiases(_net->getnBiases()), bTrain(settings.bTrain), net(_net), profiler(_prof),
eta(settings.lRate), lambda(settings.nnLambda), alpha(0.5), nepoch(0)
{
    _allocateClean(_1stMomW, nWeights)
    _allocateClean(_1stMomB, nBiases)
}

AdamOptimizer::AdamOptimizer(Network* const _net, Profiler* const _prof,
  Settings& settings) : Optimizer(_net, _prof, settings),
  beta_1(0.9), beta_2(0.999), epsilon(1e-8), beta_t_1(0.9), beta_t_2(0.999)
{
    _allocateClean(_2ndMomW, nWeights)
    _allocateClean(_2ndMomB, nBiases)
}

EntropySGD::EntropySGD(Network* const _net, Profiler* const _prof,
  Settings& settings) : AdamOptimizer(_net, _prof, settings), alpha_eSGD(0.75),
  gamma_eSGD(0.1), eta_eSGD(0.1/settings.dqnUpdateC), eps_eSGD(1e-4), L_eSGD(settings.dqnUpdateC)
{
    assert(L_eSGD>0);
    _allocateClean(_muW_eSGD, nWeights)
    _allocateClean(_muB_eSGD, nBiases)

	for (int i=0; i<nWeights; i++) _muW_eSGD[i] = net->weights[i];
  for (int i=0; i<nBiases; i++)  _muB_eSGD[i] = net->biases[i];
}

void Optimizer::moveFrozenWeights(const Real _alpha)
{
  if (net->allocatedFrozenWeights==false || _alpha>1)
      return net->updateFrozenWeights();

  #pragma omp parallel
  {
      #pragma omp for nowait
      for (int j=0; j<nWeights; j++)
          net->tgt_weights[j] += _alpha*(net->weights[j] - net->tgt_weights[j]);

      #pragma omp for nowait
      for (int j=0; j<nBiases; j++)
          net->tgt_biases[j] += _alpha*(net->biases[j] - net->tgt_biases[j]);
  }
}

void EntropySGD::moveFrozenWeights(const Real _alpha)
{
    assert(_alpha>1);

    if (net->allocatedFrozenWeights==false) return net->updateFrozenWeights();

    #pragma omp parallel
    {
        const Real fac = eta_eSGD * gamma_eSGD;

        #pragma omp for nowait
        for (int j=0; j<nWeights; j++) {
          net->tgt_weights[j] += fac * (_muW_eSGD[j] - net->tgt_weights[j]);
          net->weights[j] = net->tgt_weights[j];
          _muW_eSGD[j] = net->tgt_weights[j];
        }

        #pragma omp for nowait
        for (int j=0; j<nBiases; j++){
          net->tgt_biases[j] += fac * (_muB_eSGD[j] - net->tgt_biases[j]);
          net->biases[j] = net->tgt_biases[j];
          _muB_eSGD[j] = net->tgt_biases[j];
        }
    }
}

void EntropySGD::update(Real* const dest, const Real* const target, Real* const grad,
  Real* const _1stMom, Real* const _2ndMom, Real* const _mu, const int N,
  const int batchsize, const Real _lambda, const Real _eta)
{
    //const Real fac_ = std::sqrt(1.-beta_t_2)/(1.-beta_t_1);
    const Real eta_ = _eta*std::sqrt(1.-beta_t_2)/(1.-beta_t_1);
    const Real norm = 1./(Real)max(batchsize,1);
    // TODO const Real lambda_ = _lambda*eta_;

	#pragma omp parallel
  {
    const int thrID = omp_get_thread_num();
    Saru gen(nepoch, thrID, net->generators[thrID]());

    #pragma omp for
    for (int i=0; i<N; i++)
    {
        const Real DW  = grad[i]*norm;
        const Real M1_ = beta_1* _1stMom[i] +(1.-beta_1) *DW;
        const Real M2  = beta_2* _2ndMom[i] +(1.-beta_2) *DW*DW;
        const Real M2_ = std::max(M2,epsilon);

        const Real RNG = std::sqrt(eta_) * eps_eSGD * gen.d_mean0_var1();
        const Real DW_ = eta_*M1_/std::sqrt(M2_);

        _1stMom[i] = M1_;
        _2ndMom[i] = M2_;
        grad[i] = 0.; //reset grads

        dest[i] += DW_ + RNG + eta_*gamma_eSGD*(target[i]-dest[i]);
        _mu[i]  += alpha_eSGD*(dest[i] - _mu[i]);
    }
  }

}

void EntropySGD::update(Grads* const G, const int batchsize)
{
  //const Real _eta = eta/(1.+std::log(1. + (double)nepoch));
  update(net->weights,net->tgt_weights,G->_W,_1stMomW,_2ndMomW,_muW_eSGD,nWeights,batchsize,lambda,eta);
  update(net->biases, net->tgt_biases, G->_B,_1stMomB,_2ndMomB,_muB_eSGD,nBiases, batchsize,     0,eta);
  //Optimizer::update(net->weights, G->_W, _1stMomW, nWeights, batchsize, lambda);
  //Optimizer::update(net->biases,  G->_B, _1stMomB, nBiases, batchsize);
	beta_t_1 *= beta_1;
  if (beta_t_1<2.2e-16) beta_t_1 = 0;

	beta_t_2 *= beta_2;
  if (beta_t_2<2.2e-16) beta_t_2 = 0;
	//printf("%d %f %f\n",nepoch, beta_t_1,beta_t_2);
}

void Optimizer::stackGrads(Grads* const G, const Grads* const g) const
{
    for (int j=0; j<nWeights; j++) G->_W[j] += g->_W[j];
    for (int j=0; j<nBiases; j++)  G->_B[j] += g->_B[j];
}

void Optimizer::stackGrads(Grads* const G, const vector<Grads*> g) const
{
    const int nThreads = g.size();
    #pragma omp parallel
    {
		    #pragma omp for nowait
        for (int j=0; j<nWeights; j++)
        for (int k=0; k<nThreads; k++) {
            //G->_W[j] += std::max(std::min(g[k]->_W[j], 10.), -10.);
            G->_W[j] += g[k]->_W[j];
            g[k]->_W[j] = 0.;
        }

        #pragma omp for nowait
        for (int j=0; j<nBiases; j++)
        for (int k=0; k<nThreads; k++) {
            //G->_B[j] += std::max(std::min(g[k]->_B[j], 10.), -10.);
            G->_B[j] += g[k]->_B[j];
            g[k]->_B[j] = 0.;
        }
    }
}

void Optimizer::update(Grads* const G, const int batchsize)
{
    update(net->weights, G->_W, _1stMomW, nWeights, batchsize, lambda);
    update(net->biases,  G->_B, _1stMomB, nBiases, batchsize);
}

void AdamOptimizer::update(Grads* const G, const int batchsize)
{
  //const Real _eta = eta/(1.+std::log(1. + (double)nepoch));
  const Real _eta = eta/(1.+(Real)nepoch/1e4);
  update(net->weights,G->_W,_1stMomW,_2ndMomW,nWeights,batchsize,lambda,_eta);
  update(net->biases, G->_B,_1stMomB,_2ndMomB,nBiases, batchsize,     0,_eta);
  //Optimizer::update(net->weights, G->_W, _1stMomW, nWeights, batchsize, lambda);
  //Optimizer::update(net->biases,  G->_B, _1stMomB, nBiases, batchsize);
	beta_t_1 *= beta_1;
  if (beta_t_1<2.2e-16) beta_t_1 = 0;
	beta_t_2 *= beta_2;
  if (beta_t_2<2.2e-16) beta_t_2 = 0;
	//printf("%d %f %f\n",nepoch, beta_t_1,beta_t_2);
}

void Optimizer::update(Real* const dest, Real* const grad, Real* const _1stMom,
                    const int N, const int batchsize, const Real _lambda) const
{
    const Real norm = 1./(Real)max(batchsize,1);
    //const Real eta_ = eta*norm/std::log((double)nepoch/1.);
    const Real eta_ = eta*norm/(1.+std::log(1. + (double)nepoch/1e3));
    const Real lambda_ = _lambda*eta;

    #pragma omp parallel for
    for (int i=0; i<N; i++) {
        //const Real W = fabs(dest[i]);
        const Real M1 = alpha * _1stMom[i] + eta_ * grad[i];
        _1stMom[i] = std::max(std::min(M1,eta_),-eta_);
        grad[i] = 0.; //reset grads

        if (lambda_>0)
             //dest[i] += _1stMom[i] + (dest[i]<0 ? lambda_ : -lambda_);
             dest[i] += _1stMom[i] - dest[i]*lambda_;
        else dest[i] += _1stMom[i];
    }
}

#if 1
void AdamOptimizer::update(Real* const dest, Real* const grad,
                           Real* const _1stMom, Real* const _2ndMom,
                           const int N, const int batchsize,
						   const Real _lambda, const Real _eta)
{
    //const Real fac_ = std::sqrt(1.-beta_t_2)/(1.-beta_t_1);
    const Real eta_ = _eta*std::sqrt(1.-beta_t_2)/(1.-beta_t_1);
    const Real norm = 1./(Real)max(batchsize,1);
    const Real lambda_ = _lambda*eta_;
    const Real eps = std::numeric_limits<Real>::epsilon();
	#pragma omp parallel for
    for (int i=0; i<N; i++) {
        //const Real DW  = std::max(std::min(grad[i]*norm, 1.), -1.);
        const Real DW  = grad[i]*norm;
        const Real M1  = beta_1* _1stMom[i] +(1.-beta_1) *DW;
        const Real M2  = beta_2* _2ndMom[i] +(1.-beta_2) *DW*DW;
        //const Real DW_ = std::max(std::min(eta_*M1_/std::sqrt(M2_),eta_),-eta_);
        const Real M2_ = std::max(M2, eps);
        _1stMom[i] = M1;
        _2ndMom[i] = M2_;
        grad[i] = 0.; //reset grads

        //dest[i] += eta_*M1_/std::sqrt(M2_); 
        dest[i] += eta_*((1-beta_1)*DW + beta_1*M1)/std::sqrt(M2_); 
    }
}
#else
void AdamOptimizer::update(Real* const dest, Real* const grad,
                           Real* const _1stMom, Real* const _2ndMom,
                           const int N, const int batchsize,
						   const Real _lambda, const Real _eta)
{
    //const Real fac_ = std::sqrt(1.-beta_t_2)/(1.-beta_t_1);
    const Real eta_ = _eta/(1.-beta_t_1);
    const Real norm = 1./(Real)max(batchsize,1);
    const Real lambda_ = _lambda*_eta;
    const Real eps = std::numeric_limits<Real>::epsilon();
    #pragma omp parallel for
    for (int i=0; i<N; i++) {
        //const Real scale = std::max(1.,std::fabs(dest[i]));
        //const Real DW  = std::max(std::min(grad[i]*norm, 1.), -1.);
        const Real DW  = grad[i]*norm;
        const Real M1  = beta_1* _1stMom[i] +(1.-beta_1) *DW;
        const Real M2_ = std::max(beta_2*_2ndMom[i]+eps, std::fabs(DW));
        //const Real M1_ = std::max(std::min(M1,M2_),-M2_);
        const Real M1_ = M1;
        const Real DW_ = eta_*M1_/M2_;
        _1stMom[i] = M1_;
        _2ndMom[i] = M2_;
        grad[i] = 0.; //reset grads

        if (lambda_>0)
             dest[i] += DW_ + (dest[i]<0 ? lambda_ : -lambda_);   // L1
             //dest[i] += DW_ - dest[i]*lambda_;                      // L2
        else dest[i] += DW_;
    }
}
#endif

void Optimizer::init(Real* const dest, const int N, const Real ini)
{
    for (int j=0; j<N; j++) dest[j] = ini;
}

void Optimizer::save(const string fname)
{
  const int nNeurons(net->getnNeurons()), nLayers(net->getnLayers());
  const int nAgents(net->getnAgents()), nStates(net->getnStates());

  {
    printf("Saving into %s\n", fname.c_str());
    fflush(0);
    string nameBackup = fname + "_net_tmp";
    ofstream out(nameBackup.c_str());

    if (!out.good()) die("Unable to open save into file %s\n", fname.c_str());

    out.precision(20);
    out << nWeights << " " << nBiases << " " << nLayers  << " " << nNeurons << endl;

    for (int i=0; i<nWeights; i++) {
        if (std::isnan(net->weights[i]) || std::isinf(net->weights[i]))
            die("Caught a nan\n")
        else
            out << net->weights[i] <<" "<< _1stMomW[i] << "\n";
    }

    for (int i=0; i<nBiases; i++) {
      if (std::isnan(net->biases[i]) || std::isinf(net->biases[i]))
            die("Caught a nan\n")
        else
            out << net->biases[i] <<" "<< _1stMomB[i] << "\n";
    }

    out.flush();
    out.close();
    string command = "cp " + nameBackup + " " + fname + "_net";
    system(command.c_str());
  }
  {
    string nameBackup = fname + "_mems_tmp";
    ofstream out(nameBackup.c_str());

    if (!out.good())
      die("Unable to open save into file %s\n", nameBackup.c_str());

    for(int agentID=0; agentID<nAgents; agentID++) {
      for (int j=0; j<nNeurons; j++) out << net->mem[agentID]->outvals[j] << "\n";
      for (int j=0; j<nStates;  j++) out << net->mem[agentID]->ostates[j] << "\n";
    }

    out.flush();
    out.close();
    string command = "cp " + nameBackup + " " + fname + "_mems";
    system(command.c_str());
  }
}

bool EntropySGD::restart(const string fname)
{
  const bool ret = AdamOptimizer::restart(fname);
  if (!ret) return ret; 
  for (int i=0; i<nWeights; i++) _muW_eSGD[i] = net->weights[i];
  for (int i=0; i<nBiases; i++)  _muB_eSGD[i] = net->biases[i];
  return ret;
}

void AdamOptimizer::save(const string fname)
{
  const int nNeurons(net->getnNeurons()), nLayers(net->getnLayers());
  const int nAgents(net->getnAgents()), nStates(net->getnStates());

  {
    printf("Saving into %s\n", fname.c_str());
    fflush(0);
    string nameBackup = fname + "_net_tmp";
    ofstream out(nameBackup.c_str());

    if (!out.good()) die("Unable to open save into file %s\n", fname.c_str());

    out.precision(20);
    out << nWeights << " " << nBiases << " " << nLayers  << " " << nNeurons << endl;

    for (int i=0; i<nWeights; i++) {
        if (std::isnan(net->weights[i]) || std::isinf(net->weights[i]))
            die("Caught a nan\n")
        else
            out<<net->weights[i]<<" "<<_1stMomW[i]<<" "<<_2ndMomW[i]<<"\n";
    }

    for (int i=0; i<nBiases; i++) {
      if (std::isnan(net->biases[i]) || std::isinf(net->biases[i]))
            die("Caught a nan\n")
        else
            out<<net->biases[i]<<" "<<_1stMomB[i]<<" "<<_2ndMomB[i]<<"\n";
    }

    out.flush();
    out.close();
    string command = "cp " + nameBackup + " " + fname + "_net";
    system(command.c_str());
  }
  {
    string nameBackup = fname + "_mems_tmp";
    ofstream out(nameBackup.c_str());

    if (!out.good())
      die("Unable to open save into file %s\n", nameBackup.c_str());

    for(int agentID=0; agentID<nAgents; agentID++) {
      for (int j=0; j<nNeurons; j++) out << net->mem[agentID]->outvals[j] << "\n";
      for (int j=0; j<nStates;  j++) out << net->mem[agentID]->ostates[j] << "\n";
    }

    out.flush();
    out.close();
    string command = "cp " + nameBackup + " " + fname + "_mems";
    system(command.c_str());
  }
}

bool Optimizer::restart(const string fname)
{
  const int nNeurons(net->getnNeurons()), nLayers(net->getnLayers());
  const int nAgents(net->getnAgents()), nStates(net->getnStates());

  {
    string nameBackup = fname + "_net";
    ifstream in(nameBackup.c_str());
    debug1("Reading from %s\n", nameBackup.c_str());
    if (!in.good()) {
        error("Couldnt open file %s \n", nameBackup.c_str());
        #ifndef NDEBUG //if debug, you might want to do this
        if(!bTrain) {die("...and I'm not training\n");}
        #endif
        return false;
    }

    int readTotWeights, readTotBiases, readNNeurons, readNLayers;
    in >> readTotWeights  >> readTotBiases >> readNLayers >> readNNeurons;

    if (readTotWeights != nWeights || readTotBiases != nBiases || readNLayers != nLayers || readNNeurons != nNeurons)
    die("Network parameters differ!");

    Real tmp, tmp1;
    for (int i=0; i<nWeights; i++) {
        in >> tmp >> tmp1;
        if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
        net->weights[i] = tmp;
        _1stMomW[i] = tmp1;
    }

    for (int i=0; i<nBiases; i++) {
        in >> tmp >> tmp1;
        if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
        net->biases[i] = tmp;
        _1stMomB[i] = tmp1;
    }
    in.close();
    net->updateFrozenWeights();
  }
  {
    string nameBackup = fname + "_mems";
    ifstream in(nameBackup.c_str());
    debug1("Reading from %s\n", nameBackup.c_str());
    if (!in.good()) {
        error("Couldnt open file %s \n", nameBackup.c_str());
        return false;
    }

    Real tmp;
    for(int agentID=0; agentID<nAgents; agentID++) {
      for (int j=0; j<nNeurons; j++) {
        in >> tmp;
        if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
        net->mem[agentID]->outvals[j] = tmp;
      }
      for (int j=0; j<nStates; j++) {
        in >> tmp;
        if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
        net->mem[agentID]->ostates[j] = tmp;
      }
    }
    in.close();
  }
  return true;
}


bool AdamOptimizer::restart(const string fname)
{
  const int nNeurons(net->getnNeurons()), nLayers(net->getnLayers());
  const int nAgents(net->getnAgents()), nStates(net->getnStates());

  {
    string nameBackup = fname + "_net";
    ifstream in(nameBackup.c_str());
    debug1("Reading from %s\n", nameBackup.c_str());
    if (!in.good()) {
        error("Couldnt open file %s \n", nameBackup.c_str());
        #ifndef NDEBUG //if debug, you might want to do this
        if(!bTrain) {die("...and I'm not training\n");}
        #endif
        return false;
    }

    int readTotWeights, readTotBiases, readNNeurons, readNLayers;
    in >> readTotWeights  >> readTotBiases >> readNLayers >> readNNeurons;

    if (readTotWeights != nWeights || readTotBiases != nBiases ||
           readNLayers != nLayers  || readNNeurons  != nNeurons )
    die("Network parameters differ!");

    Real tmp, tmp1, tmp2;
    for (int i=0; i<nWeights; i++) {
        in >> tmp >> tmp1 >> tmp2;
        if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
        net->weights[i] = tmp;
        _1stMomW[i] = tmp1;
        _2ndMomW[i] = tmp2;
    }

    for (int i=0; i<nBiases; i++) {
        in >> tmp >> tmp1 >> tmp2;
        if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
        net->biases[i] = tmp;
        _1stMomB[i] = tmp1;
        _2ndMomB[i] = tmp2;
    }
    in.close();
    net->updateFrozenWeights();
  }
  {
    string nameBackup = fname + "_mems";
    ifstream in(nameBackup.c_str());
    debug1("Reading from %s\n", nameBackup.c_str());
    if (!in.good()) {
        error("Couldnt open file %s \n", nameBackup.c_str());
        return false;
    }

    Real tmp;
    for(int agentID=0; agentID<nAgents; agentID++) {
      for (int j=0; j<nNeurons; j++) {
        in >> tmp;
        if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
        net->mem[agentID]->outvals[j] = tmp;
      }
      for (int j=0; j<nStates; j++) {
        in >> tmp;
        if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
        net->mem[agentID]->ostates[j] = tmp;
      }
    }
    in.close();
  }
  return true;
}

/*
LMOptimizer::LMOptimizer(Network * _net, Profiler * _prof, Settings  & settings) : muMax(1e10), muMin(1e-6), muFactor(10), net(_net), profiler(_prof), nInputs(net->nInputs), nOutputs(net->nOutputs), iOutputs(net->iOutputs), nWeights(net->nWeights), nBiases(net->nBiases), totWeights(net->nWeights+net->nBiases), mu(0.1)
{
    dw.set_size(totWeights);
    Je.set_size(totWeights);
    diagJtJ.eye(totWeights, totWeights);
}

void LMOptimizer::stackGrads(Grads * g, const int k, const int i)
{
    #pragma omp parallel for nowait
    for (int j=0; j<nWeights; j++)
        J(i + k*nOutputs, j) = -*(g->_W + j);

    #pragma omp parallel for
    for (int j=0; j<nBiases; j++)
        J(i + k*nOutputs, j+nWeights) = -*(g->_B + j);
}

void LMOptimizer::tryNew()
{
    #pragma omp parallel for nowait
    for (int j=0; j<nWeights; j++)
        *(net->weights +j) += dw(j);

    #pragma omp parallel for
    for (int j=0; j<nBiases; j++)
        *(net->biases +j) += dw(j+nWeights);
}

void LMOptimizer::goBack()
{
    #pragma omp parallel for nowait
    for (int j=0; j<nWeights; j++)
        *(net->weights +j) -= dw(j);

    #pragma omp parallel for
    for (int j=0; j<nBiases; j++)
        *(net->biases +j) -= dw(j+nWeights);
}

void LMOptimizer::trainSeries(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE)
{
    trainMSE = 0.0;
    vector<Real> res;
    int nseries = inputs.size();
    net->allocateSeries(nseries+1);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);

    J.set_size(nOutputs*nseries, totWeights);
    e.set_size(nOutputs*nseries);

    #pragma omp parallel
    {
        //STEP 1: go through the data to compute predictions
        #pragma omp master
            profiler->start("F");

        for (int k=0; k<nseries; k++)
        {
            net->predict(inputs[k], res, net->series[k], net->series[k+1]);

            #pragma omp master
            for (int i=0; i<nOutputs; i++)
            { //put this loop here to slightly reduce overhead on second step
                Real err = *(net->series[k+1]->outvals+iOutputs+i) - targets[k][i];
                e(i + k*nOutputs) = err;
                *(net->series[k+1]->errvals +iOutputs+i) = 0.0;
                trainMSE += err*err;
            }
        }

        #pragma omp master
            profiler->stop("F");

        //STEP 2: go backwards to backpropagate deltas (errors)
        #pragma omp master
            profiler->start("B");

        net->clearErrors(net->series[nseries+1]); //there is a omp for in here
        for (int i=0; i<nOutputs; i++)
        {
            for (int k=nseries; k>=1; k--)
            {
                #pragma omp single
                for (int j=0; j<nOutputs; j++)
                    *(net->series[k]->errvals +iOutputs+i) = j==i;

                net->computeDeltasSeries(net->series, k);
            }

            net->clearDsdw();
            for (int k=1; k<=nseries; k++)
            {
                net->computeGradsSeries(net->series, k, net->grad);
                stackGrads(net->grad, k-1, i);
            }
        }
        #pragma omp master
            profiler->stop("B");
    }

    {
        Real Q = trainMSE+1.;

        JtJ = J.t() * J;
        Je  = J.t() * e;
        //diagJtJ = diagmat(JtJ);

        while (Q > trainMSE)
        {
            profiler->start("S");
            tmp = chol( JtJ + mu*diagJtJ );
            dw = solve(tmp, Je, arma::solve_opts::fast);
            profiler->stop("S");
            bool _nan = false;
            for (int w=0; w<totWeights; w++)
                if (std::isnan((dw(w))) || std::isinf((dw(w))))
                    _nan = true;
            if (_nan)
            {
                printf("Found nans :( \n");
                mu *= muFactor;
                Q = trainMSE+1.;
                continue;
            }
            //printf("Solved?\n");
            profiler->start("N");
            tryNew();
            profiler->stop("N");
            Q = 0;

            profiler->start("T");
            #pragma omp parallel
            for (int k=0; k<nseries; k++)
            {
                net->predict(inputs[k], res, net->series[k], net->series[k+1]);

                #pragma omp master
                for (int i=0; i<nOutputs; i++)
                { //put this loop here to slightly reduce overhead on second step
                    Real err = targets[k][i]- *(net->series[k+1]->outvals+iOutputs+i);
                    Q += err*err;
                }
            }
            profiler->stop("T");

            if (Q > trainMSE)
            {
                profiler->start("O");
                goBack();
                profiler->stop("O");

                printf("Nope \n");
                if (mu < muMax)
                    mu *= muFactor;
                else
                    break;
            }
            else
            printf("Yeap \n");
        }

        if (mu > muMin) mu /= muFactor;

    }

}
 */
