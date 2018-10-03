//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "StatsTracker.h"

DelayedReductor::DelayedReductor(const Settings& S, const LDvec init) :
mpicomm(MPIComDup(S.mastersComm)), bAsync(S.bAsync), arysize(init.size()),
mpisize(getSize(S.mastersComm)), mpi_mutex(S.mpi_mutex), return_ret(init) {  }

LDvec DelayedReductor::get(const bool accurate)
{
  if(buffRequest not_eq MPI_REQUEST_NULL) {
    int completed = 0;
    if(accurate) {
      completed = 1;
      MPI(Wait, &buffRequest, MPI_STATUS_IGNORE);
    } else {
      MPI(Test, &buffRequest, &completed, MPI_STATUS_IGNORE);
    }
    if( completed ) {
      return_ret = reduce_ret;
      buffRequest = MPI_REQUEST_NULL;
    }
  }
  return return_ret;
}

void DelayedReductor::update(const LDvec ret)
{
  assert(ret.size() == arysize);
  if (mpisize <= 1) { return_ret = ret; return; }

  if(buffRequest not_eq MPI_REQUEST_NULL) {
    MPI(Wait, &buffRequest, MPI_STATUS_IGNORE);
    buffRequest = MPI_REQUEST_NULL;
    return_ret = reduce_ret;
  }
  reduce_ret = ret;
  assert(mpicomm not_eq MPI_COMM_NULL);
  assert(buffRequest == MPI_REQUEST_NULL);
  MPI(Iallreduce, MPI_IN_PLACE, reduce_ret.data(), arysize,
                 MPI_LONG_DOUBLE, MPI_SUM, mpicomm, &buffRequest);
}

TrainData::TrainData(const string _name, const Settings&set, bool bPPol,
  const string extrah, const Uint nextra) : n_extra(nextra),
  nThreads(set.nThreads), bPolStats(bPPol), name(_name), extra_header(extrah)
{
  resetSoft();
  resetHead();
}

TrainData::~TrainData() { }

void TrainData::log(const Real Q, const Real Qerr,
  const std::vector<Real> polG, const std::vector<Real> penal,
  std::initializer_list<Real> extra, const int thrID) {
  cntVec[thrID+1] ++;
  trackQ(Q, Qerr, thrID);
  trackPolicy(polG, penal, thrID);
  const vector<Real> tmp = extra;
  assert(tmp.size() == n_extra && bPolStats);
  for(Uint i=0; i<n_extra; i++) eVec[thrID+1][i] += tmp[i];
}

void TrainData::log(const Real Q, const Real Qerr,
  std::initializer_list<Real> extra, const int thrID) {
  cntVec[thrID+1] ++;
  trackQ(Q, Qerr, thrID);
  const vector<Real> tmp = extra;
  assert(tmp.size() == n_extra && not bPolStats);
  for(Uint i=0; i<n_extra; i++) eVec[thrID+1][i] += tmp[i];
}

void TrainData::log(const Real Q, const Real Qerr, const int thrID) {
  cntVec[thrID+1] ++;
  trackQ(Q, Qerr, thrID);
  assert(not bPolStats);
}

void TrainData::getMetrics(ostringstream& buff)
{
  reduce();
  real2SS(buff, qVec[0][0], 6, 1);
  real2SS(buff, qVec[0][1], 6, 0);
  real2SS(buff, qVec[0][2], 6, 1);
  real2SS(buff, qVec[0][3], 6, 0);
  real2SS(buff, qVec[0][4], 6, 0);
  if(bPolStats) {
    real2SS(buff, pVec[0][0], 6, 1);
    real2SS(buff, pVec[0][1], 6, 1);
    real2SS(buff, pVec[0][2], 6, 0);
  }
  for(Uint i=0; i<n_extra; i++) real2SS(buff, eVec[0][i], 6, 1);
}

void TrainData::getHeaders(ostringstream& buff) const
{
  buff <<"| RMSE | avgQ | stdQ | minQ | maxQ ";

  // polG, penG : average norm of policy/penalization gradients
  // proj : average norm of projection of polG along penG
  //        it is usually negative because curr policy should be as far as
  //        possible from behav. pol. in the direction of update
  if(bPolStats) buff <<"| polG | penG | proj ";

  // beta: coefficient of update gradient to penalization gradient:
  //       g = g_loss * beta + (1-beta) * g_penal
  // dAdv : average magnitude of Qret update
  // avgW : average importance weight
  if(n_extra) buff << extra_header;
}

void TrainData::resetSoft() {
  for(Uint i=1; i<=nThreads; i++) {
    cntVec[i] = 0;
    qVec[i][0] = 0;
    qVec[i][1] = 0;
    qVec[i][2] = 0;
    pVec[i][0] = 0;
    pVec[i][1] = 0;
    pVec[i][2] = 0;
    qVec[i][3] =  1e9;
    qVec[i][4] = -1e9;
    for(Uint j=0; j<n_extra; j++) eVec[i][j] = 0;
  }
}

void TrainData::resetHead() {
  cntVec[0]  = 0;
  qVec[0][0] = 0;
  qVec[0][1] = 0;
  qVec[0][2] = 0;
  pVec[0][0] = 0;
  pVec[0][1] = 0;
  pVec[0][2] = 0;
  qVec[0][3] =  1e9;
  qVec[0][4] = -1e9;
  for(Uint j=0; j<n_extra; j++) eVec[0][j] = 0;
}

void TrainData::reduce()
{
  resetHead();
  for (Uint i=0; i<nThreads; i++) {
    cntVec[0] += cntVec[i+1];
    qVec[0][0] += qVec[i+1][0];
    qVec[0][1] += qVec[i+1][1];
    qVec[0][2] += qVec[i+1][2];
    qVec[0][3]  = std::min(qVec[i+1][3], qVec[0][3]);
    qVec[0][4]  = std::max(qVec[i+1][4], qVec[0][4]);
    pVec[0][0] += pVec[i+1][0];
    pVec[0][1] += pVec[i+1][1];
    pVec[0][2] += pVec[i+1][2];
    for(Uint j=0; j<n_extra; j++)
      eVec[0][j] += eVec[i+1][j];
  }
  resetSoft();

  qVec[0][0] = std::sqrt(qVec[0][0]/cntVec[0]);
  qVec[0][1] /= cntVec[0]; // average Q
  qVec[0][2] /= cntVec[0]; // second moment of Q
  qVec[0][2] = std::sqrt(qVec[0][2] - qVec[0][1]*qVec[0][1]); // sdev of Q

  pVec[0][0] /= cntVec[0];
  pVec[0][1] /= cntVec[0];
  pVec[0][2] /= cntVec[0];
  for(Uint j=0; j<n_extra; j++) eVec[0][j] /= cntVec[0];

  #if 0
    if(outBuf.size()) {
      fwrite(outBuf.data(), sizeof(float), outBuf.size(), qFile);
      fflush(qFile);
      outBuf.resize(0);
    }
  #endif
}

void TrainData::trackQ(const Real Q, const Real err, const int thrID) {
  qVec[thrID+1][0] += err*err;
  qVec[thrID+1][1] += Q;
  qVec[thrID+1][2] += Q*Q;
  qVec[thrID+1][3] = std::min(qVec[thrID+1][3], static_cast<long double>(Q));
  qVec[thrID+1][4] = std::max(qVec[thrID+1][4], static_cast<long double>(Q));
}

void TrainData::trackPolicy(const std::vector<Real> polG,
  const std::vector<Real> penal, const int thrID)
{
  #if 0
    if(thrID == 1) {
      float normT = 0, dot = 0;
      for(Uint i = 0; i < polG.size(); i++) {
        dot += polG[i] * penalG[i]; normT += penalG[i] * penalG[i];
      }
      float ret[]={dot/std::sqrt(normT)};
      fwrite(ret, sizeof(float), 1, wFile);
    }
  #endif

  #if 0
    if(thrID == 1) {
      Rvec Gcpy = gradient;
      F[0]->gradStats->clip_vector(Gcpy);
      Gcpy = Rvec(&Gcpy[pol_start[0]], &Gcpy[pol_start[0]+polG.size()]);
      float normT = 0, dot = 0;
      for(Uint i = 0; i < polG.size(); i++) {
        dot += Gcpy[i] * penalG[i]; normT += penalG[i] * penalG[i];
      }
      float ret[]={dot/std::sqrt(normT)};
      fwrite(ret, sizeof(float), 1, wFile);
    }
  #endif

  Real tmpPol = 0, tmpPen = 0, tmpPrj = 0;
  for(Uint i=0; i<polG.size(); i++) {
    tmpPol +=  polG[i]* polG[i];
    tmpPen += penal[i]*penal[i];
    tmpPrj +=  polG[i]*penal[i];
  }
  pVec[thrID+1][0] += std::sqrt(tmpPol);
  pVec[thrID+1][1] += std::sqrt(tmpPen);
  static constexpr Real eps = numeric_limits<Real>::epsilon();
  pVec[thrID+1][2] += tmpPrj/(std::sqrt(tmpPen)+eps);
}

StatsTracker::StatsTracker(const Uint N, const Settings& set, Real fac) :
n_stats(N), comm(set.mastersComm), nThreads(set.nThreads),
learn_size(set.learner_size), learn_rank(set.learner_rank), grad_cut_fac(fac)
{
  avgVec[0].resize(n_stats, 0); stdVec[0].resize(n_stats, 10);
  instMean.resize(n_stats, 0); instStdv.resize(n_stats, 0);
  #pragma omp parallel for schedule(static, 1) num_threads(nThreads)
  for (Uint i=0; i<nThreads; i++) // numa aware allocation
   #pragma omp critical
   {
     avgVec[i+1].resize(n_stats, 0);
     stdVec[i+1].resize(n_stats, 0);
   }
}

void StatsTracker::track_vector(const Rvec grad, const Uint thrID) const
{
  assert(n_stats==grad.size());
  cntVec[thrID+1] += 1;
  for (Uint i=0; i<n_stats; i++) {
    avgVec[thrID+1][i] += grad[i];
    stdVec[thrID+1][i] += grad[i]*grad[i];
  }
}

void StatsTracker::advance()
{
  std::fill(avgVec[0].begin(),  avgVec[0].end(), 0);
  std::fill(stdVec[0].begin(),  stdVec[0].end(), 0);
  cntVec[0] = 0;

  for (Uint i=1; i<=nThreads; i++) {
    cntVec[0] += cntVec[i];
    for (Uint j=0; j<n_stats; j++) {
      avgVec[0][j] += avgVec[i][j];
      stdVec[0][j] += stdVec[i][j];
    }
    cntVec[i] = 0;
    std::fill(avgVec[i].begin(),  avgVec[i].end(), 0);
    std::fill(stdVec[i].begin(),  stdVec[i].end(), 0);
  }
}

void StatsTracker::update()
{
  cntVec[0] = std::max((long double)2.2e-16, cntVec[0]);
  for (Uint j=0; j<n_stats; j++) {
    const Real   mean = avgVec[0][j] / cntVec[0];
    const Real sqmean = stdVec[0][j] / cntVec[0];
    stdVec[0][j] = std::sqrt(sqmean); // - mean*mean
    avgVec[0][j] = mean;
  }
}

void StatsTracker::printToFile(const string base)
{
  if(!learn_rank) {
    FILE * pFile;
    if(!nStep) {
      // write to log the number of variables, so that it can be then unwrangled
      pFile = fopen((base + "_outGrad_stats.raw").c_str(), "wb");
      float printvals = n_stats +.1; // to be floored to an integer in post
      fwrite(&printvals, sizeof(float), 1, pFile);
    }
    else pFile = fopen((base + "_outGrad_stats.raw").c_str(), "ab");
    vector<float> printvals(n_stats*2);
    for (Uint i=0; i<n_stats; i++) {
      printvals[i]         = avgVec[0][i];
      printvals[i+n_stats] = stdVec[0][i];
    }
    fwrite(printvals.data(), sizeof(float), n_stats*2, pFile);
    fflush(pFile); fclose(pFile);
  }
}

void StatsTracker::finalize(const LDvec&oldM, const LDvec&oldS)
{
  instMean = avgVec[0];
  instStdv = stdVec[0];
  nStep++;
  for (Uint i=0; i<n_stats; i++) {
    avgVec[0][i] = (1-CLIP_LEARNR)*oldM[i] +CLIP_LEARNR*avgVec[0][i];
    stdVec[0][i] = (1-CLIP_LEARNR)*oldS[i] +CLIP_LEARNR*stdVec[0][i];
    //stdVec[0][i]=std::max((1-CLIP_LEARNR)*oldstd[i], stdVec[0][i]);
  }
}

void StatsTracker::reduce_stats(const string base, const Uint iter)
{
  const LDvec oldsum = avgVec[0], oldstd = stdVec[0];
  assert(cntVec.size()>1);
  advance();
  update();
  if(iter % 1000 == 0) printToFile(base);
  finalize(oldsum, oldstd);
}
