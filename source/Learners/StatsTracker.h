/*
 *  Learner.h
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "../Settings.h"

template<typename T, MPI_Datatype MPI_RDX_TYPE>
struct ApproximateReductor
{
  const MPI_Comm mpicomm;
  const Uint mpisize, arysize;
  MPI_Request buffRequest = MPI_REQUEST_NULL;
  vector<T> reduce_ret = vector<T>(arysize, 0);
  vector<T> local_vals = vector<T>(arysize, 0);

  static int getSize(const MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
  }
  ApproximateReductor(const MPI_Comm c, const Uint N) :
  mpicomm(c), mpisize(getSize(c)), arysize(N)
  { }

  int sync(vector<T>& ret, bool accurate = false)
  {
    if (mpisize <= 1) return 1;
    const bool firstUpdate = buffRequest == MPI_REQUEST_NULL;
    if(not firstUpdate) MPI_Wait(&buffRequest, MPI_STATUS_IGNORE);
    assert(ret.size() == arysize);
    local_vals = ret;
    ret = reduce_ret;
    if(accurate){
      if(not firstUpdate) die("undefined behavior")
      MPI_Allreduce( local_vals.data(), reduce_ret.data(), arysize,
                     MPI_RDX_TYPE, MPI_SUM, mpicomm);
    } else {
      MPI_Iallreduce(local_vals.data(), reduce_ret.data(), arysize,
                     MPI_RDX_TYPE, MPI_SUM, mpicomm, &buffRequest);
    }
    // if no reduction done, partial sums are meaningless
    return firstUpdate and not accurate;
  }
};

struct trainData
{
  trainData():MSE(0),avgQ(0),stdQ(0),minQ(1e9),maxQ(-1e9),relE(0),dCnt(0) {}
  long double MSE, avgQ, stdQ, minQ, maxQ, relE, dCnt;

  void reduce(vector<trainData>& Vstats)
  {
    minQ= 1e9;MSE =0;dCnt=0;
    maxQ=-1e9;avgQ=0;relE=0;
    for (Uint i=0; i<Vstats.size(); i++) {
      MSE  += Vstats[i].MSE;
      avgQ += Vstats[i].avgQ;
      stdQ += Vstats[i].stdQ;
      dCnt += Vstats[i].dCnt;
      minQ = std::min(minQ, Vstats[i].minQ);
      maxQ = std::max(maxQ, Vstats[i].maxQ);
      Vstats[i].minQ= 1e9; Vstats[i].MSE =0; Vstats[i].dCnt=0;
      Vstats[i].maxQ=-1e9; Vstats[i].avgQ=0; Vstats[i].stdQ=0;
    }

    #if 0
    if (learn_size > 1) {
    long double ary[4] = {stats.MSE, stats.dCnt, stats.avgQ, stats.stdQ};
    MPI_Allreduce(MPI_IN_PLACE,ary,4,MPI_LONG_DOUBLE,MPI_SUM,mastersComm);
    stats.MSE=ary[0]; stats.dCnt=ary[1]; stats.avgQ=ary[2]; stats.stdQ=ary[3];
    #if 0
    MPI_Allreduce(MPI_IN_PLACE,&stats.minQ,1,MPI_LONG_DOUBLE,MPI_MIN,mastersComm);
    MPI_Allreduce(MPI_IN_PLACE,&stats.maxQ,1,MPI_LONG_DOUBLE,MPI_MAX,mastersComm);
    #endif
    }
    #endif

    const long double sum = avgQ, sumsq = stdQ, cnt = dCnt;
    MSE   = std::sqrt(MSE/cnt);
    avgQ /= cnt; //stats.relE/=stats.dCnt;
    stdQ  = std::sqrt((sumsq-sum*sum/cnt)/cnt);
  }

  void getMetrics(ostringstream& buff) const
  {
    {
      const auto prec = std::fabs( MSE)>1e3? 0 : (std::fabs( MSE)>1e2? 1:2);
      buff<<" "<<std::setw(6)<<std::setprecision(prec)<<std::fixed<< MSE;
    }
    {
      const auto prec = std::fabs(avgQ)>1e3? 0 : (std::fabs(avgQ)>1e2? 1:2);
      buff<<" "<<std::setw(6)<<std::setprecision(prec)<<std::fixed<<avgQ;
    }
    {
      const auto prec = std::fabs(stdQ)>1e3? 0 : (std::fabs(stdQ)>1e2? 1:2);
      buff<<" "<<std::setw(6)<<std::setprecision(prec)<<std::fixed<<stdQ;
    }
    {
      const auto prec = std::fabs(minQ)>1e3? 0 : (std::fabs(minQ)>1e2? 1:2);
      buff<<" "<<std::setw(6)<<std::setprecision(prec)<<std::fixed<<minQ;
    }
    {
      const auto prec = std::fabs(maxQ)>1e3? 0 : (std::fabs(maxQ)>1e2? 1:2);
      buff<<" "<<std::setw(6)<<std::setprecision(prec)<<std::fixed<<maxQ;
    }
  }
  void getHeaders(ostringstream& buff) const
  {
    buff <<"| RMSE | avgQ | stdQ | minQ | maxQ ";
  }

  inline void dumpStats(const Real&Q, const Real&err)
  {
    MSE += err*err;
    avgQ += Q;
    stdQ += Q*Q;
    minQ = std::min(minQ, static_cast<long double>(Q));
    maxQ = std::max(maxQ, static_cast<long double>(Q));
    dCnt++;
  }
};

struct StatsTracker
{
  const Uint n_stats;
  const string name;
  const MPI_Comm comm;
  const Uint nThreads, learn_size, learn_rank;
  const Real grad_cut_fac, learnR;
  mutable LDvec cntVec = LDvec(nThreads+1,0);
  mutable vector<LDvec> avgVec = vector<LDvec>(nThreads+1, LDvec());
  mutable vector<LDvec> stdVec = vector<LDvec>(nThreads+1, LDvec());
  LDvec instMean, instStdv;
  mutable Real numCut = 0, numTot = 0;
  unsigned long nStep = 0;
  Real cutRatio = 0;

  ApproximateReductor<long double, MPI_LONG_DOUBLE> reductor =
  ApproximateReductor<long double, MPI_LONG_DOUBLE>(comm, 2*n_stats +1);

  StatsTracker(const Uint N, const string _name, Settings& set, Real fac) :
  n_stats(N), name(_name), comm(set.mastersComm), nThreads(set.nThreads),
  learn_size(set.learner_size), learn_rank(set.learner_rank), grad_cut_fac(fac),
  learnR(set.learnrate)
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
     // write to log the number of variables, so that it can be then unwrangled
     FILE * pFile = fopen((name + ".raw").c_str(), "wb");
     float printvals = n_stats +0.1; // to be floored to an integer in post
     fwrite(&printvals, sizeof(float), 1, pFile);
     fflush(pFile); fclose(pFile);
  }

  inline void track_vector(const Rvec grad, const Uint thrID) const
  {
    assert(n_stats==grad.size());
    cntVec[thrID+1] += 1;
    for (Uint i=0; i<n_stats; i++) {
      avgVec[thrID+1][i] += grad[i];
      stdVec[thrID+1][i] += grad[i]*grad[i];
    }
  }
  inline void clip_vector(Rvec& grad) const
  {
    assert(grad.size() == n_stats);
    Uint ret = 0;
    Real change = 0;
    for (Uint i=0; i<n_stats && grad_cut_fac>=1; i++) {
      //#ifdef IMPORTSAMPLE
      //  assert(data->Set[seq]->tuples[samp]->weight>0);
      //  grad[i] *= data->Set[seq]->tuples[samp]->weight;
      //#endif
      if(grad[i]> grad_cut_fac*stdVec[0][i] && stdVec[0][i]>2.2e-16) {
        //printf("Cut %u was:%f is:%LG\n",i,grad[i], grad_cut_fac*stdVec[0][i]);
        change+=(grad[i]-grad_cut_fac*stdVec[0][i])/(grad_cut_fac*stdVec[0][i]);
        grad[i] = grad_cut_fac*stdVec[0][i];
        ret += 1;
      } else
      if(grad[i]< -grad_cut_fac*stdVec[0][i] && stdVec[0][i]>2.2e-16) {
        //printf("Cut %u was:%f is:%LG\n",i,grad[i],-grad_cut_fac*stdVec[0][i]);
        change-=(grad[i]+grad_cut_fac*stdVec[0][i])/(grad_cut_fac*stdVec[0][i]);
        grad[i] = -grad_cut_fac*stdVec[0][i];
        ret += 1;
      }
      //else printf("Not cut\n");
    }
    #pragma omp atomic
    //numCut += ret;
    numCut += change;
    #pragma omp atomic
    numTot += n_stats;
  }

  inline void advance()
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
  inline void update()
  {
    cntVec[0] = std::max((long double)2.2e-16, cntVec[0]);
    for (Uint j=0; j<n_stats; j++) {
      const Real   mean = avgVec[0][j] / cntVec[0];
      const Real sqmean = stdVec[0][j] / cntVec[0];
      stdVec[0][j] = std::sqrt(sqmean); // - mean*mean
      avgVec[0][j] = mean;
    }
  }
  inline void printToFile()
  {
    if(!learn_rank) {
      FILE * pFile = fopen((name + ".raw").c_str(), "ab");
      vector<float> printvals(n_stats*2);
      for (Uint i=0; i<n_stats; i++) {
        printvals[i]         = avgVec[0][i];
        printvals[i+n_stats] = stdVec[0][i];
      }
      fwrite(printvals.data(), sizeof(float), n_stats*2, pFile);
      fflush(pFile); fclose(pFile);
    }
  }
  void finalize(const LDvec&oldM, const LDvec&oldS)
  {
    instMean = avgVec[0];
    instStdv = stdVec[0];
    nStep++;
    //const Real learnRate = learnR / (1 + nStep * ANNEAL_RATE);
    for (Uint i=0; i<n_stats; i++) {
      avgVec[0][i] = (1-CLIP_LEARNR)*oldM[i] +CLIP_LEARNR*avgVec[0][i];
      stdVec[0][i] = (1-CLIP_LEARNR)*oldS[i] +CLIP_LEARNR*stdVec[0][i];
      //stdVec[0][i]=std::max((1-CLIP_LEARNR)*oldstd[i], stdVec[0][i]);
    }
  }
  double clip_ratio()
  {
    cutRatio = numCut / (Real) numTot;
    numCut = 0; numTot = 0;
    return cutRatio;
  }
  inline void reduce_stats(const Uint iter = 0)
  {
    const LDvec oldsum = avgVec[0], oldstd = stdVec[0];
    assert(cntVec.size()>1);

    advance();

    if (learn_size > 1) {
      LDvec res = avgVec[0];
      res.insert(res.end(), stdVec[0].begin(), stdVec[0].end());
      res.push_back(cntVec[0]);
      assert(res.size() == 2*n_stats+1);
      bool skipped = reductor.sync(res);
      if(skipped) {
        avgVec[0] = oldsum; stdVec[0] = oldstd;
        return;
      } else {
        for (Uint i=0; i<n_stats; i++) {
          avgVec[0][i] = res[i]; stdVec[0][i] = res[i+n_stats];
        }
        cntVec[0] = res[2*n_stats];
      }
    }
    update();

    if(iter % 1000 == 0) printToFile();
    finalize(oldsum, oldstd);
  }
  inline void reduce_approx(const Uint iter = 0)
  {
    const LDvec oldsum = avgVec[0], oldstd = stdVec[0];
    assert(cntVec.size()>1);
    advance();
    update();
    if(iter % 1000 == 0) printToFile();
    finalize(oldsum, oldstd);
  }

  //void getMetrics(ostringstream&fileOut, ostringstream&screenOut) const
  //{
  //  screenOut<<" "<<name<<" avg:["<<print(instMean)
  //                <<"] std:["<<print(instStdv)<<"]";
  //  fileOut<<" "<<print(instMean)<<" "<<print(stdVec[0]);
  //}
};
