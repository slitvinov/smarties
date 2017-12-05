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

  void getMetrics(ostringstream&fileOut, ostringstream&screenOut) const
  {
    fileOut<<" "<<MSE<<" "<<avgQ<<" "<<stdQ<<" "<<minQ<<" "<<maxQ;
    screenOut<<" MSE:"<<MSE<<" avgQ:"<<avgQ<<" stdQ:"<<stdQ<<" minQ:"<<minQ
    <<" maxQ:"<<maxQ;
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
  mutable vector<long double> cntVec;
  mutable vector<vector<long double>> avgVec, stdVec;

  StatsTracker(const Uint nvars, const string _name, Settings& sett) :
  n_stats(nvars), name(_name), comm(sett.mastersComm), nThreads(sett.nThreads),
  learn_size(sett.learner_size), learn_rank(sett.learner_rank),
  cntVec(nThreads+1,0), avgVec(nThreads+1,vector<long double>()),
  stdVec(nThreads+1,vector<long double>())
  {
    avgVec[0].resize(n_stats, 0); stdVec[0].resize(n_stats, 1e2);

    #pragma omp parallel for
    for (Uint i=0; i<nThreads; i++) // numa aware allocation
     #pragma omp critical
     { avgVec[i+1].resize(n_stats, 0); stdVec[i+1].resize(n_stats, 0); }
  }

  inline void track_vector(const vector<Real> grad, const Uint thrID) const
  {
    assert(n_stats==grad.size());
    cntVec[thrID+1] += 1;
    for (Uint i=0; i<n_stats; i++) {
      avgVec[thrID+1][i] += grad[i];
      stdVec[thrID+1][i] += grad[i]*grad[i];
    }
  }
  inline int clip_vector(vector<Real>& grad) const
  {
    int ret = 0;
    for (Uint i=0; i<n_stats; i++) {
      //#ifdef importanceSampling
      //  assert(data->Set[seq]->tuples[samp]->weight>0);
      //  grad[i] *= data->Set[seq]->tuples[samp]->weight;
      //#endif
      #ifdef ACER_GRAD_CUT
        if(grad[i]>  ACER_GRAD_CUT*stdVec[0][i] && stdVec[0][i]>2.2e-16) {
        //printf("Cut %u was:%f is:%LG\n",i,grad[i], ACER_GRAD_CUT*stdVec[0][i]);
          grad[i] =  ACER_GRAD_CUT*stdVec[0][i];
          ret = 1;
        } else
        if(grad[i]< -ACER_GRAD_CUT*stdVec[0][i] && stdVec[0][i]>2.2e-16) {
        //printf("Cut %u was:%f is:%LG\n",i,grad[i],-ACER_GRAD_CUT*stdVec[0][i]);
          grad[i] = -ACER_GRAD_CUT*stdVec[0][i];
          ret = 1;
        }
        //else printf("Not cut\n");
      #endif
    }
    return ret;
  }

  void advance()
  {
    for(Uint i=0; i<n_stats; i++){ avgVec[0][i]=0; stdVec[0][i]=0; }
    cntVec[0] = 0;

    for (Uint i=1; i<=nThreads; i++) {
      cntVec[0] += cntVec[i]; cntVec[i] = 0;
      for (Uint j=0; j<n_stats; j++) {
        avgVec[0][j] += avgVec[i][j]; avgVec[i][j] = 0;
        stdVec[0][j] += stdVec[i][j]; stdVec[i][j] = 0;
      }
    }
  }
  void update()
  {
    cntVec[0] = std::max((long double)2.2e-16, cntVec[0]);
    for (Uint j=0; j<n_stats; j++) {
      const Real   mean = avgVec[0][j] / cntVec[0];
      const Real sqmean = stdVec[0][j] / cntVec[0];
      stdVec[0][j] = std::sqrt(sqmean - mean*mean);
      avgVec[0][j] = mean;
    }
  }
  void printToFile()
  {
    if(!learn_rank) {
      ofstream filestats;
      filestats.open(name + ".txt", ios::app);
      filestats<<print(avgVec[0])<<" "<<print(stdVec[0])<<endl;
      filestats.close();
    }
  }
  inline void reduce_stats()
  {
    const vector<long double> oldsum = avgVec[0], oldstd = stdVec[0];
    assert(cntVec.size()>1);

    advance();
    if (learn_size > 1) {
      MPI_Allreduce(MPI_IN_PLACE, &cntVec[0], 1,
          MPI_LONG_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(MPI_IN_PLACE, avgVec[0].data(), avgVec[0].size(),
          MPI_LONG_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(MPI_IN_PLACE, stdVec[0].data(), stdVec[0].size(),
          MPI_LONG_DOUBLE, MPI_SUM, comm);
    }
    update();
    printToFile();

    for (Uint i=0; i<n_stats; i++) {
      avgVec[0][i] = .99*oldsum[i] +.01*avgVec[0][i];
      stdVec[0][i] = .99*oldstd[i] +.01*stdVec[0][i];
      //stdVec[0][i] = std::max(0.99*oldstd[i], stdVec[0][i]);
    }
  }
  inline void reduce_approx()
  {
    const vector<long double> oldsum = avgVec[0], oldstd = stdVec[0];
    assert(cntVec.size()>1);
    advance();
    update();
    printToFile();
    for (Uint i=0; i<n_stats; i++) {
      avgVec[0][i] = .99*oldsum[i] +.01*avgVec[0][i];
      //stdVec[0][i] = .99*oldstd[i] +.01*stdVec[0][i];
      stdVec[0][i] = std::max(0.99*oldstd[i], stdVec[0][i]);
    }
  }
  void getMetrics(ostringstream&fileOut, ostringstream&screenOut) const
  {
    screenOut<<" "<<name<<" avg:["<<print(avgVec[0])
                  <<"] std:["<<print(stdVec[0])<<"]";
    fileOut<<" "<<print(avgVec[0])<<" "<<print(stdVec[0]);
  }
};
