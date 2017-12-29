/*
 *  Functions.h
 *  rl
 *
 *  Guido Novati on 04.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include "Functions.h"

struct Parameters
{
 private:
  vector<Uint> indBiases, indWeights;
  vector<Uint> nBiases, nWeights;
 public:
  const Uint nParams, nLayers;

  // array containing all parameters of network contiguously
  //(used by optimizer and for MPI reductions)
  nnReal*const params;

  //each layer requests a certain number of parameters, here compute contiguous
  //memory required such that each layer gets an aligned pointer to both
  //its first bias and and first weight, allowing SIMD ops on all layers
  Uint computeNParams(vector<Uint> _nWeights, vector<Uint> _nBiases)
  {
    assert(_nWeights.size() == _nBiases.size());
    const Uint nL = _nWeights.size();
    Uint nTotPara = 0;
    indBiases = vector<Uint>(nL, 0);
    indWeights = vector<Uint>(nL, 0);
    for(Uint i=0; i<nL; i++) {
      indWeights[i] = nTotPara;
      nTotPara += std::ceil(_nWeights[i]*sizeof(nnReal)/32.)*32/sizeof(nnReal);
      indBiases[i] = nTotPara;
      nTotPara += std::ceil( _nBiases[i]*sizeof(nnReal)/32.)*32/sizeof(nnReal);
    }
    //printf("Weight sizes:[%s] inds:[%s] Bias sizes:[%s] inds[%s] Total:%u\n",
    //  print(_nWeights).c_str(), print(indWeights).c_str(),
    //  print(_nBiases).c_str(), print(indBiases).c_str(), nTotPara);
    return nTotPara;
  }

  Parameters* allocateGrad() const
  {
    return new Parameters(nWeights, nBiases);
  }

  inline void broadcast(const MPI_Comm comm) const
  {
    MPI_Bcast(params, nParams, MPI_NNVALUE_TYPE, 0, comm);
  }

  inline void copy(const Parameters* const tgt) const
  {
    assert(nParams == tgt->nParams);
    #pragma omp parallel for
    for (Uint j=0; j<nParams; j++) params[j] = tgt->params[j];
  }

  inline void penalization(const nnReal lambda) const {
    nnReal* const dst = params;    
    #pragma omp parallel for simd aligned(dst : VEC_WIDTH) 
    for (Uint i=0; i<nParams; i++)
    #ifdef NET_L1_PENAL
      dst[i] += (dst[i]<0 ? lambda : -lambda);
    #else
      dst[i] -= dst[i]*lambda;
    #endif
  }

  Parameters(vector<Uint> _nWeights, vector<Uint> _nBiases) :
   nBiases(_nBiases), nWeights(_nWeights),
   nParams(computeNParams(_nWeights, _nBiases)), nLayers(_nWeights.size()),
   params(allocate_ptr(nParams))  { }

  ~Parameters() { if(params not_eq nullptr) free(params); }

  void reduceThreadsGrad(const vector<Parameters*>& g) const
  {
    #pragma omp parallel
    {
      const Uint thrID = static_cast<Uint>(omp_get_thread_num());
      assert(nParams == g[thrID]->nParams);
      const nnReal* const src = g[thrID]->params;    
      nnReal* const dst = params; 
      // every thread starts staggered to avoid race conditions:  
      const Uint shift = thrID*((nParams/g.size())/ARY_WIDTH)*ARY_WIDTH;
      #pragma omp simd aligned(dst, src : VEC_WIDTH) 
      for(Uint j=0; j<nParams; j++) {
        const Uint ind = (j + shift) % nParams;
        dst[ind] += src[ind];
      }
      g[thrID]->clear(); 
    }
  }

  long double compute_weight_norm() const
  {
    long double sumWeights = 0;
    #pragma omp parallel for reduction(+:sumWeights)
    for (Uint w=0; w<nParams; w++)
      sumWeights += std::fabs(params[w]);
      //sumWeightsSq += net->weights[w]*net->weights[w];
      //distTarget += std::fabs(net->weights[w]-net->tgt_weights[w]);
    return sumWeights;
  }

  void compute_dist_norm(long double& norm, long double& dist,
    const Parameters*const TGT) const
  {
    norm = 0; dist = 0;
    #pragma omp parallel for reduction(+ : norm, dist)
    for (Uint w=0; w<nParams; w++) {
      norm += std::fabs(params[w]);
      dist += std::fabs(params[w] - TGT->params[w]);
    }
  }

  inline void clear() const {
    std::memset(params, 0, nParams*sizeof(nnReal));
  }

  void save(const std::string fname) const {
    FILE * wFile = fopen((fname+".raw").c_str(), "wb");
    fwrite(params, sizeof(nnReal), nParams, wFile); fflush(wFile);
    fclose(wFile);
  }
  int restart(const std::string fname) const {
    FILE * wFile = fopen((fname+".raw").c_str(), "rb");
    if(wFile == NULL) {
      _warn("Parameters restart file %s not found.", (fname+".raw").c_str());
      return 1;
    }
    size_t wsize = fread(params, sizeof(nnReal), nParams, wFile);
    fclose(wFile);
    if(wsize not_eq nParams)
      _die("Mismatch in restarted file %s; contains:%lu read:%lu.",
        fname.c_str(), wsize, nParams);
    return 0;
  }

  inline nnReal* W(const Uint layerID) const {
    assert(layerID < nLayers);
    return params + indWeights[layerID];
  }
  inline nnReal* B(const Uint layerID) const {
    assert(layerID < nLayers);
    return params + indBiases[layerID];
  }
  inline Uint NW(const Uint layerID) const {
    assert(layerID < nLayers);
    return nWeights[layerID];
  }
  inline Uint NB(const Uint layerID) const {
    assert(layerID < nLayers);
    return nBiases[layerID];
  }
};

#if 0
  inline void circle_region(Grads*const trust, Grads*const grad, const Real delta, const int ngrads)
  {
    #if 1
      assert(trust->nWeights==grad->nWeights && trust->nBiases==grad->nBiases);
      long double norm = 0, fac = 1./(trust->nWeights+trust->nBiases)/ngrads/ngrads;
      for(Uint j=0; j<trust->nWeights; j++)
        norm += fac*std::pow(grad->_W[j]+trust->_W[j], 2);
        //norm += std::fabs(grad->_W[j]+trust->_W[j]);
      for(Uint j=0; j<trust->nBiases; j++)
        norm += fac*std::pow(grad->_B[j]+trust->_B[j], 2);
        //norm += std::fabs(grad->_B[j]+trust->_B[j]);

      const Real nG = std::sqrt(norm), softclip = delta/(nG+delta);
      //printf("%Lg %Lg %g %f\n",fac, norm, nG, softclip);
      //printf("grad norm %f\n",nG);
      for(Uint j=0; j<trust->nWeights; j++)
        grad->_W[j] = (grad->_W[j]+trust->_W[j])*softclip -trust->_W[j];
      for(Uint j=0; j<trust->nBiases; j++)
        grad->_B[j] = (grad->_B[j]+trust->_B[j])*softclip -trust->_B[j];
    #else
      Real dot=0, norm = numeric_limits<Real>::epsilon();
      for(Uint j=0; j<trust->nWeights; j++) {
        norm += std::pow(trust->_W[j]/ngrads, 2);
        dot += grad->_W[j]*trust->_W[j]/(ngrads*ngrads);
      }
      for(Uint j=0; j<trust->nBiases; j++)  {
        norm += std::pow(trust->_B[j]/ngrads, 2);
        dot += grad->_B[j]*trust->_B[j]/(ngrads*ngrads);
      }
      const Real proj = std::max( (Real)0, (dot - delta)/norm );
      //printf("grad norm %f %f %f\n", proj, dot, norm);
      for(Uint j=0; j<trust->nWeights; j++)
        grad->_W[j] = grad->_W[j] -proj*trust->_W[j];
      for(Uint j=0; j<trust->nBiases; j++)
        grad->_B[j] = grad->_B[j] -proj*trust->_B[j];
    #endif
    trust->clear();
  }

  inline void circle_region(Grads*const grad, Grads*const trust, Grads*const dest, const Real delta)
  {
    assert(trust->nWeights==grad->nWeights && trust->nBiases==grad->nBiases);
    long double norm = 0, fac = 1./(trust->nWeights+trust->nBiases);
    {
      for(Uint j=0;j<trust->nWeights;j++)
      norm += fac*std::pow(grad->_W[j]+trust->_W[j],2);
      //norm += fac*std::fabs(grad->_W[j]+trust->_W[j]);
      for(Uint j=0;j<trust->nBiases; j++)
      norm += fac*std::pow(grad->_B[j]+trust->_B[j],2);
      //norm += fac*std::fabs(grad->_B[j]+trust->_B[j]);
    }
    const auto nG = std::sqrt(norm);
    //const Real nG = norm;
    const Real softclip = delta/(nG+delta);
    //printf("%Lg %Lg %Lg %f\n",fac, norm, nG, softclip);

    for(Uint j=0;j<trust->nWeights;j++)
      dest->_W[j] += (grad->_W[j]+trust->_W[j])*softclip -trust->_W[j];
    for(Uint j=0;j<trust->nBiases; j++)
      dest->_B[j] += (grad->_B[j]+trust->_B[j])*softclip -trust->_B[j];
    trust->clear();
    grad->clear();
  }

  inline void fullstats(Grads*const grad, Grads*const trust, Grads*const dest, const Real delta)
  {
    assert(trust->nWeights==grad->nWeights && trust->nBiases==grad->nBiases);
    Real EO1 = 0, EO2 = 0, EO3 = 0, EO4 = 0;
    Real EL1 = 0, EL2 = 0, EL3 = 0, EL4 = 0;
    Real EC1 = 0, EC2 = 0, EC3 = 0, EC4 = 0;

    Real dot=0, norm=numeric_limits<Real>::epsilon(), sum=0,  dotL=0, dotC=0;
    for(Uint j=0; j<trust->nWeights; j++) {
      sum += std::pow(grad->_W[j]+trust->_W[j],2);
      norm += trust->_W[j]*trust->_W[j];
      dot +=   grad->_W[j]*trust->_W[j];
    }
    for(Uint j=0; j<trust->nBiases; j++)  {
      sum += std::pow(grad->_B[j]+trust->_B[j],2);
      norm += trust->_B[j]*trust->_B[j];
      dot +=   grad->_B[j]*trust->_B[j];
    }
    const Real proj = std::max( (Real)0, (dot - delta)/norm );
    const Real nG = std::sqrt(sum), clip = delta/(nG+delta);

    for(Uint j=0; j<trust->nWeights; j++) {
      const long double linear = grad->_W[j] -proj*trust->_W[j];
      const long double circle = (grad->_W[j]+trust->_W[j])*clip -trust->_W[j];
      dotL += linear*trust->_W[j];
      dotC += circle*trust->_W[j];
      dest->_W[j] += circle;
    }
    for(Uint j=0; j<trust->nBiases; j++)  {
      const long double linear = grad->_B[j] -proj*trust->_B[j];
      const long double circle = (grad->_B[j]+trust->_B[j])*clip -trust->_B[j];
      dotL += linear*trust->_B[j];
      dotC += circle*trust->_B[j];
      dest->_B[j] += circle;
    }

    if(omp_get_thread_num() == 1) {
      EO1 =          dot/std::sqrt(norm);      // to compute E[grad_proj_dkl]
      EO2 = std::pow(dot/std::sqrt(norm), 2);  //higher order statistics
      EO3 = std::pow(dot/std::sqrt(norm), 3);
      EO4 = std::pow(dot/std::sqrt(norm), 4);
      EL1 =          dotL/std::sqrt(norm);      // to compute E[grad_proj_dkl]
      EL2 = std::pow(dotL/std::sqrt(norm), 2);  //higher order statistics
      EL3 = std::pow(dotL/std::sqrt(norm), 3);
      EL4 = std::pow(dotL/std::sqrt(norm), 4);
      EC1 =          dotC/std::sqrt(norm);      // to compute E[grad_proj_dkl]
      EC2 = std::pow(dotC/std::sqrt(norm), 2);  //higher order statistics
      EC3 = std::pow(dotC/std::sqrt(norm), 3);
      EC4 = std::pow(dotC/std::sqrt(norm), 4);
      ofstream fs;
      fs.open("gradproj_dist.txt", ios::app);
      fs<<EO1<<"\t"<<EO2<<"\t"<<EO3<<"\t"<<EO4<<"\t"
        <<EL1<<"\t"<<EL2<<"\t"<<EL3<<"\t"<<EL4<<"\t"
        <<EC1<<"\t"<<EC2<<"\t"<<EC3<<"\t"<<EC4<<endl;
      fs.close(); fs.flush();
    }

    trust->clear();
    grad->clear();
  }
#endif
