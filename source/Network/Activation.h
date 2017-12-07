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

struct Memory //Memory light recipient for recurrent connections
{
  Memory(vector<Uint>_sizes, vector<Uint>_bOut): nLayers(_sizes.size()),
  outvals(allocate_vec(_sizes)), sizes(_sizes) {}

  inline void clearOutput() const {
    for(Uint i=0; i<nLayers; i++) {
      const int sizesimd = std::ceil(sizes[i]*sizeof(nnReal)/32.)*32;
      assert(outvals[i] not_eq nullptr && sizes[i]>0);
      std::memset(outvals[i], 0, sizesimd);
    }
  }

  ~Memory() { for(auto& p : outvals) if(p not_eq nullptr) free(p); }
  const Uint nLayers;
  const vector<nnReal*> outvals;
  const vector<Uint> sizes;
};

struct Activation
{
  Uint _nOuts(vector<Uint> _sizes, vector<Uint> _bOut) {
    assert(_sizes.size() == _bOut.size() && nLayers == _bOut.size());
    Uint ret = 0;
    for(Uint i=0; i<_bOut.size(); i++) if(_bOut[i]) ret += _sizes[i];
    if(!ret) {
      ret = _sizes.back();
      //warn("had to overwrite nOutputs");
    }
    //printf("sizes:%s outputs:%s Total:%u\n", print(_sizes).c_str(),
    //  print(_bOut).c_str(), ret);
    assert(ret>0);
    return ret;
  }

  Activation(vector<Uint>_sizes, vector<Uint>_bOut): nLayers(_sizes.size()),
    nOutputs(_nOuts(_sizes,_bOut)), sizes(_sizes), output(_bOut),
    suminps(allocate_vec(_sizes)), outvals(allocate_vec(_sizes)),
    errvals(allocate_vec(_sizes)) {
    assert(suminps.size()==nLayers);
    assert(outvals.size()==nLayers);
    assert(errvals.size()==nLayers);
  }

  ~Activation() {
    for(auto& p : suminps) if(p not_eq nullptr) free(p);
    for(auto& p : outvals) if(p not_eq nullptr) free(p);
    for(auto& p : errvals) if(p not_eq nullptr) free(p);
  }

  inline void setInput(const vector<nnReal> inp) const {
    assert(sizes[0] == inp.size()); //alternative not supported
    memcpy(outvals[0], &inp[0], sizes[0]*sizeof(nnReal));
  }

  inline void setOutputDelta(const vector<nnReal> delta) const {
    assert(nOutputs == delta.size()); //alternative not supported
    Uint k=0;
    for(Uint i=0; i<nLayers; i++) if(output[i]) {
      memcpy(errvals[i], &delta[k], sizes[i]*sizeof(nnReal));
      k += sizes[i];
    }
    if(k==0) {
      assert(nOutputs == sizes.back());
      memcpy(errvals.back(), &delta[0], nOutputs*sizeof(nnReal));
    } else assert(k == nOutputs);
  }

  inline void addOutputDelta(const vector<nnReal> delta) const {
    assert(nOutputs == delta.size()); //alternative not supported
    Uint k=0;
    for(Uint i=0; i<nLayers; i++) if(output[i])
      for (Uint j=0; j<sizes[i]; j++, k++) errvals[i][j] += delta[k];

    if(k==0) {
      assert(nOutputs == sizes.back());
      for (Uint j=0; j<nOutputs; j++) errvals.back()[j] += delta[j];
    } else assert(k == nOutputs);
  }

  inline vector<nnReal> getOutput() const {
    vector<nnReal> ret(nOutputs);
    Uint k=0;
    for(Uint i=0; i<nLayers; i++) if(output[i]) {
      memcpy(&ret[k], outvals[i], sizes[i]*sizeof(nnReal));
      k += sizes[i];
    }
    if(k==0) {
      assert(nOutputs ==sizes.back());
      memcpy(&ret[0], outvals.back(), nOutputs*sizeof(nnReal));
    } else assert(k == nOutputs);
    return ret;
  }

  inline vector<nnReal> getInputGradient() const {
    vector<nnReal> ret(sizes[0]);
    memcpy(&ret[0], errvals[0], sizes[0]*sizeof(nnReal));
    return ret;
  }

  inline void clearOutput() const {
    for(Uint i=0; i<nLayers; i++) {
      const int sizesimd = std::ceil(sizes[i]*sizeof(nnReal)/32.)*32;
      assert(outvals[i] not_eq nullptr && sizes[i]>0);
      std::memset(outvals[i], 0, sizesimd);
    }
  }

  inline void clearErrors() const {
    for(Uint i=0; i<nLayers; i++) {
      const int sizesimd = std::ceil(sizes[i]*sizeof(nnReal)/32.)*32;
      assert(errvals[i] not_eq nullptr && sizes[i]>0);
      std::memset(errvals[i], 0, sizesimd);
    }
  }

  inline void clearInputs() const {
    for(Uint i=0; i<nLayers; i++) {
      const int sizesimd = std::ceil(sizes[i]*sizeof(nnReal)/32.)*32;
      assert(suminps[i] not_eq nullptr && sizes[i]>0);
      std::memset(suminps[i], 0, sizesimd);
    }
  }

  inline void loadMemory(Memory*const _M) const {
    for(Uint i=0; i<nLayers; i++) {
      assert(outvals[i] not_eq nullptr);
      assert(_M->outvals[i] not_eq nullptr);
      assert(sizes[i] == _M->sizes[i]);
      memcpy(outvals[i], _M->outvals[i], sizes[i]*sizeof(nnReal));
    }
  }

  inline void storeMemory(Memory*const _M) const {
    for(Uint i=0; i<nLayers; i++) {
      assert(outvals[i] not_eq nullptr);
      assert(_M->outvals[i] not_eq nullptr);
      assert(sizes[i] == _M->sizes[i]);
      memcpy(_M->outvals[i], outvals[i], sizes[i]*sizeof(nnReal));
    }
  }

  inline nnReal* X(const Uint layerID) const {
    assert(layerID < nLayers);
    return suminps[layerID];
  }
  inline nnReal* Y(const Uint layerID) const {
    assert(layerID < nLayers);
    return outvals[layerID];
  }
  inline nnReal* E(const Uint layerID) const {
    assert(layerID < nLayers);
    return errvals[layerID];
  }
  inline Uint nInputs() const { return sizes[0]; }

  const Uint nLayers, nOutputs;
  const vector<Uint> sizes, output;
  //contains all inputs to each neuron (inputs to network input layer is empty)
  const vector<nnReal*> suminps;
  //contains all neuron outputs that will be the incoming signal to linked layers (outputs of input layer is network inputs)
  const vector<nnReal*> outvals;
  //deltas for each neuron
  const vector<nnReal*> errvals;
  mutable bool written = false;
};

inline void deallocateUnrolledActivations(vector<Activation*>& r)
{
  for (auto & ptr : r) _dispose_object(ptr);
  r.clear();
}
inline void deallocateUnrolledActivations(vector<Activation*>* r)
{
  for (auto & ptr : *r) _dispose_object(ptr);
  r->clear();
}

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
