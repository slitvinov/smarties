//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_ThreadSafeVec_h
#define smarties_ThreadSafeVec_h

#include "Definitions.h"
#include <cassert>
#include <memory>
#include <omp.h>

namespace smarties
{

template<typename T>
struct THRvec
{
  Uint nThreads;
  const T initial;
  std::vector<std::shared_ptr<T>> m_v;

  THRvec(const Uint size, const T init=T()) : nThreads(size), initial(init)
  {
    m_v.resize(nThreads);
    #pragma omp parallel for num_threads(nThreads) schedule(static, 1)
    for(Uint i=0; i<nThreads; ++i) m_v[i] = std::make_shared<T>(initial);
  }

  THRvec(const THRvec&c) : nThreads(c.nThreads), initial(c.initial)
  {
    m_v.resize(nThreads);
    #pragma omp parallel for num_threads(nThreads) schedule(static, 1)
    for(Uint i=0; i<nThreads; ++i) m_v[i] = c.m_v[i];
  }

  void resize(const Uint N)
  {
    if(N == nThreads) return;

    if(N < nThreads) m_v.resize(N);
    else { // N > nThreads
      m_v.reserve(N);
    }
    nThreads = N;
    m_v.resize(N, nullptr);
  }

  Uint size() const { return nThreads; }

  T& operator[] (const Uint i) const
  {
    assert(m_v[i] not_eq nullptr);
    return * m_v[i];
  }
};

} // end namespace smarties
#endif // smarties_Settings_h
