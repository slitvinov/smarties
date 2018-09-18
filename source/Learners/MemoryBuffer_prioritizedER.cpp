//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#if 0 // rank based probability

static inline float Q_rsqrt( const float number )
{
	union { float f; uint32_t i; } conv;
	static constexpr float threehalfs = 1.5F;
	const float x2 = number * 0.5F;
	conv.f  = number;
	conv.i  = 0x5f3759df - ( conv.i >> 1 );
  // Uncomment to do 2 iterations:
  //conv.f  = conv.f * ( threehalfs - ( x2 * conv.f * conv.f ) );
	return conv.f * ( threehalfs - ( x2 * conv.f * conv.f ) );
}

void MemoryBuffer::updateImportanceWeights()
{
  // we need to collect all errors and rank them by magnitude
  // how do we manage to do most of the work in one pass?
  // 1) gather errors along with index
  // 2) sort them by decreasing error
  // 3) compute inv sqrt of all errors, same sweep also get minP
  //const float EPS = numeric_limits<float>::epsilon();
  using USI = unsigned short;
  using TupEST = tuple<float, USI, USI>;
  const Uint nData = nTransitions.load();
  vector<TupEST> errors(nData);
  // 1)
  #pragma omp parallel for schedule(dynamic)
  for(Uint i=0; i<Set.size(); i++) {
    const auto err_i = errors.data() + Set[i]->prefix;
    for(Uint j=0; j<Set[i]->ndata(); j++)
      err_i[j] = std::make_tuple(Set[i]->SquaredError[j], (USI)i, (USI)j );
  }

  // 2)
  const auto isAbeforeB = [&] ( const TupEST& a, const TupEST& b) {
                          return std::get<0>(a) > std::get<0>(b); };
  #if 0
    __gnu_parallel::sort(errors.begin(), errors.end(), isAbeforeB);
  #else //approximate 2 pass sort
  vector<Uint> thdStarts(nThreads, 0);
  #pragma omp parallel num_threads(nThreads)
  {
    const int thrI = omp_get_thread_num();
    const Uint stride = std::ceil(nData / (Real) nThreads);
    // avoid cache thrashing: create new vector for second sort
    { // first each sorts one chunk
      const Uint start = thrI*stride;
      const Uint end = std::min( (thrI+1)*stride, nData);
      std::sort(errors.begin()+start, errors.begin()+end, isAbeforeB);
    }

    #pragma omp barrier

    // now each thread gets a quantile of partial sorts
    vector<TupEST> load_loc;
    load_loc.reserve(stride); // because we want to push back
    for(Uint t=0; t<nThreads; t++) {
      const Uint i = (t + thrI) % nThreads;
      const Uint start = i*stride, end = std::min( (i+1)*stride, nData);
      #pragma omp for schedule(static) nowait // equally divided
      for(Uint j=start; j<end; j++) load_loc.push_back( errors[j] );
    }
    const Uint locSize = load_loc.size();
    thdStarts[thrI] = locSize;
    std::sort(load_loc.begin(), load_loc.end(), isAbeforeB);

    #pragma omp barrier // wait all those thdStarts values

    Uint threadStart = 0;
    for(int i=0; i<thrI; i++) threadStart += thdStarts[i];
    for(Uint i=0; i<locSize; i++) errors[i+threadStart] = load_loc[i];
  }
  #endif
  //for(Uint i=0; i<errors.size(); i++) cout << std::get<0>(errors[i]) << endl;

  // 3)
  float minP = 1e9;
  vector<float> probs = vector<float>(nData, 1);
  #pragma omp parallel for reduction(min:minP) schedule(static)
  for(Uint i=0; i<nData; i++) {
    // if samples never seen by optimizer the samples have high priority
    const float P = std::get<0>(errors[i])>0 ? Q_rsqrt(i+1) : 1;
    const Uint seq = get<1>(errors[i]), t = get<2>(errors[i]);
    probs[Set[seq]->prefix + t] = P;
    Set[seq]->priorityImpW[t] = P;
    minP = std::min(minP, P);
  }
  minPriorityImpW = minP;

  distPER = discrete_distribution<Uint>(probs.begin(), probs.end());
}
#else // error based probability
void MemoryBuffer::updateImportanceWeights()
{
  const float EPS = numeric_limits<float>::epsilon();
  const Uint nData = nTransitions.load();
  vector<float> probs = vector<float>(nData, 1);
  float minP = 1e9, maxP = 0;

  #pragma omp parallel for schedule(dynamic) reduction(min:minP) reduction(max:maxP)
  for(Uint i=0; i<Set.size(); i++) {
    const auto ndata = Set[i]->ndata();
    const auto probs_i = probs.data() + Set[i]->prefix;

    for(Uint j=0; j<ndata; j++) {
      const float deltasq = (float)Set[i]->SquaredError[j];
      // do sqrt(delta^2)^alpha with alpha = 0.5
      const float P = deltasq>0.0f? std::sqrt(std::sqrt(deltasq+EPS)) : 0.0f;
      const float Pe = P + EPS, Qe = (P>0.0f? P : 1.0e9f) + EPS;
      minP = std::min(minP, Qe); // avoid nans in impW
      maxP = std::max(maxP, Pe);

      Set[i]->priorityImpW[j] = P;
      probs_i[j] = P;
    }
  }
  //for(Uint i=0; i<probs.size(); i++) cout << probs[i]<< endl;
  //cout <<minP <<" " <<maxP<<endl;
  // if samples never seen by optimizer the samples have high priority
  #pragma omp parallel for schedule(static)
  for(Uint i=0; i<nData; i++) if(probs[i]<=0) probs[i] = maxP;
  minPriorityImpW = minP;
  maxPriorityImpW = maxP;
  // std::discrete_distribution handles normalizing by sum P
  distPER = discrete_distribution<Uint>(probs.begin(), probs.end());
}
#endif
