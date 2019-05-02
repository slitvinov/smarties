//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Conv2DLayer_h
#define smarties_Conv2DLayer_h

#include "Layers.h"

namespace smarties
{

#define ALIGNSPEC __attribute__(( aligned(VEC_WIDTH) ))

// Im2MatLayer gets as input an image of sizes InX * InY * InC
// and prepares the output for convolution with a filter of size KnY * KnX * KnC
// and output an image of size OpY * OpX * KnC
template
< typename func,
  int InX, int InY, int InC, //input image: x:width, y:height, c:color channels
  int KnX, int KnY, int KnC, //filter:      x:width, y:height, c:color channels
  int Sx, int Sy, int Px, int Py, // stride and padding x/y
  int OpX, int OpY //output img: x:width, y:height, same color channels as KnC
>
struct Conv2DLayer: public Layer
{
  using Input = ALIGNSPEC nnReal[InC][InY][InX];
  using Output = ALIGNSPEC nnReal[KnC][OpY][OpX];
  using Kernel = ALIGNSPEC nnReal[InC][KnC][KnY][KnX];
  static constexpr int inp_size = InC*InX*InY;
  static constexpr int out_size = KnC*OpY*OpX;
  const Uint link;

  void requiredParameters(std::vector<Uint>& nWeight,
                          std::vector<Uint>& nBiases ) const override {
    nBiases.push_back(out_size);
    nWeight.push_back(InC * KnC * KnY * KnX);
  }
  void requiredActivation(std::vector<Uint>& sizes,
                          std::vector<Uint>& bOutputs,
                          std::vector<Uint>& bInputs) const override {
    sizes.push_back(out_size);
    bOutputs.push_back(bOutput);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const std::vector<Real> init) override { }

  Conv2DLayer(int _ID, bool bO, Uint iL): Layer(_ID, out_size, bO), link(iL)
  {
    spanCompInpGrads = inp_size;
    static_assert(InC>0 && InY>0 && InX>0, "Invalid input image size");
    static_assert(KnC>0 && KnY>0 && KnX>0, "Invalid kernel size");
    static_assert(OpY>0 && OpX>0, "Invalid output image size");
    static_assert(Px>=0 && Py>=0 && "Invalid padding");
    static_assert(Sx>0 && Sy>0 && "Invalid stride");
  }

  std::string printSpecs() const override {
    std::ostringstream o;
    auto nonlin = std::make_unique<func>();
    o<<"("<<ID<<") "<<nonlin->name()
     <<"Conv Layer with Input:["<<InC<<"x"<<InY<<"x"<<InX
     <<"] Filter:["<<InC<<"x"<<KnC<<"x"<<KnY<<"x"<<KnX
     <<"] Output:["<<KnC<<"x"<<OpY<<"x"<<OpX
     <<"] Stride:["<<Sx<<"x"<<Sy <<"] Padding:["<<Px<<"x"<<Py
     <<"] linked to Layer:"<<ID-link<<"\n";
    return o.str();
  }

  void forward(const Activation*const prev,
               const Activation*const curr,
               const Parameters*const para) const override
  {
    // clean up memory space of lin_out. Why? Because padding, that's why.
    memcpy(curr->X(ID), para->B(ID), out_size * sizeof(nnReal));
    // Convert pointers to a reference to multi dim arrays for easy access:
    const Input & __restrict__ INP = * (Input*) curr->Y(ID-link);
    Output & __restrict__ LINOUT = * (Output*) curr->X(ID);
    const Kernel & __restrict__ K = * (Kernel*) para->W(ID);

    for (int ic = 0; ic < InC; ++ic) for (int oc = 0; oc < KnC; ++oc)
    for (int oy = 0; oy < OpY; ++oy) for (int ox = 0; ox < OpX; ++ox)
    for (int fy = 0; fy < KnY; ++fy) for (int fx = 0; fx < KnX; ++fx) {
      //starting position along input map for convolution with kernel
      const int ix = ox*Sx - Px + fx; //index along input map of
      const int iy = oy*Sy - Py + fy; //the convolution op
      //padding: skip addition if outside input boundaries
      if (ix < 0 || ix >= InX || iy < 0 || iy >= InY) continue;
      OUT[oc][oy][ox] += K[ic][oc][fy][fx] * INP[ic][iy][ix];
    }
    func::_eval(curr->X(ID), curr->Y(ID), KnC * OpY * OpX);
  }

  void bckward(const Activation*const prev,
               const Activation*const curr,
               const Activation*const next,
               const Parameters*const grad,
               const Parameters*const para) const override
  {
    { // premultiply with derivative of non-linearity & grad of bias
            nnReal* const deltas = curr->E(ID);
            nnReal* const grad_b = grad->B(ID);
      const nnReal* const suminp = curr->X(ID);
      const nnReal* const outval = curr->Y(ID);
      for(int o = 0; o < out_size; ++o) {
        deltas[o] *= func::_evalDiff(suminp[o], outval[o]);
        grad_b[o] += deltas[o];
      }
    }

    // Output is d Loss d Input, same size as INP before:
    Input  & __restrict__ dLdINP = * (Input *) curr->E(ID-link);
    Kernel & __restrict__ dLdK   = * (Kernel*) grad->W(ID);
    // Input is d Loss d Output, same size as OUT before:
    const Input  & __restrict__ INP    = * (Input *) curr->Y(ID-link);
    const Output & __restrict__ dLdOUT = * (Output*) curr->E(ID);
    const Kernel & __restrict__ K      = * (Kernel*) para->W(ID);

    for (int ic = 0; ic < InC; ++ic) for (int oc = 0; oc < KnC; ++oc)
    for (int oy = 0; oy < OpY; ++oy) for (int ox = 0; ox < OpX; ++ox)
    for (int fy = 0; fy < KnY; ++fy) for (int fx = 0; fx < KnX; ++fx) {
      //starting position along input map for convolution with kernel
      const int ix = ox*Sx - Px + fx; //index along input map of
      const int iy = oy*Sy - Py + fy; //the convolution op
      //padding: skip addition if outside input boundaries
      if (ix < 0 || ix >= InX || iy < 0 || iy >= InY) continue;
      dLdK[ic][oc][fy][fx] += dLdOUT[oc][oy][ox] * INP[ic][iy][ix];
      dLdINP[ic][iy][ix] += dLdOUT[oc][oy][ox] * K[ic][oc][fy][fx];
    }
  }

  void initialize(std::mt19937& G, const Parameters*const W,
                  Real initializationFac) const override
  {
    const nnReal fac = (initializationFac>0) ? initializationFac : 1;
    const int nAdded = InC * KnX * KnY;
    const nnReal init = fac * func::_initFactor(nAdded, KnC);
    std::uniform_real_distribution<nnReal> dis(-init, init);
    nnReal* const biases = W->B(ID);
    nnReal* const weight = W->W(ID);
    assert(W->NB(ID) == out_size);
    assert(W->NW(ID) == Kn_X * Kn_Y * Kn_C * In_C);
    for(Uint o=0; o<para->NB(ID); ++o) biases[o] = dis(G);
    for(Uint o=0; o<para->NW(ID); ++o) weight[o] = dis(G);
  }

  void orthogonalize(const Parameters*const para) const {}
};

#undef ALIGNSPEC

} // end namespace smarties
#endif // smarties_Quadratic_term_h
