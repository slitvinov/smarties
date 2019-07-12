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
struct Im2MatLayer: public Layer
{
  using Input = ALIGNSPEC nnReal[InC][InY][InX];
  using Output = ALIGNSPEC nnReal[KnC][OpY][OpX];
  using Kernel = ALIGNSPEC nnReal[InC][KnC][KnY][KnX];
  using OutMat = ALIGNSPEC nnReal[InC][Kn_Y][Kn_X][OutY][OutX];
  static constexpr int inp_size = InC*InX*InY;
  static constexpr int out_size = InC*Kn_Y*Kn_X*OutY*OutX;
  const Uint link;

  void requiredParameters(std::vector<Uint>& nWeight,
                          std::vector<Uint>& nBiases ) const override {
    nBiases.push_back(0);
    nWeight.push_back(0);
  }
  void requiredActivation(std::vector<Uint>& sizes,
                          std::vector<Uint>& bOutputs,
                          std::vector<Uint>& bInputs) const override {
    sizes.push_back(out_size);
    bOutputs.push_back(bOutput);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const std::vector<Real> init) override { }

  Conv2DLayer(int _ID, bool bO, Uint iL): Layer(_ID, OpX*OpY*KnC, bO), link(iL)
  {
    spanCompInpGrads = InC * InY * InX;
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
    // Convert pointers to a reference to multi dim arrays for easy access:
    const Input & __restrict__ INP = * (Input*) curr->Y(ID-link);
    Output & __restrict__ LINOUT = * (Output*) curr->X(ID);
    const Kernel & __restrict__ K = * (Kernel*) para->W(ID);

    for (int ic = 0; ic < InC; ++ic)
      for (int fy = 0; fy < KnY; ++fy) for (int fx = 0; fx < KnX; ++fx)
        for (int oy = 0; oy < OpY; ++oy) for (int ox = 0; ox < OpX; ++ox) {
          //index along input map of the convolution op:
          const int ix = ox*Sx -Px +fx, iy = oy*Sy -Py +fy;
          //padding: skip addition if outside input boundaries
          if (ix < 0 || ix >= InX || iy < 0 || iy >= InY) continue;
          OUT[ic][fy][fx][oy][ox] = INP[ic][iy][ix];
        }
  }

  void bckward(const Activation*const prev,
               const Activation*const curr,
               const Activation*const next,
               const Parameters*const grad,
               const Parameters*const para) const override
  {
    // Output is d Loss d Input, same size as INP before:
    Input  & __restrict__ dLdINP = * (Input *) curr->E(ID-link);
    Kernel & __restrict__ dLdK   = * (Kernel*) grad->W(ID);
    // Input is d Loss d Output, same size as OUT before:
    const Input  & __restrict__ INP    = * (Input *) curr->Y(ID-link);
    const Output & __restrict__ dLdOUT = * (Output*) curr->E(ID);
    const Kernel & __restrict__ K      = * (Kernel*) para->W(ID);

    for (int ic = 0; ic < InC; ++ic)
      for (int fy = 0; fy < KnY; ++fy) for (int fx = 0; fx < KnX; ++fx)
        for (int oy = 0; oy < OpY; ++oy) for (int ox = 0; ox < OpX; ++ox) {
          //index along input map of the convolution op:
          const int ix = ox*Sx -Px +fx, iy = oy*Sy -Py +fy;
          //padding: skip addition if outside input boundaries
          if (ix < 0 || ix >= InX || iy < 0 || iy >= InY) continue;
          dLdINP[ic][iy][ix] += dLdOUT[ic][fy][fx][oy][ox];
        }
  }

  void initialize(std::mt19937* const G, const Parameters*const para,
    Real initializationFac) const override
  {
    const nnReal fac = (initializationFac>0) ? initializationFac : 1;
    const int nAdded = InC * KnX * KnY;
    const nnReal init = fac * func::_initFactor(nAdded, KnC);
    std::uniform_real_distribution<nnReal> dis(-init, init);
    nnReal* const biases = para->B(ID);
    nnReal* const weight = para->W(ID);
    assert(para->NB(ID) == OutX * OutY * Kn_C);
    assert(para->NW(ID) == Kn_X * Kn_Y * Kn_C * In_C);
    for(Uint o=0; o<para->NB(ID); ++o) biases[o] = dis(*G);
    for(Uint o=0; o<para->NW(ID); ++o) weight[o] = dis(*G);
  }

  void orthogonalize(const Parameters*const para) const {}
};

#undef ALIGNSPEC

} // end namespace smarties
#endif // smarties_Quadratic_term_h
