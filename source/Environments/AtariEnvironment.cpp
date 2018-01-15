/*
 *  TwoFishEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "AtariEnvironment.h"
#include "../Network/Builder.h"

AtariEnvironment::AtariEnvironment(Settings& _sett): Environment(_sett)
{
  printf("AtariEnvironment.\n");
}

bool AtariEnvironment::predefinedNetwork(Builder & input_net) const
{
  assert(input_net.nInputs);
  // CNN is entirely templated for speed!
  input_net.addConv2d<PRelu, //nonlineariy
                  84, 84, 4,  // size of iunput x, y, c
                  8, 8, 32,   // size of kernel x, y, c
                  4, 4>();    // size of stride x, y

  input_net.addConv2d<PRelu, //nonlineariy
                  20, 20, 32, // size of iunput x, y, c
                  4, 4, 64,   // size of kernel x, y, c
                  2, 2>();    // size of stride x, y

  input_net.addConv2d<PRelu, //nonlineariy
                  9, 9, 64,   // size of iunput x, y, c
                  3, 3, 64,   // size of kernel x, y, c
                  1, 1>(true);    // size of stride x, y
  return true;
}
