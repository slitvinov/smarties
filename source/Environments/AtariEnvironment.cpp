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

void AtariEnvironment::predefinedNetwork(Network* &net, Optimizer* &opt) const
{
  assert(net == nullptr && opt == nullptr);
  Builder net(settings);
  net.addInput( 84 * 84 * 4 );
  // CNN is entirely templated for speed!
  net.addConv2d<PRelu, //nonlineariy
                84, 84, 4,  // size of iunput x, y, c
                8, 8, 32,   // size of kernel x, y, c
                4, 4>();    // size of stride x, y

  net.addConv2d<PRelu, //nonlineariy
                20, 20, 32, // size of iunput x, y, c
                4, 4, 64,   // size of kernel x, y, c
                2, 2>();    // size of stride x, y

  net.addConv2d<PRelu, //nonlineariy
                9, 9, 64,   // size of iunput x, y, c
                3, 3, 64,   // size of kernel x, y, c
                1, 1>();    // size of stride x, y
  net = build.build();
  opt = build.opt;
}
