//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Core/Engine.h"

int main (int argc, char** argv)
{
  smarties::Engine e(argc, argv);
  if( e.parse(argc, argv) ) return 1;
  e.init();
  e.run();
  return 0;
}
