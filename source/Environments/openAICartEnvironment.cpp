/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "openAICartEnvironment.h"
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <cstdio>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <iostream>
#include <algorithm>
#include <stdio.h>
using namespace std;

openAICartEnvironment::openAICartEnvironment(const int _nAgents, const string _execpath, Settings & _s) :
Environment(_nAgents, _execpath, _s), allSenses(_s.senses==0)
{
//   cheaperThanNetwork=false;
}
