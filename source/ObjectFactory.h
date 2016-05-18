/*
 *  ObjectFactory.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <string>
#include <vector>

#include "Agents/Agent.h"
#include "Environments/Environment.h"
//#include "Agents/ExternalAgent.h"

using namespace std;

class ObjectFactory
{
private:
    Settings settings;
	string filename;
	inline string _parse(string source, string pattern, bool req = true);
	inline int    _parseInt(string source, string pattern, bool req = true);
	inline Real _parseReal(string source, string pattern, bool req = true);
	
	
public:
	ObjectFactory(Settings & settings) : settings(settings), filename(settings.configFile) {};
	Environment* createEnvironment(int rank, int index);
};
