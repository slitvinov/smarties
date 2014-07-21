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

using namespace std;

class ObjectFactory
{
private:
	string filename;
	
	inline string _parse(string source, string pattern, bool req = true);
	inline int    _parseInt(string source, string pattern, bool req = true);
	inline double _parseDouble(string source, string pattern, bool req = true);
	
	
public:
	ObjectFactory(string newFilename) : filename(newFilename) {};
	System getAgentVector(int argc = 0, const char** argv = NULL);
};