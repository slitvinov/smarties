/*
 *  ArgumentParser.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Settings.h"
#include <map>
using namespace std;

namespace ArgumentParser
{
	enum Types { NONE, INT, REAL, CHAR, STRING };
	
	struct OptionStruct
	{
		char   shortOpt;
		string longOpt;
		Types  type;
		string description;
		void*  value;
        
        template <typename T>
        OptionStruct(char shortOpt, string longOpt, Types type, string description, T* val, T defVal) :
        shortOpt(shortOpt), longOpt(longOpt), type(type), description(description)
        {
            value = (void*)val;
            *val = defVal;
        }
        
        OptionStruct() {};

    };

	class Parser
	{
	private:
		int nOpt; 
		vector<OptionStruct> opts;
		map<char, OptionStruct> optsMap;
		struct option* long_options;
		string ctrlString;
		
	public:
		
		Parser(const std::vector<OptionStruct> optionsMap);
		void parse(int argc, char * const * argv, bool verbose = false);
	};
}
