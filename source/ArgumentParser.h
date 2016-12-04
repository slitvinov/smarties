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
#include <getopt.h>
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
        OptionStruct(char _shortOpt, string _longOpt, Types _type,
													 string _description, T* _val, T _defVal) :
        shortOpt(_shortOpt), longOpt(_longOpt), type(_type),
				description(_description), value((void*)_val)
        {
            *_val = _defVal;
        }

        OptionStruct() {};
        ~OptionStruct() {};

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
		~Parser() {delete [] long_options;}
		Parser(const std::vector<OptionStruct> optionsMap);
		void parse(int argc, char * const * argv, bool verbose = false);
	};
}
