/*
 *  ArgumentParser.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <getopt.h>
#include <map>
#include <cstdlib>

#include "ErrorHandling.h"
#include "ArgumentParser.h"

using namespace ErrorHandling;

namespace ArgumentParser
{
	
	Parser::Parser(const std::vector<OptionStruct> optionsMap):opts(optionsMap)
	{
		ctrlString = "";
		nOpt = opts.size();
		long_options = new option[nOpt + 1];
		
		for (int i=0; i<nOpt; i++)
		{
			long_options[i].name = opts[i].longOpt.c_str();
			long_options[i].flag = NULL;
			long_options[i].val = opts[i].shortOpt;

			if (opts[i].type == NONE) long_options[i].has_arg = no_argument;
			else                      long_options[i].has_arg = required_argument;

			
			ctrlString += opts[i].shortOpt;
			if (opts[i].type != NONE) ctrlString += ':';
			
			if (optsMap.find(long_options[i].val) != optsMap.end())
				die("Duplicate short options in declaration, please correct the source code\n");
			else optsMap[long_options[i].val] = opts[i];
			
		}
		
		long_options[nOpt].has_arg = 0;
		long_options[nOpt].flag = NULL;
		long_options[nOpt].name = NULL;
		long_options[nOpt].val  = 0;
	}
	
	void Parser::parse(int argc, char * const * argv)
	{
		int option_index = 0;
		int c = 0;
		
		while((c = getopt_long (argc, argv, ctrlString.c_str(), long_options, &option_index)) != -1)
		{
			if (c == 0) continue;
			if (optsMap.find(c) == optsMap.end())
			{
				info("Available options:\n");
				
				for (int i=0; i<nOpt; i++)
				{
					OptionStruct& myOpt = opts[i];
					info("-%c\tor --%s : %s\n", myOpt.shortOpt, myOpt.longOpt.c_str(), myOpt.description.c_str());
				}
					
				die("Finishing program\n");
			}
			
			OptionStruct& myOpt = optsMap[c];
			
			switch (myOpt.type)
			{
				case NONE:
					*((bool*)myOpt.value) = true;
					break;
					
				case INT:
					*((int*)myOpt.value) = atoi(optarg);
					break;
					
				case UINT:
					*((unsigned int*)myOpt.value) = atoi(optarg);
					break;
					
				case DOUBLE:
					*((double*)myOpt.value) = atof(optarg);
					break;
					
				case STRING:
					*((string*)myOpt.value) = optarg;
					break;
					
			}
		}
		
		for (int i=0; i<nOpt; i++)
		{
			OptionStruct& myOpt = opts[i];
			warn("%s: ", myOpt.description.c_str());
			
			switch (myOpt.type)
			{
				case NONE:
					warn( ( *((bool*)myOpt.value)) ? "enabled" : "disabled" );
					break;
					
				case INT:
					warn("%d", *((int*)myOpt.value));
					break;
					
				case UINT:
					warn("%d", *((unsigned int*)myOpt.value));
					break;
					
				case DOUBLE:
					warn("%f", *((double*)myOpt.value));
					break;
					
				case STRING:
					warn("%s", ((string*)myOpt.value)->c_str());
					break;
			}
			
			warn("\n");
		}
			
	}
}
