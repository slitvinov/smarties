/*
 *  ErrorHandling.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <cstdlib>
#include <cstdio>

namespace ErrorHandling
{
	extern int debugLvl;
	
#define    die(format, ...) fprintf(stderr, format, ##__VA_ARGS__), abort()
#define  error(format, ...) fprintf(stderr, format, ##__VA_ARGS__)
	
#define   warn(format, ...)	{if (debugLvl > 0) fprintf(stderr, format, ##__VA_ARGS__);}
#define   info(format, ...)	{if (debugLvl > 1) fprintf(stderr, format, ##__VA_ARGS__);}

#define  debug(format, ...)	{if (debugLvl > 2) fprintf(stderr, format, ##__VA_ARGS__);}
#define debug1(format, ...)	{if (debugLvl > 3) fprintf(stderr, format, ##__VA_ARGS__);}
#define debug2(format, ...)	{if (debugLvl > 4) fprintf(stderr, format, ##__VA_ARGS__);}
#define debug3(format, ...)	{if (debugLvl > 5) fprintf(stderr, format, ##__VA_ARGS__);}
#define debug4(format, ...)	{if (debugLvl > 6) fprintf(stderr, format, ##__VA_ARGS__);}
#define debug5(format, ...)	{if (debugLvl > 7) fprintf(stderr, format, ##__VA_ARGS__);}
}