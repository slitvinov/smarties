/*
 *  Settings.h
 *  rl
 *
 *  Created by Dmitry Alexeev and extended by Guido Novati on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#pragma once
#include "Bund.h"

namespace ErrorHandling
{
 enum Debug_level { SILENT, WARNINGS, SCHEDULER, ENVIRONMENT, NETWORK, COMMUNICATOR, LEARNERS, TRANSITIONS };

#if defined(NDEBUG)
	static const Debug_level level = SILENT;
#elif defined(SMARTIES_DEBUG)
	static const Debug_level level = SMARTIES_DEBUG;
#else
	static const Debug_level level = WARNINGS;
	//static const Debug_level level = SCHEDULER;
#endif

#define    die(format)      {fprintf(stderr,format);                fflush(stdout); fflush(stderr); MPI_Abort(MPI_COMM_WORLD, 1);}
#define   _die(format, ...) {fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr); MPI_Abort(MPI_COMM_WORLD, 1);}
#define  error(format, ...) {fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr);}

#define   warn(format, ...)	{if(ErrorHandling::level >= ErrorHandling::WARNINGS)     fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr);}
#define debugS(format, ...)	{if(ErrorHandling::level == ErrorHandling::SCHEDULER)    fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr);}
#define debugE(format, ...)	{if(ErrorHandling::level == ErrorHandling::ENVIRONMENT)  fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr);}
#define debugN(format, ...)	{if(ErrorHandling::level == ErrorHandling::NETWORK)      fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr);}
#define debugC(format, ...)	{if(ErrorHandling::level == ErrorHandling::COMMUNICATOR) fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr);}
#define debugL(format, ...)	{if(ErrorHandling::level == ErrorHandling::LEARNERS)     fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr);}
#define debugT(format, ...)	{if(ErrorHandling::level == ErrorHandling::TRANSITIONS)  fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr);}
}
