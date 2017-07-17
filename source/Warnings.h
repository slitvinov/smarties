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
#include <mutex>

namespace ErrorHandling
{
 static std::mutex warn_mutex;
 enum Debug_level { SILENT, WARNINGS, SCHEDULER, ENVIRONMENT, NETWORK, COMMUNICATOR, LEARNERS, TRANSITIONS };

#if defined(NDEBUG)
	static const Debug_level level = SILENT;
#elif defined(SMARTIES_DEBUG)
	static const Debug_level level = SMARTIES_DEBUG;
#else
	//static const Debug_level level = WARNINGS;
	static const Debug_level level = SCHEDULER;
#endif

#define    die(format)      {fprintf(stderr,format); fflush(stdout); fflush(stderr); fflush(0); MPI_Abort(MPI_COMM_WORLD, 1);}
#define   _die(format, ...) {fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr); fflush(0); MPI_Abort(MPI_COMM_WORLD, 1);}
#define  error(format, ...) {lock_guard<mutex> wlock(ErrorHandling::warn_mutex);  fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr);}

#define   warn(format)	{if(ErrorHandling::level >= ErrorHandling::WARNINGS) { lock_guard<mutex> wlock(ErrorHandling::warn_mutex); fprintf(stdout,format); fflush(stdout); fflush(stderr);  fflush(0);} }
#define  _warn(format, ...)	{if(ErrorHandling::level >= ErrorHandling::WARNINGS) { lock_guard<mutex> wlock(ErrorHandling::warn_mutex); fprintf(stdout,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr); fflush(0);} }
#define debugS(format, ...)	{if(ErrorHandling::level == ErrorHandling::SCHEDULER) { lock_guard<mutex> wlock(ErrorHandling::warn_mutex); fprintf(stdout,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr); fflush(0);} }
#define debugE(format, ...)	{if(ErrorHandling::level == ErrorHandling::ENVIRONMENT) { lock_guard<mutex> wlock(ErrorHandling::warn_mutex); fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr); fflush(0);} }
#define debugN(format, ...)	{if(ErrorHandling::level == ErrorHandling::NETWORK) { lock_guard<mutex> wlock(ErrorHandling::warn_mutex); fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr); fflush(0);} }
#define debugC(format, ...)	{if(ErrorHandling::level == ErrorHandling::COMMUNICATOR) { lock_guard<mutex> wlock(ErrorHandling::warn_mutex); fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr); fflush(0);} }
#define debugL(format, ...)	{if(ErrorHandling::level == ErrorHandling::LEARNERS) { lock_guard<mutex> wlock(ErrorHandling::warn_mutex); fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr); fflush(0);} }
#define debugT(format, ...)	{if(ErrorHandling::level == ErrorHandling::TRANSITIONS) { lock_guard<mutex> wlock(ErrorHandling::warn_mutex); fprintf(stderr,format, ##__VA_ARGS__); fflush(stdout); fflush(stderr); fflush(0);} }
}
