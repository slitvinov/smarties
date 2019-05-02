//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Warnings_h
#define smarties_Warnings_h

#include "MPIUtilities.h"
#include <mutex>
//#include <sstream>
//#include <stdarg.h>

namespace smarties
{
namespace Warnings
{
static std::mutex warn_mutex;
enum Debug_level { SILENT, WARNINGS, SCHEDULER, ENVIRONMENT, NETWORK, COMMUNICATOR, LEARNERS, TRANSITIONS };

static constexpr Debug_level level = WARNINGS;
//static constexpr Debug_level level = LEARNERS;
//static constexpr Debug_level level = SCHEDULER;

static inline void flushAll() { fflush(stdout); fflush(stderr); fflush(0); }

#ifdef MPI_VERSION
static inline void abortAll() { MPI_Abort(MPI_COMM_WORLD, 1); }
#else
static inline void abortAll() { abort(); }
#endif

#define SMARTIES_LOCKCOMM std::lock_guard<std::mutex> wlock(Warnings::warn_mutex)

inline static void printfmt(char*const p, const int N, const char*const a, ... )
{
  va_list args;
  va_start (args, a);
  vsnprintf (p, N, a, args);
  va_end (args);
}
// THESE ARE ALL DEFINES ALLOWING PRINTING FILE, FUNC, LINE

#define    die(format)      do { \
  SMARTIES_LOCKCOMM; const auto wrnk = MPIworldRank(); \
  fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
  " FATAL",format); Warnings::flushAll(); Warnings::abortAll(); } while(0)

#define   _die(format, ...) do { \
  SMARTIES_LOCKCOMM; const auto wrnk = MPIworldRank(); \
  char BUF[512]; Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
  fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
  " FATAL", BUF); Warnings::flushAll(); Warnings::abortAll(); } while(0)

#define  error(format, ...) do { \
  SMARTIES_LOCKCOMM; const auto wrnk = MPIworldRank(); \
  char BUF[512]; Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
  fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
  " ERROR", BUF); Warnings::flushAll(); } while(0)

#define   warn(format)  do { \
  if(Warnings::level >= Warnings::WARNINGS) { \
    SMARTIES_LOCKCOMM; const auto wrnk = MPIworldRank(); \
    fprintf(stdout,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " WARNING",format); Warnings::flushAll(); } } while(0)

#define  _warn(format, ...)  do { \
  if(Warnings::level >= Warnings::WARNINGS) { \
    SMARTIES_LOCKCOMM; const auto wrnk = MPIworldRank(); \
    char BUF[512]; Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stdout,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " WARNING", BUF); Warnings::flushAll(); } } while(0)

#define debugS(format, ...)  do { \
  if(Warnings::level == Warnings::SCHEDULER) { \
    SMARTIES_LOCKCOMM; const auto wrnk = MPIworldRank(); \
    char BUF[512]; Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stdout,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); Warnings::flushAll(); } } while(0)

#define debugE(format, ...)  do { \
  if(Warnings::level == Warnings::ENVIRONMENT) { \
    SMARTIES_LOCKCOMM; const auto wrnk = MPIworldRank(); \
    char BUF[512]; Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); Warnings::flushAll(); } } while(0)

#define debugN(format, ...)  do { \
  if(Warnings::level == Warnings::NETWORK) { \
    SMARTIES_LOCKCOMM; const auto wrnk = MPIworldRank(); \
    char BUF[512]; Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); Warnings::flushAll(); } } while(0)

#define debugC(format, ...)  do { \
  if(Warnings::level == Warnings::COMMUNICATOR) { \
    SMARTIES_LOCKCOMM; const auto wrnk = MPIworldRank(); \
    char BUF[512]; Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); Warnings::flushAll(); } } while(0)

#define _debugL(format, ...)  do { \
  if(Warnings::level == Warnings::LEARNERS) { \
    SMARTIES_LOCKCOMM; const auto wrnk = MPIworldRank(); \
    char BUF[512]; Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); Warnings::flushAll(); } } while(0)

#define debugL(format)  do { \
  if(Warnings::level == Warnings::LEARNERS) { \
    SMARTIES_LOCKCOMM; const auto wrnk = MPIworldRank(); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " LEARNER",format); Warnings::flushAll(); } } while(0)

#define debugT(format, ...)  do { \
  if(Warnings::level == Warnings::TRANSITIONS) { \
    SMARTIES_LOCKCOMM; const auto wrnk = MPIworldRank(); \
    char BUF[512]; Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); Warnings::flushAll(); } } while(0)

} // end namespace Warnings

} // end namespace smarties
#endif
