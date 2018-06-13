//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the “CC BY-SA 4.0” license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "../Bund.h"
#include <mutex>
#include <stdarg.h>

namespace ErrorHandling
{
  static std::mutex warn_mutex;
  enum Debug_level { SILENT, WARNINGS, SCHEDULER, ENVIRONMENT, NETWORK, COMMUNICATOR, LEARNERS, TRANSITIONS };

  static constexpr Debug_level level = WARNINGS;

//#define PR_ERR(fd,a,...) fprintf(fd,a,__func__,__FILE__,__LINE__,##__VA_ARGS__);
#define FLUSHALL fflush(stdout); fflush(stderr); fflush(0);
#define KILLALL MPI_Abort(MPI_COMM_WORLD, 1);
#define LOCKCOMM lock_guard<mutex> wlock(ErrorHandling::warn_mutex);

inline static void printfmt(char*const p, const int N, const char*const a, ... )
{
  va_list args;
  va_start (args, a);
  vsnprintf (p, N, a, args);
  va_end (args);
}

// THESE ARE ALL DEFINES ALLOWING PRINTING FILE, FUNC, LINE

#define    die(format)      { LOCKCOMM  \
  int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
  fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
  " FATAL",format); FLUSHALL KILLALL }

#define   _die(format, ...) { LOCKCOMM  \
  int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
  char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
  fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
  " FATAL", BUF); FLUSHALL KILLALL }

#define  error(format, ...) { LOCKCOMM \
  int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
  char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
  fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
  " ERROR", BUF); FLUSHALL}

#define   warn(format)  { \
  if(ErrorHandling::level >= ErrorHandling::WARNINGS) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    fprintf(stdout,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " WARNING",format); FLUSHALL} }

#define  _warn(format, ...)  { \
  if(ErrorHandling::level >= ErrorHandling::WARNINGS) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stdout,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " WARNING", BUF); FLUSHALL} }

#define debugS(format, ...)  { \
  if(ErrorHandling::level == ErrorHandling::SCHEDULER) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stdout,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} }

#define debugE(format, ...)  { \
  if(ErrorHandling::level == ErrorHandling::ENVIRONMENT) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} }

#define debugN(format, ...)  { \
  if(ErrorHandling::level == ErrorHandling::NETWORK) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} }

#define debugC(format, ...)  { \
  if(ErrorHandling::level == ErrorHandling::COMMUNICATOR) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} }

#define _debugL(format, ...)  { \
  if(ErrorHandling::level == ErrorHandling::LEARNERS) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} }
#define debugL(format)  { \
  if(ErrorHandling::level == ErrorHandling::LEARNERS) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " LEARNER",format); FLUSHALL} }

#define debugT(format, ...)  { \
  if(ErrorHandling::level == ErrorHandling::TRANSITIONS) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} }

}
