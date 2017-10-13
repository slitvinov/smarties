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
#include <stdarg.h>

namespace ErrorHandling
{
 static std::mutex warn_mutex;
 enum Debug_level { SILENT, WARNINGS, SCHEDULER, ENVIRONMENT, NETWORK, COMMUNICATOR, LEARNERS, TRANSITIONS };

#if defined(NDEBUG)
  static const Debug_level level = SILENT;
#elif defined(SMARTIES_DEBUG)
  static const Debug_level level = SMARTIES_DEBUG;
#else
  static const Debug_level level = WARNINGS;
  //static const Debug_level level = SCHEDULER;
#endif


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
  fprintf(stderr,"%s(%s:%d)%s %s\n",__func__,__FILE__,__LINE__, \
  " FATAL",format); FLUSHALL KILLALL }

#define   _die(format, ...) { LOCKCOMM  \
  char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
  fprintf(stderr,"%s(%s:%d)%s %s\n",__func__,__FILE__,__LINE__, \
  " FATAL", BUF); FLUSHALL KILLALL }

#define  error(format, ...) { LOCKCOMM \
  char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
  fprintf(stderr,"%s(%s:%d)%s %s\n",__func__,__FILE__,__LINE__, \
  " ERROR", BUF); FLUSHALL}

#define   warn(format)  { \
  if(ErrorHandling::level >= ErrorHandling::WARNINGS) { LOCKCOMM \
    fprintf(stdout,"%s(%s:%d)%s %s\n",__func__,__FILE__,__LINE__, \
    " WARNING",format); FLUSHALL} }

#define  _warn(format, ...)  { \
  if(ErrorHandling::level >= ErrorHandling::WARNINGS) { LOCKCOMM \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stdout,"%s(%s:%d)%s %s\n",__func__,__FILE__,__LINE__, \
    " WARNING", BUF); FLUSHALL} }

#define debugS(format, ...)  { \
  if(ErrorHandling::level == ErrorHandling::SCHEDULER) { LOCKCOMM \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stdout,"%s(%s:%d)%s %s\n",__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} }

#define debugE(format, ...)  { \
  if(ErrorHandling::level == ErrorHandling::ENVIRONMENT) { LOCKCOMM \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"%s(%s:%d)%s %s\n",__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} }

#define debugN(format, ...)  { \
  if(ErrorHandling::level == ErrorHandling::NETWORK) { LOCKCOMM \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"%s(%s:%d)%s %s\n",__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} }

#define debugC(format, ...)  { \
  if(ErrorHandling::level == ErrorHandling::COMMUNICATOR) { LOCKCOMM \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"%s(%s:%d)%s %s\n",__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} }

#define debugL(format, ...)  { \
  if(ErrorHandling::level == ErrorHandling::LEARNERS) { LOCKCOMM \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"%s(%s:%d)%s %s\n",__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} }

#define debugT(format, ...)  { \
  if(ErrorHandling::level == ErrorHandling::TRANSITIONS) { LOCKCOMM \
    char BUF[256]; ErrorHandling::printfmt(BUF, 256, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"%s(%s:%d)%s %s\n",__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} }

}
