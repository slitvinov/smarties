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

namespace smarties
{
namespace Warnings
{
enum Debug_level { SILENT, WARNINGS, SCHEDULER, ENVIRONMENT, NETWORK, COMMUNICATOR, LEARNERS, TRANSITIONS };

static constexpr Debug_level level = WARNINGS;
//static constexpr Debug_level level = LEARNERS;
//static constexpr Debug_level level = SCHEDULER;

void print_warning(const char * funcname, const char * filename,
                   int line, const char * fmt, ...);
void print_stacktrace();

#define    die(format)      do {                                               \
  using namespace smarties::Warnings;                                          \
  print_warning(__func__, __FILE__, __LINE__, format);                         \
  print_stacktrace(); MPI_Abort(MPI_COMM_WORLD, 1); } while(0)

#define   _die(format, ...) do {                                               \
  using namespace smarties::Warnings;                                          \
  print_warning(__func__, __FILE__, __LINE__, format, ##__VA_ARGS__);          \
  print_stacktrace(); MPI_Abort(MPI_COMM_WORLD, 1); } while(0)

#define   warn(format)  do { \
  if(Warnings::level == smarties::Warnings::WARNINGS) {                        \
    using namespace smarties::Warnings;                                        \
    print_warning(__func__, __FILE__, __LINE__, format);                       \
  } } while(0)

#define  _warn(format, ...)  do { \
  if(Warnings::level == smarties::Warnings::WARNINGS) {                        \
    using namespace smarties::Warnings;                                        \
    print_warning(__func__, __FILE__, __LINE__, format, ##__VA_ARGS__);        \
  } } while(0)

#define debugS(format, ...)  do { \
  if(Warnings::level == smarties::Warnings::SCHEDULER) {                       \
    using namespace smarties::Warnings;                                        \
    print_warning(__func__, __FILE__, __LINE__, format, ##__VA_ARGS__);        \
  } } while(0)

#define _debugL(format, ...)  do { \
  if(Warnings::level == smarties::Warnings::LEARNERS) {                        \
    using namespace smarties::Warnings;                                        \
    print_warning(__func__, __FILE__, __LINE__, format, ##__VA_ARGS__);        \
  } } while(0)

#define debugL(format)  do { \
  if(Warnings::level == smarties::Warnings::LEARNERS) {                        \
    using namespace smarties::Warnings;                                        \
    print_warning(__func__, __FILE__, __LINE__, format);                       \
  } } while(0)

} // end namespace Warnings

} // end namespace smarties
#endif
