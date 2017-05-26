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
extern Uint debugLvl;
#define    die(format, ...) {fprintf(stderr, format, ##__VA_ARGS__); MPI_Abort(MPI_COMM_WORLD, 1);}
#define  error(format, ...) fprintf(stderr, format, ##__VA_ARGS__)

#define   warn(format, ...)	{if (debugLvl > 0) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define  _info(format, ...)	{if (debugLvl > 1) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}

#define  debug(format, ...)	{if (debugLvl > 2) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug1(format, ...)	{if (debugLvl > 3) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug2(format, ...)	{if (debugLvl > 4) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug3(format, ...)	{if (debugLvl > 5) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug4(format, ...)	{if (debugLvl > 6) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug5(format, ...)	{if (debugLvl > 7) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug6(format, ...)	{if (debugLvl > 8) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug7(format, ...)	{if (debugLvl > 9) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug8(format, ...)	{if (debugLvl >10) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug9(format, ...)	{if (debugLvl >11) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
}
