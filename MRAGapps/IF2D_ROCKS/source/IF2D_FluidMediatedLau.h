/*
 *  untitled.h
 *  IF2D_ROCKS
 *
 *  Created by Chloe Mimeau on 4/1/11.
 *  Copyright 2011 ETHZ. All rights reserved.
 *
 */

#pragma once

#include "IF2D_Headers.h"
#include "IF2D_Types.h"
#include "IF2D_FloatingObstacleOperator.h"
#include "IF2D_VelocityOperator.h"
#include "IF2D_PenalizationOperator.h"
#include "IF2D_AdvectionOperator.h"
#include "IF2D_DiffusionOperator.h"
#include "IF2D_KillVortRightBoundaryOperator.h"
#include "IF2D_ObstacleOperator.h"

#include "IF2D_LauFishSmart.h"

// for memory usage
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

namespace ComputationDiagnostics
{
  static void print_memory_usage(long & peak_rss_bytes, long & current_rss_bytes)
  {
    peak_rss_bytes = -1;
    current_rss_bytes=-1;
        
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
        
    peak_rss_bytes = rusage.ru_maxrss*1024;
    //printf("peak resident set size = %ld bytes (%.2lf Mbytes)\n", peak_rss_bytes, peak_rss_bytes/(1024.0*1024.0));
        
    long rss = 0;
    FILE* fp = NULL;
    if ( (fp = fopen( "/proc/self/statm", "r" )) == NULL ) {
      return;
    }
        
    if ( fscanf( fp, "%*s%ld", &rss ) != 1 ) {
      fclose( fp );
      return;
    }
    fclose( fp );
        
    current_rss_bytes = rss * sysconf( _SC_PAGESIZE);
    //printf("current resident set size = %ld bytes (%.2lf Mbytes)\n", current_rss_bytes, current_rss_bytes/(1024.0*1024.0));
  }
}

class IF2D_FluidMediatedLau: public IF2D_Test
{	
protected:
	//"constants" of the sim
	int BPD, JUMP, LMAX, ADAPTFREQ, SAVEFREQ, RAMP, MOLLFACTOR, nbObstacle;
	Real DUMPFREQ, RE, CFL, LCFL, RTOL, CTOL, LAMBDA, D, TEND, Uinf[2], nu, LAMBDADT, XPOS, YPOS, epsilon, FC;
    Real d, xm, ym, tau, angle, T;
	bool bPARTICLES, bUNIFORM, bCORRECTION, bRESTART, bREFINEOMEGAONLY, bFMMSKIP, bADAPTVEL, bLEARNING;
	string sFMMSOLVER, sOBSTACLE, sRIGID_INLET_TYPE;
	
	//state of the sim
	double t;
	long unsigned int step_id;
	
	ArgumentParser parser;
	
	Grid<W,B> * grid;
	
	Refiner * refiner;
	Compressor * compressor;
	
	BlockFWT<W, B, vorticity_projector, false, 1> fwt_omega;
	BlockFWT<W, B, velocity_projector, false, 2> fwt_velocity;
	BlockFWT<W, B, obstacle_projector, false, 1> fwt_obstacle;
	BlockFWT<W, B, vorticityANDvelocityANDchi_projector, false, 4> fwt_wuvx;
	BlockFWT<W, B, vorticityANDvelocity_projector, false, 3> fwt_wuv;
	
	set<int> _getBoundaryBlockIDs();
	
	void _dump(string filename);
	void _dump();
	
	void _ic(Grid<W,B>& grid);
	void _restart();
	void _save();
	
	virtual Real _initial_dt(int nsteps);
	virtual void _tnext(double &tnext, double& tnext_dump, double& tend);
	void _printMemoryConsumption();
	void _refine(bool bUseIC);
	void _compress(bool bUseIC);
	
	IF2D_VelocityOperator * velsolver;
	IF2D_PenalizationOperator * penalization;
	IF2D_AdvectionOperator * advection;
	IF2D_DiffusionOperator * diffusion;
    IF2D_LauFishSmart * myObstacle;
	
	Profiler profiler;
	bool bUSEPOTENTIAL;
	Real charLength, charVel;
	IF2D_VelocityOperator * potsolver;
	bool bUSEKILLVORT;
	int KILLVORT;
	IF2D_KillVortRightBoundaryOperator * killVort;
	bool bUSEOPTIMIZER;
	Real TBOUND;
	
	double reward,state;    
	int action;
    int nbReset;
    bool bReset;

    void saveLearning();

public:	
	IF2D_FluidMediatedLau(const int argc, const char ** argv);
	~IF2D_FluidMediatedLau();
	virtual void run();
        void paint(){}
    void act(const int action);
    double getState();
    double getReward();
    double getTime();
	void reset();

};
