#include <stdio.h>
#include <stdlib.h> /* free() */
#include "cmaes_interface.h"
#include "Communicator.h"
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <sstream>
#define VERBOSE 0
/*#define _RESTART_*/
#define _IODUMP_ 0
#define JOBMAXTIME	0

#include "cmaes_learn.h"
#include "fitfun.h"

void write_cmaes_perf::write( const int thrid ){
	char filename[256];
	sprintf(filename, "cma_perf_%02d.dat", thrid);
	FILE *fp = fopen(filename, "a");
	fprintf(fp, "dim func nstep dist feval\n");
	fclose(fp);
}

void write_cmaes_perf::write( const int thrid, const int func_dim, int func_id, int step, const double final_dist, double ffinal ){
	//print to file dim, function ID, nsteps, distance from opt, function value
	char filename[256];
	sprintf(filename, "cma_perf_%02d.dat", thrid);
	FILE *fp = fopen(filename, "a");
	fprintf(fp, "%d %d %d %e %e\n", func_dim, func_id, step, final_dist, ffinal);
	fclose(fp);
}



bool evaluate_and_update( cmaes_t* const evo, double* const*  pop, double *arFunvals, int* const info  ){
	
	int lambda = evo->sp.lambda;
	int func_dim = evo->sp.N;


	printf("%lf\n",arFunvals[0]);

	for (int i = 0; i < lambda; i++)
		fitfun(pop[i], func_dim, &arFunvals[i], info);

	printf("%lf\n",arFunvals[0]);

	cmaes_UpdateDistribution(evo, arFunvals);
	
	if(evo->isStuck == 1) return true;

	return false;
				
}
				



bool check_for_nan_inf(cmaes_t* const evo, double* const* pop ){
	
	int lambda   = evo->sp.lambda;
	int func_dim = evo->sp.N;

	bool foundnan = false;
	
	for (int i = 0; i < lambda; ++i)
		for (int j = 0; j < func_dim; j++)
			if (std::isnan(pop[i][j]) || std::isinf(pop[i][j]))
				foundnan = true;

	if(foundnan){
		fprintf(stderr, "It was nan all along!!!\n");
		evo->isStuck = 1;
		return true;
	}

	return false;
}






void actions_to_cma( double* const actions, int act_dim,  cmaes_t* const evo, 
					int *lambda, double *lambda_fac, const int lambda_0, double **arFunvals ){
	
	int func_dim = evo->sp.N;

	evo->sp.ccov1   = actions[0]; //rank 1 covariance update
	
	if (act_dim>1) evo->sp.ccovmu  = actions[1]; //rank mu covariance update
	if (act_dim>2) evo->sp.ccumcov = actions[2]; //path update c_c
	if (act_dim>3) evo->sp.cs      = actions[3]; //step size control c_sigmai
	
	if (act_dim>4)	evo->sp.damps   = actions[4]; //step size control c_sigmai
	else 			update_damps(evo, func_dim, *lambda);
	
	if (act_dim>5){ 
		*lambda_fac	= actions[5]; //pop size ratio
		*lambda 	= floor( lambda_0*(*lambda_fac) );
		*lambda 	= *lambda < 4 ? 4 : *lambda;
		*lambda_fac = *lambda/(double) lambda_0;
		*arFunvals 	= cmaes_ChangePopSize(evo, *lambda);
	}

	//printf("selected action %f %f %f %f %f\n",
	//evo->sp.ccov1,evo->sp.ccovmu,evo->sp.ccumcov,evo->sp.cs,lambda_fac);
	//fflush(0);
}








void copy_state( std::vector<double>& state, std::vector<double> from_state ){
	
	for( unsigned int i=0; i< state.size(); i++)
		state[i]=from_state[i];


}


void dump_curgen( double* const* pop, double *arFunvals, int step, int lambda, int func_dim ){
	char filename[256];
	sprintf(filename, "curgen_db_%03d.txt", step);
	FILE *fp = fopen(filename, "w");
	
	for (int i = 0; i < lambda; i++){
		for (int j = 0; j < func_dim; j++)
			fprintf(fp, "%.6le ", pop[i][j]);
		fprintf(fp, "%.6le\n", arFunvals[i]);
	}
	fclose(fp);
}


void print_best_ever( cmaes_t* const evo,  int step ){
	int func_dim = evo->sp.N;
	const double *xbever = cmaes_GetPtr(evo, "xbestever");
	double fbever = cmaes_Get(evo, "fbestever");
	
	printf("BEST @ %5d: ", step);
	for (int i = 0; i < func_dim; i++)
		printf("%25.16lf ", xbever[i]);
	printf("%25.16lf\n", fbever);
}



void update_damps(cmaes_t* const evo,const int N, const int lambda)
{
	evo->sp.damps =
		(1 + 2*std::max(0., sqrt((evo->sp.mueff-1.)/(N+1.)) - 1 ) )     /* basic factor */
		* std::max(0.3, 1. -                                       /* modify for short runs */
				(double)N / (1e-6+std::min(evo->sp.stopMaxIter, evo->sp.stopMaxFunEvals/lambda)))
		+ evo->sp.cs;
}



bool resample( cmaes_t* const evo, double* const* pop, double* const lower_bound, double* const upper_bound){
	int lambda = evo->sp.lambda;
	int func_dim = evo->sp.N;
	int safety = 0;
	for (int i = 0; i < lambda; ++i){
		while (!is_feasible(pop[i],lower_bound,upper_bound,func_dim) && safety++ < 1e3){
			cmaes_ReSampleSingle(evo, i);
			if(evo->isStuck == 1) return true;
		}
	}

	return false;
}




int is_feasible(double* const pop, double* const lower_bound, double* const upper_bound, int dim)
{
	int good;
	for (int i = 0; i < dim; i++) {
		if (std::isnan(pop[i]) || std::isinf(pop[i])){
			printf("Sampled nan: FU cmaes \n"); 
			abort(); 
		}
		
		good = (lower_bound[i] <= pop[i]) && (pop[i] <= upper_bound[i]);
		if (!good) return 0;
	}
	return 1;
}




void update_state(cmaes_t* const evo, double* const state, double* oldFmedian, double* oldXmean)
{
	int func_dim = evo->sp.N;
	double eps = 1e-16;
	double* xMean = cmaes_GetNew(evo, "xmean");
	
	double xProgress = 0.;
	for (int i=0; i<func_dim; i++)
		xProgress += pow(xMean[i]-oldXmean[i],2);
	
	const double fmedian = cmaes_Get(evo, "fmedian");
	const double fProgress = (fmedian - *oldFmedian)
		/(fabs(fmedian) + fabs(*oldFmedian));


	//ratio between standard deviations
	const double d1 = evo->meandiagC < 0 ? evo->meandiagC-eps : evo->meandiagC+eps;
	const double d2 = evo->meanEW < 0 ? evo->meanEW-eps : evo->meanEW+eps;
	/*printf("%g %g %g %g %g %g\n",
	  d1,d2,
	  evo->mindiagC, evo->maxdiagC,
	  evo->minEW, evo->maxEW);
	  fflush(0);*/
	const double ratio1 = (evo->mindiagC/d1);
	//ratio between eigenvalues
	const double ratio2 = (evo->minEW/d2);
	const double ratio3 = (evo->maxdiagC/d1);
	//ratio between eigenvalues
	const double ratio4 = (evo->maxEW/d2);
	//distinction makes sense if function is rotated

	//prevent nans/infs from infecting the delicate snowflake that is the RL code
	if (std::isnan(ratio1) || std::isinf(ratio1))
	{ perror("Ratio1 is nan, FU CMAES \n"); abort(); }
	if (std::isnan(ratio2) || std::isinf(ratio2))
	{ perror("Ratio2 is nan, FU CMAES \n"); abort(); }
	if (std::isnan(xProgress) || std::isinf(xProgress))
	{ perror("xProgress is nan, FU CMAES \n"); abort(); }
	if (std::isnan(fProgress) || std::isinf(fProgress))
	{ perror("fProgress is nan, FU CMAES \n"); abort(); }
	if (std::isnan(evo->trace) || std::isinf(evo->trace))
	{ perror("evo->trace is nan, FU CMAES \n"); abort(); }

	state[0] = ratio1;
	state[1] = ratio2;
	state[2] = ratio3;
	state[3] = ratio4;
	state[4] = sqrt(xProgress);
	state[5] = fProgress;
	state[6] = (double)func_dim;
	state[7] = evo->trace;

	//advance:
	*oldFmedian = fmedian;
	for (int i=0; i<func_dim; i++) oldXmean[i] = xMean[i];
	free(xMean);
}
