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

#include "cmaes_learn.h"
#include "fitfun.h"


void random_action( cmaes_t* const evo, std::mt19937 gen ){

	std::uniform_real_distribution<double> act0dist(.02,0.1);
	std::uniform_real_distribution<double> act1dist(.02,0.1);
	std::uniform_real_distribution<double> act2dist(.2,.8);
	std::uniform_real_distribution<double> act3dist(.2,.8);


	evo->sp.ccov1   = act0dist(gen);
	evo->sp.ccovmu  = act1dist(gen);
	evo->sp.ccumcov = act2dist(gen);
	evo->sp.cs      = act3dist(gen);
	update_damps( evo );

}



void write_cmaes_perf::write( const int thrid ){
	char filename[256];
	sprintf(filename, "cma_perf_%02d.dat", thrid);
	FILE *fp = fopen(filename, "a");
	fprintf(fp, "dim func nstep dist feval\n");
	fclose(fp);
}

void write_cmaes_perf::write( cmaes_t* const evo, const int thrid, const int func_dim, int func_id, int step, const double final_dist, double ffinal ){
	//print to file dim, function ID, nsteps, distance from opt, function value
	char filename[256];
	FILE *fp;

	sprintf(filename, "cma_perf_%02d.dat", thrid);
	fp = fopen(filename, "a");
	fprintf(fp, "%d %d %d %e %e\n", func_dim, func_id, step, final_dist, ffinal);
	fclose(fp);


	const double *xbever = cmaes_GetPtr(evo, "xbestever");

	sprintf(filename, "cma_xbever_%02d.dat", thrid);
	fp = fopen(filename, "a");
	for(int i=0; i<func_dim; i++)
		fprintf(fp,"%lf  ",xbever[i]);
	fprintf(fp,"\n");

	fclose(fp);
}



bool evaluate_and_update( cmaes_t* const evo, double* const*  pop, double *arFunvals, int* const info  ){
	
	int lambda = evo->sp.lambda;
	int func_dim = evo->sp.N;

	for (int i = 0; i < lambda; i++)
		fitfun(pop[i], func_dim, &arFunvals[i], info);

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




void Action::update(  cmaes_t* const evo, double **arFunvals ){
	
	evo->sp.ccov1   = data[0]; //rank 1 covariance update
	
	if (dim>1) evo->sp.ccovmu  = data[1]; //rank mu covariance update
	if (dim>2) evo->sp.ccumcov = data[2]; //path update c_c
	if (dim>3) evo->sp.cs      = data[3]; //step size control c_sigmai
	
	if (dim>4)	evo->sp.damps   = data[4]; //step size control c_sigmai
	else 		update_damps( evo );
	
	if (dim>5){ 
		lambda_frac	= data[5]; //pop size ratio
		lambda 	= floor( lambda_0 * lambda_frac );
		lambda 	= lambda < 4 ? 4 : lambda;
		lambda_frac = lambda/(double) lambda_0;
		*arFunvals 	= cmaes_ChangePopSize(evo, lambda);
	}

	//printf("selected action %f %f %f %f %f\n",
	//evo->sp.ccov1,evo->sp.ccovmu,evo->sp.ccumcov,evo->sp.cs,lambda_fac);
	//fflush(0);
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



void update_damps( cmaes_t* const evo )
{
	int lambda = evo->sp.lambda;
	int N = evo->sp.N;

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




 
void State::update_state( cmaes_t* const evo, double* oldFmedian, double* oldXmean )
{
	int func_dim = evo->sp.N;
	double* xMean = cmaes_GetNew(evo, "xmean");
	
	double xProgress = 0.;
	for (int i=0; i<func_dim; i++)
		xProgress += pow(xMean[i]-oldXmean[i],2);
	
	const double fmedian = cmaes_Get(evo, "fmedian");
	const double fProgress = (fmedian - *oldFmedian) /(fabs(fmedian) + fabs(*oldFmedian));


	//ratio between standard deviations
	double eps = 1e-16;
	const double d1 = evo->maxdiagC - evo->mindiagC;
	const double d2 = evo->maxEW - evo->minEW;
	//const double d1 = evo->meandiagC < 0 ? evo->meandiagC-eps : evo->meandiagC+eps;
	//const double d2 = evo->meanEW < 0 ? evo->meanEW-eps : evo->meanEW+eps;
	
	const double ratio1 = d1>eps ? (evo->mindiagC/d1) : 1e-12;
	const double ratio2 = d2>eps ? (evo->minEW/d2)    : 1e-12;
	const double ratio3 = d1>eps ? (evo->maxdiagC/d1) : 1e-12;
	const double ratio4 = d2>eps ? (evo->maxEW/d2)    : 1e-12;
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

	data[0] = ratio1;
	data[1] = ratio2;
	data[2] = ratio3;
	data[3] = ratio4;
	data[4] = sqrt(xProgress);
	data[5] = fProgress;
	data[6] = (double)func_dim;
	data[7] = evo->trace;

	//advance:
	*oldFmedian = fmedian;
	for (int i=0; i<func_dim; i++) oldXmean[i] = xMean[i];
	free(xMean);
}





