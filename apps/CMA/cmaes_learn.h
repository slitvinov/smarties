#include <stdio.h>
#include "cmaes.h"

#ifndef CMAES_LEARN_H
#define CMAES_LEARN_H



class write_cmaes_perf{
	
	public:
		void write( const int thrid );
		void write( const int thrid, const int func_dim, int func_id, int step, const double final_dist, double ffinal );
};


void dump_curgen( double* const* pop, double *arFunvals, int step, int lambda, int func_dim );

void print_best_ever( cmaes_t* const evo, int func_dim );

void update_damps(cmaes_t* const evo,const int N, const int lambda);

int is_feasible(double* const pop, double* const lower_bound,
							double* const upper_bound, int dim);

void update_state(cmaes_t* const evo, double* const state, double* oldFmedian, double* oldXmean);

void copy_state( std::vector<double>& state, std::vector<double> from_state );

void actions_to_cma( double* const actions, int act_dim,  cmaes_t* const evo, 
						int *lambda, double *lambda_fac, const int lambda_0, double **arFunvals );

bool check_for_nan_inf(cmaes_t* const evo, double* const* pop );

bool resample( cmaes_t* const evo, double* const* pop, double* const lower_bound, double* const upper_bound );


bool evaluate_and_update( cmaes_t* const evo, double* const*  pop, double *arFunvals, int* const info  );


#endif
