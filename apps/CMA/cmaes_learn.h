#include <stdio.h>
#include <stdlib.h> /* free() */
#include "cmaes.h"
#include <random>

#include <string.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <sstream>

#ifndef CMAES_LEARN_H
#define CMAES_LEARN_H




class State{
	
	const int state_dim = 8;
	std::vector<double> data;

	int func_dim;

	State( int N ){
		data.resize(state_dim);
		func_dim = N;
	}

	void initial_state( ){
		data = {1,1,1,1,0,0,(double)func_dim,0};
	}
	
	void final_state(){
		data = {0,0,0,0,0,0,(double)func_dim,0};
	}

};




class write_cmaes_perf{
	
	public:
		void write( const int thrid );
		//void write( cmaes_t* const evo, const int thrid, int func_id, int step, const double final_dist, double ffinal );
		void write( cmaes_t* const evo,const int thrid, const int func_dim, int func_id, int step, const double final_dist, double ffinal );
};


void dump_curgen( double* const* pop, double *arFunvals, int step, int lambda, int func_dim );

void print_best_ever( cmaes_t* const evo, int func_dim );

void update_damps( cmaes_t* const evo );

int is_feasible(double* const pop, double* const lower_bound, double* const upper_bound, int dim);

void update_state(cmaes_t* const evo, double* const state, double* oldFmedian, double* oldXmean);

void copy_state( std::vector<double>& state, std::vector<double> from_state );

void actions_to_cma( double* const actions, int act_dim,  cmaes_t* const evo, 
						int *lambda, double *lambda_fac, const int lambda_0, double **arFunvals );

bool check_for_nan_inf(cmaes_t* const evo, double* const* pop );

bool resample( cmaes_t* const evo, double* const* pop, double* const lower_bound, double* const upper_bound );


bool evaluate_and_update( cmaes_t* const evo, double* const*  pop, double *arFunvals, int* const info  );


void random_action( cmaes_t* const evo, std::mt19937 gen );

#endif
