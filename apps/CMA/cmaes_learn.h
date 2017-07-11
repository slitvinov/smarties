#include "cmaes.h"
#include <random>

#include <string.h>
#include <chrono>
#include <algorithm>
#include <iostream>

#ifndef CMAES_LEARN_H
#define CMAES_LEARN_H



class Action{

	public:
		const int dim   = 6;
		std::vector<double> data;

		int 	lambda_0;
		double	lambda_frac;
		int 	lambda;

		Action(){
			data.resize(dim);
		}

		void initialize( int func_dim ){
			lambda_0 	= 4+floor(3*std::log(func_dim));
			lambda_frac = 1.001;
			lambda 		= floor(lambda_0*lambda_frac);
		}

		void update(  cmaes_t* const evo, double **arFunvals );

};




class State{

	public:

		const int dim = 8;
		std::vector<double> data;

		State( ){
			data.resize(dim);
		}

		void initial_state( int func_dim ){
			data = {1,1,1,1,0,0,(double)func_dim,0};
		}

		void final_state( int func_dim ){
			data = {0,0,0,0,0,0,(double)func_dim,0};
		}

		void update_state( cmaes_t* const evo, double* oldFmedian, double* oldXmean );

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
