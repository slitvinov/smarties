/* --------------------------------------------------------- */
/* --------------- A Very Short Example -------------------- */
/* --------------------------------------------------------- */

#define _XOPEN_SOURCE 500
#define _BSD_SOURCE
#define __RLON 1
#define __RANDACT 0
#define __NGENSKIP 1
#include <stdio.h>
#include <stdlib.h> /* free() */
#include "cmaes_interface.h"
#include "Communicator.h"
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <random>
#include <chrono>
//#include <omp.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#define VERBOSE 0
/*#define _RESTART_*/
#define _IODUMP_ 0
#define JOBMAXTIME	0

#include "cmaes_learn.h"

#include "fitfun.h"


/* the optimization loop */
int main(int argn, char **args)
{
	if (argn<2) {
		printf("Did not receive the socket and nthreads. Aborting.\n");
		abort();
	}

	const int sock      = std::stoi(args[1]);
	const int nthreads  = 1;
	const int act_dim   = 6;
	const int state_dim = 8;
	std::seed_seq seq{sock};
	std::vector<int> seeds(nthreads);
	seq.generate(seeds.begin(), seeds.end());
	std::vector<std::mt19937*> generators(nthreads);
	for (int i=0; i<nthreads; i++) generators[i] = new std::mt19937(seeds[i]);

	std::uniform_int_distribution<int> 		func_dim_distribution(1,10);
	std::uniform_real_distribution<double> 	start_x_distribution(.3,.7);
	std::uniform_real_distribution<double> 	start_std_distribution(.2,.5);
	std::uniform_int_distribution<int> 		func_ID_distribution(0, _COUNT-1);
	std::uniform_int_distribution<int> 		cma_seed_distribution(0,std::numeric_limits<int>::max());
	

#if __RLON
	//communicator class, it needs a socket number sock, given by RL as first argument of execution
	Communicator comm(sock, state_dim, act_dim);
#endif

	//vectors of states and actions
	std::vector<double> state(state_dim);
	std::vector<double> actions(act_dim);

	std::vector<double> from_state1(act_dim); // initial state
	std::vector<double> from_state2(act_dim);

	const int thrid = 0; //omp_get_thread_num();

	write_cmaes_perf wcp;

	wcp.write(thrid);


	while (true) {
		int step = 0; // cmaes stepping
		int info[4]; //legacy: gen, chain, step, task
		cmaes_t * const evo = new cmaes_t(); /* a CMA-ES type struct or "object" */
		double oldFmedian, *oldXmean = nullptr; //related to RL rewards
		double *lower_bound, *upper_bound, *init_x, *init_std; //IC for cmaes
		double *arFunvals, *const*pop;  //cma current function values and samples

		//const int func_dim = 1 + ceil(std::fabs(func_dim_distribution(*generators[thrid])));
		const int func_dim = func_dim_distribution( *generators[thrid] );
		const int runseed  = cma_seed_distribution( *generators[thrid] );
				
		from_state1 = {1,1,1,1,0,0,(double)func_dim,0};
		from_state2 = {0,0,0,0,0,0,(double)func_dim,0};

		info[0] = func_ID_distribution(*generators[thrid]);
		
		const int 	lambda_0 	= 4+floor(3*std::log(func_dim));
		double		lambda_frac = 1.001;
		int 		lambda 		= floor(lambda_0*lambda_frac);
		
		evo->sp.funcID = info[0];
		printf("Selected function %d with dimensionality %d\n", info[0], func_dim);
		
		init_x 		= (double*)malloc(func_dim * sizeof(double));
		init_std 	= (double*)malloc(func_dim * sizeof(double));
		lower_bound = (double*)malloc(func_dim * sizeof(double));
		upper_bound = (double*)malloc(func_dim * sizeof(double));

		get_upper_lower_bounds(lower_bound, upper_bound, func_dim, info);
		
		for (int i = 0; i < func_dim; i++) { //to be returned from function?
			init_x[i] = start_x_distribution(*generators[thrid])
							*(upper_bound[i]-lower_bound[i]) + lower_bound[i];
			init_std[i] = start_std_distribution(*generators[thrid])
							*(upper_bound[i]-lower_bound[i]);
		}

		arFunvals = cmaes_init(evo, func_dim, init_x, init_std, runseed, lambda, "../cmaes_initials.par");
		printf("%s\n", cmaes_SayHello(evo));
		cmaes_ReadSignals(evo, "cmaes_signals.par");  /* write header and initial values */

#if __RLON
		{   
			copy_state(state, from_state1);
			comm.sendState(thrid, 1, state, 0);
			comm.recvAction(actions);
			actions_to_cma( actions.data(), act_dim, evo, &lambda, &lambda_frac, lambda_0, &arFunvals );
		}


#elif __RANDACT
		random_action(evo, *generators[thrid] )
#endif


		bool bConverged = false;
		while(true) { /* Iterate until stop criterion holds */
			
			// actions are constant in this loop
			for(int dG = 0; dG < __NGENSKIP*func_dim; dG++) {


				/* generate lambda new search points, check for nans */
				pop = cmaes_SamplePopulation(evo);

				bConverged = check_for_nan_inf( evo, pop );
				
				// re-sample if not feasible, check if stuck
				if( !bConverged)
					bConverged = resample( evo, pop, lower_bound, upper_bound );

				/* evaluate current pop and update distribution */
				if(!bConverged)
					bConverged = evaluate_and_update(evo, pop, arFunvals, info  );

				if(!bConverged){
					if (step == 0){ //need an initial state
						oldFmedian = cmaes_Get(evo, "fmedian");
						oldXmean = cmaes_GetNew(evo, "xmean");
					}

					if(cmaes_TestForTermination(evo)) {
						bConverged = true;
					}
				}
				
				step += lambda;

				fflush(stdout); /* useful in MinGW */
				
				if(bConverged) break;

#if VERBOSE
				print_best_ever( evo, step );
#endif

#if _IODUMP_
				dump_curgen( pop, arFunvals, step, lambda, func_dim );
#endif

			} // end of constant action loop


			if (bConverged) break; //go to send terminal state
#if __RLON
			{
				update_state(evo, state.data(), &oldFmedian, oldXmean );
				const double r = -.01*lambda_frac;
				comm.sendState(thrid, 0, state, r);
				comm.recvAction(actions);
				actions_to_cma( actions.data(), act_dim, evo, &lambda, &lambda_frac, lambda_0, &arFunvals );
			
			}
#elif __RANDACT
		random_action(evo, *generators[thrid] )
#endif
		
		} // end of single function optimization

		if (evo->isStuck == 1) {
			fprintf(stderr, "Stopping becoz stuck\n");
			copy_state(state, from_state2);
			
			for (int i = 0; i < lambda; ++i) {
				std::ostringstream o;
				o << "[";
				for (int j=0; j<func_dim; j++) {
					o << pop[i][j];
					if (i < func_dim-1) o << " ";
				}
				o << "]";
				printf("Evaluated function in %s = %e\n",
						o.str().c_str(), arFunvals[i]);
			}
#if __RLON
			comm.sendState(thrid, 2, state, -1.); // final state: info is 2
#endif
		} 
		else{
			update_state(evo, state.data(), &oldFmedian, oldXmean);
			double* xfinal = cmaes_GetNew(evo, "xmean");
			double ffinal;
			
			fitfun(xfinal, func_dim, &ffinal, info);
			
			const double final_dist = eval_distance_from_optimum(xfinal, func_dim, info);
			const double r_end 		= std::max(-1., 1-1e2*final_dist);

			//printf("Sending %f %f %f %f %f %f\n",
			//		state[0],state[1],state[2],state[3],state[4],r_end);
			//fflush(0);
#if __RLON
			comm.sendState(thrid, 2, state, r_end); // final state: info is 2
#endif

			wcp.write( thrid, func_dim, info[0], step, final_dist, ffinal );

			free(xfinal);
		}

		printf("Stop: %s\n",  cmaes_TestForTermination(evo)); /* print termination reason */
		
		//cmaes_WriteToFile(&evo, "all", "allcmaes.dat");         /* write final results */
		cmaes_exit(evo); /* release memory */
		delete evo;
		if(oldXmean not_eq nullptr)
			free(oldXmean);
		free(lower_bound);
		free(upper_bound);
		free(init_x);
		free(init_std);

	} // end of learning loop

	return 0;
}
