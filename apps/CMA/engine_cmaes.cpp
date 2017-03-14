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

/* the objective (fitness) function to be minimized */
void fitfun(double * const x, int N, double* const output, int * const info);
/* the upper and lower bounds are defined by the function */
void get_upper_lower_bounds(double*const lower_bound, double*const upper_bound,
                            int N, int * const info);
/* final evaluation of the found optimum */
double eval_distance_from_optimum(const double* const found_optimum,
                                  int N, int* const info);
#include "fitfun.c"


int is_feasible(double* const pop, double* const lower_bound,
								double* const upper_bound, int dim);
void update_state(cmaes_t* const evo, double* const state, double* oldFmedian,
																		  double* oldXmean, const int func_dim);

void update_damps(cmaes_t* const evo,const int N, const int lambda)
{
  evo->sp.damps =
    (1 + 2*std::max(0., sqrt((evo->sp.mueff-1.)/(N+1.)) - 1 ) )     /* basic factor */
    * std::max(0.3, 1. -                                       /* modify for short runs */
              (double)N / (1e-6+std::min(evo->sp.stopMaxIter, evo->sp.stopMaxFunEvals/lambda)))
    + evo->sp.cs;
}

int is_feasible(double* const pop, double* const lower_bound,
								double* const upper_bound, int dim)
{
	int good;
	for (int i = 0; i < dim; i++) {
		if (std::isnan(pop[i]) || std::isinf(pop[i]))
				{ printf("Sampled nan: FU cmaes \n"); abort(); }
		good = (lower_bound[i] <= pop[i]) && (pop[i] <= upper_bound[i]);
		if (!good) return 0;
	}
	return 1;
}

void update_state(cmaes_t* const evo, double* const state, double* oldFmedian,
																			double* oldXmean, const int func_dim)
{
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

    std::normal_distribution<double> func_dim_distribution(0, 3);
    std::uniform_real_distribution<double> start_x_distribution(.3,.7);
    std::uniform_real_distribution<double> start_std_distribution(.2,.5);
    std::uniform_int_distribution<int> func_ID_distribution(0, _COUNT-1);
    std::uniform_int_distribution<int> cma_seed_distribution(0,
																							std::numeric_limits<int>::max());
    std::uniform_real_distribution<double> act0dist(.02,0.1);
    std::uniform_real_distribution<double> act1dist(.02,0.1);
    std::uniform_real_distribution<double> act2dist(.2,.8);
    std::uniform_real_distribution<double> act3dist(.2,.8);


#if __RLON
    //communicator class, it needs a socket number sock, given by RL as first argument of execution
    Communicator comm(sock, state_dim, act_dim);
#endif

    //vector of state variables
    std::vector<double> state(state_dim);
    //vector of actions received by RL
    std::vector<double> actions(act_dim);

    const int thrid = 0; //omp_get_thread_num();

    {
        char filename[256];
        sprintf(filename, "cma_perf_%02d.dat", thrid);
        FILE *fp = fopen(filename, "a");
        fprintf(fp, "dim func nstep dist feval\n");
        fclose(fp);
    }

    while (true) {
        int step = 0; // cmaes stepping
        int info[4]; //legacy: gen, chain, step, task
        cmaes_t * const evo = new cmaes_t(); /* an CMA-ES type struct or "object" */
        double oldFmedian, *oldXmean = nullptr; //related to RL rewards
        double *lower_bound, *upper_bound, *init_x, *init_std; //IC for cmaes
        double *arFunvals, *const*pop;  //cma current function values and samples

        const int func_dim = 1 +
										ceil(std::fabs(func_dim_distribution(*generators[thrid])));
        const int runseed = cma_seed_distribution(*generators[thrid]);
        info[0] = func_ID_distribution(*generators[thrid]);
        const int lambda_0 = 4+floor(3*std::log(func_dim));
        double lambda_fac = 1.001;
        int lambda = floor(lambda_0*lambda_fac);
        evo->sp.funcID = info[0];
        printf("Selected function %d with dimensionality %d\n",
								info[0], func_dim);
				init_x = (double*)malloc(func_dim * sizeof(double));
				init_std = (double*)malloc(func_dim * sizeof(double));
        lower_bound = (double*)malloc(func_dim * sizeof(double));
				upper_bound = (double*)malloc(func_dim * sizeof(double));

				get_upper_lower_bounds(lower_bound, upper_bound, func_dim, info);
				for (int i = 0; i < func_dim; i++) { //to be returned from function?
					init_x[i] = start_x_distribution(*generators[thrid])
															*(upper_bound[i]-lower_bound[i]) + lower_bound[i];
					init_std[i] = start_std_distribution(*generators[thrid])
															*(upper_bound[i]-lower_bound[i]);
				}

        arFunvals = cmaes_init(evo, func_dim, init_x, init_std,
																runseed, lambda, "../cmaes_initials.par");
        printf("%s\n", cmaes_SayHello(evo));
        cmaes_ReadSignals(evo, "cmaes_signals.par");  /* write header and initial values */

#if __RLON
        {   // initial state
						state[0] = 1;
						state[1] = 1;
						state[2] = 1;
						state[3] = 1;
						state[4] = 0;
						state[5] = 0;
						state[6] = (double)func_dim;
						state[7] = 0;
						comm.sendState(thrid, 1, state, 0);

						comm.recvAction(actions);
									   evo->sp.ccov1   = actions[0]; //rank 1 covariance update
						if (act_dim>1) evo->sp.ccovmu  = actions[1]; //rank mu covariance update
						if (act_dim>2) evo->sp.ccumcov = actions[2]; //path update c_c
						if (act_dim>3) evo->sp.cs      = actions[3]; //step size control c_sigmai
						if (act_dim>4) evo->sp.damps   = actions[4]; //step size control c_sigmai
            else update_damps(evo,func_dim, lambda);
						if (act_dim>5) { lambda_fac    = actions[5]; //pop size
                             lambda = floor(lambda_0*lambda_fac);
                             lambda = lambda < 4 ? 4 : lambda;
                             lambda_fac = lambda/(double)lambda_0;
                             arFunvals = cmaes_ChangePopSize(evo, lambda);
            }
            //printf("selected action %f %f %f %f %f\n",
            //evo->sp.ccov1,evo->sp.ccovmu,evo->sp.ccumcov,evo->sp.cs,lambda_fac);
            //fflush(0);
        }
#elif __RANDACT
        {
            evo->sp.ccov1   = act0dist(*generators[thrid]);
            evo->sp.ccovmu  = act1dist(*generators[thrid]);
            evo->sp.ccumcov = act2dist(*generators[thrid]);
            evo->sp.cs      = act3dist(*generators[thrid]);
            update_damps(evo,func_dim, lambda);
        }
#endif

        bool bConverged = false;
        while(true) {
            /* Iterate until stop criterion holds */
            for(int dG = 0; dG < __NGENSKIP*func_dim; dG++) {

							/* generate lambda new search points, sample population */
							pop = cmaes_SamplePopulation(evo);

							bool foundnan = false;
							for (int i = 0; i < lambda; ++i)
							for (int j = 0; j < func_dim; j++)
								if (std::isnan(pop[i][j]) || std::isinf(pop[i][j]))
									foundnan = true;

							if(foundnan)	{
								fprintf(stderr, "It was nan all along!!!\n");
								evo->isStuck = 1;
							}
							/*
							Here we resample each solution point pop[i] until it is feasible.
							Assumptions:
										the feasible domain is convex,
										the optimum is not on (very close to) the domain boundary,
										initialX is feasible and
										initialSTD are small enough to prevent infinite looping.
							*/

            	if(evo->isStuck == 1) bConverged = true;

							int safety = 0;
							if(!bConverged)
							for (int i = 0; i < lambda; ++i)
							{
								while (!is_feasible(pop[i],lower_bound,upper_bound,func_dim)
												&& safety++ < 1e3)
								cmaes_ReSampleSingle(evo, i);
							}

							if(!bConverged)
            	if(evo->isStuck == 1) bConverged = true;

							/* evaluate current pop */
							if(!bConverged)
							for (int i = 0; i < lambda; i++)
                fitfun(pop[i], func_dim, &arFunvals[i], info);

							if(!bConverged)
							cmaes_UpdateDistribution(evo, arFunvals);

							if(!bConverged)
							if(evo->isStuck == 1) bConverged = true;

							if(!bConverged)
            	if (step == 0) { //need an initial state
								oldFmedian = cmaes_Get(evo, "fmedian");
								oldXmean = cmaes_GetNew(evo, "xmean");
            	}

							step += lambda;
							if(!bConverged)
            	if(cmaes_TestForTermination(evo)) {
            		bConverged = true;
            	}
              fflush(stdout); /* useful in MinGW */
							if(bConverged) break;
              /* read instructions for printing output or changing termination conditions */
              //cmaes_ReadSignals(evo, "../cmaes_signals.par");
#if VERBOSE
              {
								const double *xbever = cmaes_GetPtr(evo, "xbestever");
								double fbever = cmaes_Get(evo, "fbestever");
								printf("BEST @ %5d: ", step);
								for (int i = 0; i < func_dim; i++)
									printf("%25.16lf ", xbever[i]);
								printf("%25.16lf\n", fbever);
              }
#endif
#if _IODUMP_
              {
								char filename[256];
								sprintf(filename, "curgen_db_%03d.txt", step);
								FILE *fp = fopen(filename, "w");
								for (int i = 0; i < lambda; i++) {
									for (int j = 0; j < func_dim; j++)
										fprintf(fp, "%.6le ", pop[i][j]);
									fprintf(fp, "%.6le\n", arFunvals[i]);
								}
								fclose(fp);
              }
#endif
            }
        		if (bConverged) break; //go to send terminal state
#if __RLON
					{
	        	update_state(evo, state.data(), &oldFmedian, oldXmean, func_dim);
						const double r = -.01*lambda_fac;
						comm.sendState(thrid, 0, state, r);
						comm.recvAction(actions);
									   			 evo->sp.ccov1   = actions[0]; //rank 1 covariance update
						if (act_dim>1) evo->sp.ccovmu  = actions[1]; //rank mu covariance update
						if (act_dim>2) evo->sp.ccumcov = actions[2]; //path update c_c
						if (act_dim>3) evo->sp.cs      = actions[3]; //step size control c_sigma
            if (act_dim>4) evo->sp.damps   = actions[4]; //step size control c_sigmai
            else update_damps(evo,func_dim, lambda);
						if (act_dim>5) { lambda_fac    = actions[5]; //pop size
                             lambda = floor(lambda_0*lambda_fac);
                             lambda = lambda < 4 ? 4 : lambda;
                             lambda_fac = lambda/(double)lambda_0;
                             arFunvals = cmaes_ChangePopSize(evo, lambda);
            }
            //printf("selected action %f %f %f %f %f\n",
            //evo->sp.ccov1,evo->sp.ccovmu,evo->sp.ccumcov,evo->sp.cs,lambda_fac);
            //fflush(0);
					}
#elif __RANDACT
          {
              evo->sp.ccov1   = act0dist(*generators[thrid]);
              evo->sp.ccovmu  = act1dist(*generators[thrid]);
              evo->sp.ccumcov = act2dist(*generators[thrid]);
              evo->sp.cs      = act3dist(*generators[thrid]);
              update_damps(evo,func_dim, lambda);

          }
#endif
        }

				if (evo->isStuck == 1) {
					fprintf(stderr, "Stopping becoz stuck\n");
					state[0] = 0;
					state[1] = 0;
					state[2] = 0;
					state[3] = 0;
					state[4] = 0;
					state[5] = 0;
					state[6] = (double)func_dim;
					state[7] = 0;

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
				} else {
					update_state(evo, state.data(), &oldFmedian, oldXmean, func_dim);
	        double* xfinal = cmaes_GetNew(evo, "xmean");
	        double ffinal;
          fitfun(xfinal, func_dim, &ffinal, info);
	        const double final_dist =
														 eval_distance_from_optimum(xfinal, func_dim, info);
	        const double r_end = std::max(-1., 1-1e2*final_dist);

					//printf("Sending %f %f %f %f %f %f\n",
					//		state[0],state[1],state[2],state[3],state[4],r_end);
					//fflush(0);
					#if __RLON
        	comm.sendState(thrid, 2, state, r_end); // final state: info is 2
					#endif

	        {
	            //print to file dim, function ID, nsteps, distance from opt, function value
						char filename[256];
						sprintf(filename, "cma_perf_%02d.dat", thrid);
						FILE *fp = fopen(filename, "a");
						fprintf(fp, "%d %d %d %e %e\n",
												func_dim, info[0], step, final_dist, ffinal);
						fclose(fp);
	        }
					free(xfinal);
				}

        //printf("Stop: %s\n",  cmaes_TestForTermination(evo)); /* print termination reason */
        //cmaes_WriteToFile(&evo, "all", "allcmaes.dat");         /* write final results */
        cmaes_exit(evo); /* release memory */
				delete evo;
        if(oldXmean not_eq nullptr)
            free(oldXmean);
				free(lower_bound);
				free(upper_bound);
				free(init_x);
				free(init_std);
    }
	return 0;
}
