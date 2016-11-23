/* --------------------------------------------------------- */
/* --------------- A Very Short Example -------------------- */
/* --------------------------------------------------------- */

#define _XOPEN_SOURCE 500
#define _BSD_SOURCE
#define __NGENSKIP 10
#include <stdio.h>
#include <stdlib.h> /* free() */
#include "cmaes_interface.h"
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <random>
#include <algorithm>
#define VERBOSE 1
/*#define _RESTART_*/
#define _IODUMP_ 1
#define JOBMAXTIME	0
#include "fitfun.c" 

/* the objective (fitness) function to be minimized */
void taskfun(double *x, int *pn, double *res, int *info)
{
	int n = *pn;
#if 0
	int gen, chain, step, task;
	gen = info[0]; chain = info[1]; step = info[2]; task = info[3];
	printf("executing task (%d,%d,%d,%d)\n", gen, chain, step, task);
#endif

	double f = -fitfun(x, n, (void *)NULL, info);	/* CMA-ES needs this minus sign */

	*res = f;
	return;
}

double *lower_bound;	//double lower_bound[] = {-6.0, -6.0};
double *upper_bound;	//double upper_bound[] = {+6.0, +6.0};

int is_feasible(double *pop, int dim)
{
	int i, good;
	for (i = 0; i < dim; i++) {
		good = (lower_bound[i] <= pop[i]) && (pop[i] <= upper_bound[i]);
		if (!good) {
			return 0;
		}
	}
	return 1;
}

void draw_feasible_samples(cmaes_t * evo, double* const* pop)
{
    /* generate lambda new search points, sample population */
    pop = cmaes_SamplePopulation(&evo); /* do not change content of pop */
    /*
       Here we may resample each solution point pop[i] until it
       becomes feasible. function is_feasible(...) needs to be
       user-defined.
       Assumptions: the feasible domain is convex, the optimum is
       not on (or very close to) the domain boundary, initialX is
       feasible and initialStandardDeviations are sufficiently small
       to prevent quasi-infinite looping.
    */
	for (i = 0; i < cmaes_Get(evo, "popsize"); ++i)
		while (!is_feasible(pop[i], dim))
			cmaes_ReSampleSingle(evo, i);
}

void update_state(cmaes_t * evo, double * const state, double* oldFmedian, double* oldXmean, const int func_dim)
{
	double* xMean = cmaes_GetNew(&evo, "xmean");
	double xProgress = 0.;
	for (int i=0; i<func_dim; i++)
		xProgress += pow(xMean[i]-oldXmean[i],2);

	state[0] = sqrt(evo->mindiagC/evo->maxdiagC);
	state[1] = sqrt(evo->minEW/evo->maxEW);
	state[2] = sqrt(xProgress);
	state[3] = cmaes_Get(evo, "fmedian") - *oldFmedian;
	state[4] = (double)func_dim;

	//advance:
	*oldFmedian = cmaes_Get(evo, "fmedian");
	swap(oldXmean, xMean);
	free(xMean);
}

/* the optimization loop */
int main(int argn, char **args)
{
    if (argn<2) {
        std::cout << "I did not receive the socket ID. Aborting." << std::endl;
        abort();
    }
    const int sock = std::stoi(args[1]);
    const int act_dim   = 1
    const int state_dim = 5
    mt19937 gen(sock);
    std::uniform_real_distribution<double> start_x_distribution(0.35,0.65);
    std::uniform_real_distribution<double> start_std_distribution(0.2,0.5);
    std::exponential_distribution<double> func_dim_distribution(0.5);
    std::uniform_int_distribution<int> func_ID_distribution(0,_COUNT);
    std::uniform_int_distribution<long> cma_seed_distribution(std::numeric_limits<long>::min(),
    													      std::numeric_limits<long>::max());

    //communicator class, it needs a socket number sock, given by RL as first argument of execution
    Communicator comm(sock, state_dim, act_dim);
    //vector of state variables
    vector<double> state(state_dim);
    //vector of actions received by RL
    vector<double> actions(act_dim);
        
    while (true) {
        cmaes_t evo; /* an CMA-ES type struct or "object" */
        double oldFmedian, *oldXmean; //related to RL rewards
        double *lower_bound, *upper_bound, *init_x, *init_std; //IC for cmaes
        double *arFunvals, *const*pop;  //cma current function values and samples
        int RL_info = 1; //initial state
        int step = 0; // cmaes stepping
        int info[4]; //legacy: gen, chain, step, task
        const int func_dim = 1 + ceil(func_dim_distribution(gen));
        const int runseed = cma_seed_distribution(gen);
        info[0] = func_ID_distribution(gen);

		init_x = malloc(func_dim * sizeof(double));
		init_std = malloc(func_dim * sizeof(double));
        lower_bound = malloc(func_dim * sizeof(double));
		upper_bound = malloc(func_dim * sizeof(double));

		get_upper_lower_bounds(lower_bound, upper_bound, func_dim, info);
		for (int i = 0; i < func_dim; i++) { //to be returned from function?
			init_x[i] = start_x_distribution(gen)*(upper_bound[i]-lower_bound[i]) + lower_bound[i];
			init_std[i] = start_std_distribution(gen)*(upper_bound[i]-lower_bound[i]) + lower_bound[i];
		}

        arFunvals = cmaes_init(&evo, func_dim, init_x, NULL, runseed, 0, "cmaes_initials.par");
        printf("%s\n", cmaes_SayHello(&evo));
        cmaes_ReadSignals(&evo, "cmaes_signals.par");  /* write header and initial values */

        const int lambda = cmaes_Get(&evo, "lambda");

        bool bConverged = false;
        while(true) {
            /* Iterate until stop criterion holds */
        	for(int dG = 0; dG < __NGENSKIP; dG++) {

        		draw_feasible_samples(&evo, pop);
        		/* evaluate current pop */
        		for (int i = 0; i < lambda; i++)
        			fitfun(pop[i], func_dim, &arFunvals[i]);

    			/* update the search distribution used for cmaes_SampleDistribution() */
            	cmaes_UpdateDistribution(&evo, arFunvals);
            	if (step == 0) { //need an initial state
					oldFmedian = cmaes_Get(&evo, "fmedian");
					oldXmean = cmaes_GetNew(&evo, "xmean");
            	}

                /* read instructions for printing output or changing termination conditions */
                cmaes_ReadSignals(&evo, "cmaes_signals.par");
                fflush(stdout); /* useful in MinGW */
#if VERBOSE
                {
					const double *xbever = cmaes_GetPtr(&evo, "xbestever");
					double fbever = cmaes_Get(&evo, "fbestever");
					printf("BEST @ %5d: ", step);
					for (i = 0; i < dim; i++)
						printf("%25.16lf ", xbever[i]);
					printf("%25.16lf\n", fbever);
                }
#endif
#if _IODUMP_
                {
					char filename[256];
					sprintf(filename, "curgen_db_%03d.txt", step);
					FILE *fp = fopen(filename, "w");
					for (i = 0; i < lambda; i++) {
						int j;
						for (j = 0; j < dim; j++) fprintf(fp, "%.6le ", pop[i][j]);
						fprintf(fp, "%.6le\n", arFunvals[i]);
					}
					fclose(fp);
                }
#endif
                step++;

            	if(cmaes_TestForTermination(&evo)) {
            		bConverged = true;
            		break;
            	}
            }
        	if (bConverged) break; //go to send terminal state

        	update_state(&evo, state.data(), &oldFmedian, oldXmean, func_dim);
            const double r = -.1;
			comm->sendState(0, RL_info, state, r);
			RL_info = 0; //already sent initial state

			comm->recvAction(actions);
						   evo.sp.ccov1   = actions[0]; //rank 1 covariance update
			if (act_dim>1) evo.sp.ccovmu  = actions[1]; //rank mu covariance update
			if (act_dim>2) evo.sp.ccumcov = actions[2]; //path update c_c
			if (act_dim>3) evo.sp.cs      = actions[3]; //step size control c_sigma
        }

    	update_state(&evo, state.data(), &oldFmedian, oldXmean, func_dim);
        /* get best estimator for the optimum, xmean */
        double* xfinal = cmaes_GetNew(&evo, "xmean"); /* "xbestever" might be used as well */
        const double r_end = 1. - 1e6*eval_distance_from_optimum(xfinal, func_dim, info)/(upper_bound[0]-lower_bound[0]);
        comm->sendState(0, 2, state, r_end); // final state: info is 2
        
        printf("Stop:\n%s\n",  cmaes_TestForTermination(&evo)); /* print termination reason */
        cmaes_WriteToFile(&evo, "all", "allcmaes.dat");         /* write final results */
        cmaes_exit(&evo); /* release memory */ 
		free(xfinal);
		free(oldXmean);
		free(lower_bound);
		free(upper_bound);
		free(pop);
		free(init_x);
		free(init_std);
		free(arFunvals);
    }
	return 0;
}

