#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "gsl_headers.h"


typedef enum {
    ACKLEY,  /* [-32.768, 32.768]; f(x*) = 0, x*[i] = 0 */
    DIXON_PRICE,  /* [-10, 10]; f(x*) = 0, x*[i] = pow(2, -1.+1./pow(2, i)) */
    GRIEWANK,  /* [-600, 600]; f(x*) = 0, x*[i] = 0 */
    LEVY,  /* [-10, 10]; f(x*) = 0, x*[i] = 1 */
    PERM,  /* [-N, N]; f(x*) = 0, x*[i] = i+1. */
    PERM0,  /* [-N, N]; f(x*) = 0, x*[i] = 1./(i+1.) */
    RASTRIGIN,  /* [-5.12, 5.12]; f(x*) = 0, x*[i] = 0 */
    ROSENBROCK,  /* [-5, 10]; f(x*) = 0, x*[i] = 1 */
    ROTATED_HYPER_ELLIPSOID,  /* [-65.536, 65.536]; f(x*) = 0, x*[i] = 0 */
    SCHWEFEL,  /* [-500, 500]; f(x*) = 0, x*[i] = 420.9687 */
    SPHERE,  /* [-5.12, 5.12]; f(x*) = 0, x*[i] = 0 */
    STYBLINSKI_TANG,  /* [-5, 5]; f(x*) = 0, x*[i] = -2.903534 */
    SUM_OF_POWER,  /* [-1, 1]; f(x*) = 0, x*[i] = 0 */
    SUM_OF_SQUARES, /* [-10, 10]; f(x*) = 0, x*[i] = 0 */
    ZAKHAROV,  /* [-5, 10]; f(x*) = 0, x*[i] = 0 */
    _COUNT
} function;


/* info[0] chooses a random function */
double fitfun(double * const x, int N, void * const output, int * const info)  {
    double f;
    int i;

    int rnd = info[0] % _COUNT;  /* this defines which function to use */
    //std::cout << "Function evaluation at ";
    //for (i = 0; i < N; ++i) std::cout << x[i] << " ";
    //std::cout << std::endl;

    switch (rnd) {
        case ACKLEY: {
            double a = 20, b = .2, c = 2.*M_PI, s1 = 0., s2 = 0.;
            for (i = 0; i < N; ++i) {
                s1 += x[i]*x[i];
                s2 += cos(c*x[i]);
            }
            f = -a*exp(-b*sqrt(s1/N)) - exp(s2/N) + a + exp(1.);
            break;
        }

        case DIXON_PRICE: {
            double s = 0.;
            for (i = 1; i < N; ++i)
                s += (i+1.)*pow(2*x[i]*x[i]-x[i-1], 2);
            f = pow(x[0]-1., 2) + s;
            break;
        }

        case GRIEWANK: {
            double s = 0., p = 1.;
            for (i = 0; i < N; ++i) {
                s += x[i]*x[i];
                p *= cos(x[i]/sqrt(1.+i));
            }
            f = s/4000. - p + 1.;
            break;
        }

        case LEVY: {
            double s = 0.;
            for (i = 0; i < N-1; ++i) {
                s += .0625*pow(x[i]-1, 2)
                    * (1.+10.*pow(sin(M_PI*.25*(3+x[i]) +1), 2));
            }
            f = pow(sin(M_PI*.25*(3.+x[0])), 2) + s
                + .0625*pow(x[N-1]-1, 2)*(1+pow(sin(M_PI*.5*(3+x[N-1])), 2));
            break;
        }

        case PERM: {
            double beta = .5;
            double s2 = 0.;
            int j;
            for (i = 0; i < N; ++i) {
                double s1 = 0.;
                for (j = 0; j < N; ++j)
                    s1 += (pow(j+1, i+1)+beta)*(pow(x[j]/(j+1.), i+1) - 1.);
                s2 += s1*s1;
            }
            f = s2;
            break;
        }

        case PERM0: {
            double beta = 10.;
            double s2 = 0.;
            int j;
            for (i = 0; i < N; ++i) {
                double s1 = 0.;
                for (j = 0; j < N; ++j)
                    s1 += (j+1.+beta)*(pow(x[j], i+1) - 1./pow(j+1, i+1));
                s2 += s1*s1;
            }
            f = s2;
            break;
        }

        case RASTRIGIN: {
            double s = 0.;
            for (i = 0; i < N; ++i)
                s += x[i]*x[i] - 10.*cos(2.*M_PI*x[i]);
            f = 10.*N+s;
            break;
        }

        case ROSENBROCK: {
            double s = 0.;
            for (i = 0; i < N-1; ++i)
                s += 100.*pow(x[i+1]-x[i]*x[i], 2) + pow(x[i]-1., 2);
            f = s;
            break;
        }

        case ROTATED_HYPER_ELLIPSOID: {
            int j;
            double s = 0.;
            for (i = 0; i < N; ++i)
                for (j = 0; j < i; ++j)
                    s += x[j]*x[j];
            f = s;
            break;
        }

       case SCHWEFEL: {
            double s = 0.;
            for (i = 0; i < N; ++i)
                s += x[i]*sin(sqrt(fabs(x[i])));
            f = 418.9829*N-s;
            break;
        }

        case SPHERE: {
            double s = 0.;
            for (i = 0; i < N; ++i)
                s += x[i]*x[i];
            f = s;
            break;
        }

        case STYBLINSKI_TANG: {
            double s = 0.;
            for (i = 0; i < N; ++i)
                s += pow(x[i], 4) - 16.*x[i]*x[i] + 5.*x[i];
            f = 39.16599*N + .5*s;
            break;
        }

        case SUM_OF_POWER: {
            double s = 0.;
            for (i = 0; i < N; ++i)
                s += pow(fabs(x[i]), i+2);
            f = s;
            break;
        }

        case SUM_OF_SQUARES: {
            double s = 0.;
            for (i = 0; i < N; ++i)
                s += (i+1.)*x[i]*x[i];
            f = s;
            break;
        }

        case ZAKHAROV: {
            double s1 = 0., s2 = 0.;
            for (i = 0; i < N; ++i) {
                s1 += x[i]*x[i];
                s2 += .5*(i+1.)*x[i];
            }
            f = s1 + pow(s2, 2) + pow(s2, 4);
            break;
        }

        default:
            printf("Function %d not found. Exiting.\n", rnd);
            exit(1);
    }
    //std::cout << "feval = " << f << std::endl;
    return f;  /* our CMA maximizes (there's another "-" in the code) */
}

void get_upper_lower_bounds(double* const lower_bound,
                            double* const upper_bound,
                            int N, int * const info)
{
    int i;

    int rnd = info[0] % _COUNT;  /* this defines which function to use */

    switch (rnd) {
        case ACKLEY: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -32.768;
            	upper_bound[i] =  32.768;
            }
            break;
        }

        case DIXON_PRICE: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -10;
            	upper_bound[i] =  10;
            }
            break;
        }

        case GRIEWANK: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -600;
            	upper_bound[i] =  600;
            }
            break;
        }

        case LEVY: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -10;
            	upper_bound[i] =  10;
            }
            break;
        }

        case PERM: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -N;
            	upper_bound[i] =  N;
            }
            break;
        }

        case PERM0: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -N;
            	upper_bound[i] =  N;
            }
            break;
        }

        case RASTRIGIN: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -5.12;
            	upper_bound[i] =  5.12;
            }
            break;
        }

        case ROSENBROCK: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -5;
            	upper_bound[i] =  10;
            }
            break;
        }

        case ROTATED_HYPER_ELLIPSOID: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -65.536;
            	upper_bound[i] =  65.536;
            }
            break;
        }

       case SCHWEFEL: {
       	for (i = 0; i < N; ++i) {
           	lower_bound[i] = -500;
           	upper_bound[i] =  500;
           }
           break;
       }

        case SPHERE: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -5.12;
            	upper_bound[i] =  5.12;
            }
            break;
        }

        case STYBLINSKI_TANG: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -5;
            	upper_bound[i] =  5;
            }
            break;
        }

        case SUM_OF_POWER: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -1;
            	upper_bound[i] =  1;
            }
            break;
        }

        case SUM_OF_SQUARES: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -10;
            	upper_bound[i] =  10;
            }
            break;
        }

        case ZAKHAROV: {
        	for (i = 0; i < N; ++i) {
            	lower_bound[i] = -5;
            	upper_bound[i] =  10;
            }
            break;
        }

        default:
            printf("Function %d not found. Exiting.\n", rnd);
            exit(1);
    }
}

double eval_distance_from_optimum(const double* const found_optimum,
                                  int N, int* const info) 
{
    int i;
    double dist = 0.;
    int rnd = info[0] % _COUNT;  /* this defines which function to use */

    switch (rnd) {
        case ACKLEY: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i], 2);
            }
            break;
        }

        case DIXON_PRICE: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i] - pow(2, -1.+1./pow(2,i)), 2);
            }
            break;
        }

        case GRIEWANK: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i], 2);
            }
            break;
        }

        case LEVY: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i] -1., 2);
            }
            break;
        }

        case PERM: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i] -i-1, 2);
            }
            break;
        }

        case PERM0: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i] -1./(i+1), 2);
            }
            break;
        }

        case RASTRIGIN: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i], 2);
            }
            break;
        }

        case ROSENBROCK: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i] -1., 2);
            }
            break;
        }

        case ROTATED_HYPER_ELLIPSOID: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i], 2);
            }
            break;
        }

       case SCHWEFEL: {
       	for (i = 0; i < N; ++i) {
    			dist += pow(found_optimum[i] -420.9687, 2);
           }
           break;
       }

        case SPHERE: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i], 2);
            }
            break;
        }

        case STYBLINSKI_TANG: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i] +2.903534, 2);
            }
            break;
        }

        case SUM_OF_POWER: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i], 2);
            }
            break;
        }

        case SUM_OF_SQUARES: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i], 2);
            }
            break;
        }

        case ZAKHAROV: {
        	for (i = 0; i < N; ++i) {
        		dist += pow(found_optimum[i], 2);
            }
            break;
        }

        default: {
            printf("Function %d not found. Exiting.\n", rnd);
            exit(1);
        }
    }

	//std::cout << sqrt(dist)  << std::endl;
	return sqrt(dist);
}
