/*
 *  anntest.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <fstream>

#include "Network.h"
#include "WaveletNet.h"
#include "Approximator.h"
#include "rng.h"

//using namespace ANN;

int main (int argc, char** argv)
{
    
    int A2 = atoi(argv[1]);
    int A1 = atoi(argv[2]);
    int T = atof(argv[3]);
    const double L = atof(argv[4]);
    const double D = atof(argv[5]);
    const double bound = atof(argv[6]);
    
	vector<int> lsize;
    Approximator * ann;
    string suff, morestuff;
    double A, B;
    
    if (true)
    {
        lsize.push_back(7);
        lsize.push_back(28);
        lsize.push_back(56);
        lsize.push_back(14);
        lsize.push_back(1);
        ann = new NetworkLM(lsize, 10, 1);
        suff = "ANN_act";
        morestuff = "ANN_scaling";
        A = .02;
        B = 1.;
    }
    else
    {
        lsize.push_back(7);
        lsize.push_back(100);
        lsize.push_back(1);
        ann = new WaveletNetLM(lsize, 1);
        suff = "WAVE_act";
        morestuff = "WAVE_scaling";
        A = .01;
        B = 1.;
    }
    
    bool _done = ann->restart(suff);
    
    
    FILE * f = fopen(morestuff.c_str(), "r");
    if(f != NULL)
	{
        float val;
        fscanf(f, "A: %e\n", &val);
        A = val;
        printf("A is %e\n", A);
        
        fscanf(f, "B: %e\n", &val);
        B = val;
        printf("B is %e\n", B);
        fclose(f);
    }
	
	vector<double> p1(1), p2(1), p3(1);
	int counter = 0;
    string strout1 = "./plot_T" + 0 + "_A" + 0 + "_A"+ 0 + ".dat";
    ofstream outf1("./plot_T0_A0_A0.dat");
    ofstream outf2("./plot_T1_A0_A0.dat");
	for (double dt=-2.; dt<2.1; dt+=.1)
    {
        for (double dx=-2.; dx<2.1; dx+=.1)
        {
            for (double dy=-2.; dy<2.1; dy+=.1)
            {
                {
                double val;
                ann->predict({dx, dy, dt, -2, 0, 0, -2}, p1);
                ann->predict({dx, dy, dt, -2, 0, 0,  0}, p2);
                ann->predict({dx, dy, dt, -2, 0, 0,  2}, p3);
                double a = 2*p1[0] - 4*p2[0] + 2*p3[0];
                double b = 4*p2[0] - 3*p1[0] - p3[0];
                if (a >= 0)
                {   // positive curvature, max on one boundary
                    val = p1>p3 ? 0 : 1;
                }
                else
                {
                    val = - b/(2*a);
                    val = min(1.,max(0., val));
                }
                outf1 << val << endl;
                }
                {
                double val;
                ann->predict({dx, dy, dt, 0, 0, 0, -2}, p1);
                ann->predict({dx, dy, dt, 0, 0, 0,  0}, p2);
                ann->predict({dx, dy, dt, 0, 0, 0,  2}, p3);
                double a = 2*p1[0] - 4*p2[0] + 2*p3[0];
                double b = 4*p2[0] - 3*p1[0] - p3[0];
                if (a >= 0)
                {   // positive curvature, max on one boundary
                    val = p1>p3 ? 0 : 1;
                }
                else
                {
                    val = - b/(2*a);
                    val = min(1.,max(0., val));
                }
                outf2 << val << endl;
                }
                counter++;
            }
        }
    }
    cout << counter << endl;
    outf1.close();
    outf2.close();
}