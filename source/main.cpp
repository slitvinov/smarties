/*
 *  main.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <vector>
#include <omp.h>

#include "ArgumentParser.h"
#include "ErrorHandling.h"
#include "QLearning.h"
#include "ObjectFactory.h"
#include "Settings.h"
#include "MRAGProfiler.h"

#include "Savers/AllSavers.h"

#ifdef _RL_VIZ
#ifdef __APPLE__
#include "GLUT/glut.h"
#endif
#endif

using namespace ErrorHandling;
using namespace ArgumentParser;
using namespace std;

void runTest(void);
Settings settings;
int ErrorHandling::debugLvl;

#ifdef _RL_VIZ
struct VisualSupport
{
	static void display()
	{
	}

	static void idle(void)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		runTest();
		glutSwapBuffers();
	}

	static void run(int argc,  char ** argv)
	{
		static bool bSetup = false;

		if (!bSetup)
		{
			setup(argc, argv);
			bSetup = true;
		}

		glutDisplayFunc(display);
		glutIdleFunc(idle);

		glutMainLoop();
	}

	static void setup(int argc,   char ** argv)
	{
		glutInit(&argc, const_cast<char **>(argv));
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
		//glutInitWindowSize(1024,1024);
		glutInitWindowSize(700,700);
		glutCreateWindow("School");
		glutDisplayFunc(display);
		//glClearColor(1,1,1,1);
		glClearColor(0,0,0,1);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, 1.0, 0, 1.0, -1, 1);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}
};
#endif

void runTest()
{
	MRAG::Profiler profiler;
	ObjectFactory factory(settings.configFile);
	System system = factory.getAgentVector();
	
	for (vector<Agent*>::iterator it = system.agents.begin(); it != system.agents.end(); it++)
		(*it)->setEnvironment(system.env);
	
	QLearning learner(system, settings.gamma, settings.greedyEps, settings.lRate, settings.dt, &profiler);
	
	if (settings.restart) learner.try2restart("");
	
	double dt = settings.dt;
	double time = 0;
	double timeSinceLearn = 0;
	int    iter = 0;
	
	RewardSaver* rsaver = new RewardSaver(new ofstream("reward.txt"));
	learner.registerSaver(rsaver, settings.saveFreq / 30);
	
	while (time < settings.endTime)
	{
		learner.evolve(time);
		
		if (iter % settings.saveFreq == 0)
			learner.savePolicy("");
		
		time += dt;
		timeSinceLearn += dt;
		iter++;
		debug("%d\n", iter);
		
		if (iter % (settings.saveFreq/10) == 0)
		{
		//	settings.greedyEps /= 2;
			info("Time of simulation is now %f\n", time);
			profiler.printSummary();
		}

#ifdef _RL_VIZ
		
		if (iter % settings.videoFreq == 0)
		{
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glPushAttrib(GL_ENABLE_BIT);
		
			for (vector<Agent*>::iterator it = system.agents.begin(); it != system.agents.end(); it++)
				(*it)->paint();
		
			glPopAttrib();
			glutSwapBuffers();
		}
#endif
		
	}
}

int main (int argc, char** argv)
{
	const OptionStruct opts[] =
	{
		{'x', "center_x",   DOUBLE, "X coo of domain center", &settings.centerX},
		{'y', "center_y",   DOUBLE, "Y coo of domain center", &settings.centerY},
		{'c', "config",     STRING, "Name of config file",    &settings.configFile},
		{'t', "dt",         DOUBLE, "Simulation timestep",    &settings.dt},
		{'f', "end_time",   DOUBLE, "End time of simulaiton", &settings.endTime},
		{'g', "gamma",      DOUBLE, "Gamma parameter",        &settings.gamma},
		{'e', "greedy_eps", DOUBLE, "Greedy epsilon",         &settings.greedyEps},
		{'d', "learn_dump", DOUBLE, "Learning dump",          &settings.learnDump},
		{'l', "learn_rate", DOUBLE, "Learning rate",          &settings.lRate},
		{'s', "rand_seed",  INT,    "Random seed",            &settings.randSeed},
		{'r', "restart",    NONE,   "Restart",                &settings.restart},
		{'q', "save_freq",  INT,    "Save frequency",         &settings.saveFreq},
		{'p', "video_freq", INT,    "Video frequency",        &settings.videoFreq},
		{'a', "scale",      DOUBLE, "Scaling factor",         &settings.scale},
		{'v', "debug_lvl",  INT,    "Debug level",            &debugLvl}
	};
	
	vector<OptionStruct> vopts(opts, opts + 15);
	
	debugLvl = 2;
	settings.centerX = 0.5;
	settings.centerY = 0.5;
	settings.configFile = "/Users/alexeedm/Documents/Fish/dmitry-RL/data/factoryRL_test1";
	settings.dt = 0.005;
	settings.endTime = 100000;
	settings.gamma = 0.85;
	settings.greedyEps = 0.01;
	settings.learnDump = 1;
	settings.lRate = 0.03;
	settings.randSeed = 142144;
	settings.restart = false;
	settings.saveFreq = 100000;
	settings.videoFreq = 500;
	settings.scale = 0.02;
	settings.shared = true;

	Parser parser(vopts);
	parser.parse(argc, argv);
	
	omp_set_num_threads(2);

#ifdef _RL_VIZ
	VisualSupport::run(argc, argv);
#else
	runTest();
#endif

	return 0;
}
