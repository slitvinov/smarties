/*
 *  main.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#else
void omp_set_num_threads(int n)
{
}

int omp_get_max_threads()
{
    return 1;
}

int omp_get_thread_num()
{
    return 0;
}
#endif

#include "ArgumentParser.h"
#include "ErrorHandling.h"
#include "Learners/QLearning.h"
#include "Learners/Learner.h"
#include "ObjectFactory.h"
#include "Settings.h"

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

// A glut wrapper over simulation
// Draws agents, environment, etc.
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
		glutInitWindowSize(800,800);
		glutCreateWindow("School");
		glutDisplayFunc(display);
		glClearColor(1,1,1,1);
		//glClearColor(0,0,0,1);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, 1.0, 0, 1.0, -1, 1);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glEnable(GL_DEPTH_TEST);
		
		// Setup other misc features.
		glEnable(GL_LIGHTING);
		glEnable(GL_NORMALIZE);
		glShadeModel(GL_SMOOTH);
		
		// Setup lighting model.
		glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE);
		glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE); 
		
		glEnable(GL_LIGHT0);
		glEnable(GL_COLOR_MATERIAL);
		
		GLfloat lightpos[] = {(GLfloat)settings.centerX, (GLfloat)settings.centerY, 1.0, 1};
		glLightfv(GL_LIGHT0, GL_POSITION, lightpos);
		GLfloat light0_diffuse[] = {0.9f, 0.9f, 0.9f, 0.9f};   
		glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
	}
};
#endif

// Runs the simulation
void runTest()
{
	// Class creating agents and environment from info in the file
	ObjectFactory factory(settings.configFile);
	System system = factory.getAgentVector();
	
	for (vector<Agent*>::iterator it = system.agents.begin(); it != system.agents.end(); it++)
		(*it)->setEnvironment(system.env);
	
	// Define learning algorithm
	// TODO: Make this through object factory
	Learner* learner = new QLearning(system, settings.gamma, settings.greedyEps, settings.lRate, settings.dt, &profiler);
	
	if (settings.restart) learner->try2restart("");
	
	double dt = settings.dt;
	double time = 0;
	int    iter = 0;
	
	// Save results to dir named  settings.prefix
	if (!Saver::makedir((settings.prefix+"/").c_str())) die("Unable to make a working directory!");
	
	// Various savers
	// TODO: Savers should be specified in factory file
	CouzinsSaver* dsaver = new CouzinsSaver("state.txt");
	learner->registerSaver(dsaver, settings.videoFreq);
	
	CollisionSaver* csaver = new CollisionSaver("coll.txt");
	learner->registerSaver(csaver, settings.saveFreq / 100);
	
	EfficiencySaver* esaver = new EfficiencySaver("eff.txt");
	learner->registerSaver(esaver, settings.saveFreq / 300);
	
	MomentumSaver* msaver = new MomentumSaver("mom.txt");
	learner->registerSaver(msaver, settings.saveFreq / 300);
	
	//RewardSaver* rsaver = new RewardSaver((ofstream*)&cout);//"reward.txt");
	//learner->registerSaver(rsaver, settings.saveFreq / 300);
	
	//NNSaver* nnsaver = new NNSaver((ofstream*)&cout);//new ofstream("reward_good.txt"));
	//learner->registerSaver(nnsaver, settings.saveFreq / 100);
	
	//StateSaver* ssaver = new StateSaver("state.txt");
	//learner->registerSaver(ssaver, settings.saveFreq / 30);
	
	if (settings.videoFreq > 0)
	{
		PhotoSaver* camera = new PhotoSaver("zimg");
		//learner->registerSaver(camera, settings.videoFreq);
	}
	
	while (time < settings.endTime + settings.dt/2.0)
	{
		learner->evolve(time);
		
		if (iter % settings.saveFreq == 0)
			learner->savePolicy("");
		
		time += dt;
		iter++;
		debug2("%d\n", iter);
		
		if (iter % (settings.saveFreq/10) == 0)
		{
		//	settings.greedyEps /= 2;
			info("Time of simulation is now %f\n", time);
			if (debugLvl > 1) profiler.printSummary();
		}

#ifdef _RL_VIZ
		
		if (settings.videoFreq > 0 && iter % settings.videoFreq == 0)
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
	
	exit(0);
}

int main (int argc, char** argv)
{
	vector<OptionStruct> vopts
	({
        {'x', "center_x",   DOUBLE, "X coo of domain center", &settings.centerX,    0.5},
		{'y', "center_y",   DOUBLE, "Y coo of domain center", &settings.centerY,    0.5},
		{'c', "config",     STRING, "Name of config file",    &settings.configFile, (string)"/Users/alexeedm/Documents/Fish/smarties/factory/factoryRL_test1"},
		{'t', "dt",         DOUBLE, "Simulation timestep",    &settings.dt,         0.01},
		{'f', "end_time",   DOUBLE, "End time of simulaiton", &settings.endTime,    1e9},
		{'g', "gamma",      DOUBLE, "Gamma parameter",        &settings.gamma,      0.85},
		{'e', "greedy_eps", DOUBLE, "Greedy epsilon",         &settings.greedyEps,  0.01},
		{'l', "learn_rate", DOUBLE, "Learning rate",          &settings.lRate,      0.01},
		{'s', "rand_seed",  INT,    "Random seed",            &settings.randSeed,   11111},
        {'r', "restart",    NONE,   "Restart",                &settings.restart,    false},
		{'q', "save_freq",  INT,    "Save frequency",         &settings.saveFreq,   10000},
		{'p', "video_freq", INT,    "Video frequency",        &settings.videoFreq,  10000},
		{'a', "scale",      DOUBLE, "Scaling factor",         &settings.scale,      0.012},
		{'v', "debug_lvl",  INT,    "Debug level",            &debugLvl,            2},
		
		{'1', "nneta",      DOUBLE, "NNeta",                  &settings.nnEta,      0.3},
		{'2', "nnalpha",    DOUBLE, "NNalpha",                &settings.nnAlpha,    0.2},
		{'3', "nnlayer1",   INT,    "NN layer 1",             &settings.nnLayer1,   5},
		{'4', "nnlayer2",   INT,    "NN layer 2",             &settings.nnLayer2,   5},
		
		{'6', "best",       NONE,   "Use best shape",         &settings.best,       false},
		{'7', "prefix",     STRING, "Prefix",                 &settings.prefix,     (string)"hpar_best"}
	});
		
	debugLvl = 2;
	settings.centerX = 0.5;
	settings.centerY = 0.5;
	settings.configFile = "/Users/alexeedm/Documents/Fish/smarties/factory/factoryRL_test1";
	settings.dt = 0.01;
	settings.endTime = 1000000;
	settings.gamma = 0.85;
	settings.greedyEps = 0.01;
	settings.lRate = 0.00;
	settings.randSeed = 142144;
	settings.restart = false;
	settings.saveFreq = 100000;
	settings.videoFreq = 500;
	settings.scale = 0.02;
	settings.nnEta = 0.3;
	settings.nnAlpha = 0.2;
	settings.nnLayer1 = 5;
	settings.nnLayer2 = 5;
	
	settings.best = false;
	settings.prefix = "hpar_best";
	
	
	Parser parser(vopts);
	parser.parse(argc, argv);
	if (settings.lRate < 1e-9) settings.immortal = false;
	else settings.immortal = true;
	
	omp_set_num_threads(1);

#ifdef _RL_VIZ
	VisualSupport::run(argc, argv);
#else
	runTest();
#endif

	return 0;
}
