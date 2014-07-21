/*
 *  untitled.h
 *  IF2D_ROCKS
 *
 *  Created by Chloe Mimeau on 4/1/11.
 *  Copyright 2011 ETHZ. All rights reserved.
 *
 */

#include "IF2D_FluidMediatedLau.h"

using namespace MRAG;
using namespace std;

IF2D_Test * test = NULL;
#undef _MRAG_GLUT_VIZ


int main (int argc, const char ** argv) 
{
	printf("///////////////////////////////////////////////////////////////\n");
	printf("////////////        THIS IS LAURENT'S MAIN        ///////////////\n");
	printf("///////////////////////////////////////////////////////////////\n");

	ArgumentParser parser(argc, argv);

	Environment::setup(max(1, parser("-nthreads").asInt()));

	if( parser("-study").asString() == "FLUID_MEDIATED_LAU" )
		test = new IF2D_FluidMediatedLau(argc, argv);
	else
	{
		printf("Study case not defined!\n"); 
		abort();
	}

	tbb::tick_count t1,t0;

	{
		t0=tbb::tick_count::now();
		while(true)
		  test->run();

		t1=tbb::tick_count::now();
	}

	printf("we spent: %2.2f \n",(t1-t0).seconds());

	return 0;
}
