#include <iostream>
#include <ale_interface.hpp>
#include <cmath>
#include <random>
#include <cstdio>
#include <vector>
#include <functional>
#include "Communicator.h"


//size is [x,y]
//resamples input image to newsize using bilinear interpolation
void resampleimage(const std::vector<unsigned char>& originalimage, int *originalsize, std::vector<double>& newimage, int *newsize)
{
	const double xratio=1.*(*originalsize)/(*newsize);
	const double yratio=1.*(*(originalsize+1))/(*(newsize+1));
	for(int x=0;x<*newsize;++x)
	{
		for(int y=0;y<*(newsize+1);++y)
		{
			newimage[x*(*(newsize+1))+y]=(originalimage[std::floor(x*xratio)*(*(originalsize+1))+std::floor(y*yratio)]+originalimage[std::floor(x*xratio)*(*(originalsize+1))+std::ceil(y*yratio)]+originalimage[std::ceil(x*xratio)*(*(originalsize+1))+std::floor(y*yratio)]+originalimage[std::ceil(x*xratio)*(*(originalsize+1))+std::ceil(y*yratio)])/4; //should instead use distance as prefactor instead of 1/4 const
		}
	}
}
void addframetostate(const std::vector<double>& newframe, std::vector<double>& state)
{
	//newframe has been resized to 84x84
	//order in vector is: z,y,x ?
	const int dim1=4;
	const int dim2=84;
	const int dim3=84;
	//for efficiency use 1 loop only
	for(int x=0;x<dim3;++x)
	{
		for(int y=0;y<dim2;++y)
		{
			for(int z=0;z<dim1-1;++z)
			{
				state[x*dim2*dim1+y*dim1+z]=state[x*dim2*dim1+y*dim1+z+1];
			}
			state[x*dim2*dim1+y*dim1+dim1-1]=newframe[x*dim2+y];
		}
	}	
	//convert to double (maybe done??)

}



Communicator * comm;
int main(int argc, const char * argv[])
{
	/*
	ALEInterface ale;
	
    //Settings
    ale.setInt("random_seed", 42);
    ale.setFloat("repeat_action_probability", 0.25); //default
    ale.setBool("color_averaging", true);
    ale.setInt("frame_skip", 2);


	// Load the ROM file. (Also resets the system for new settings to
    // take effect.)
    ale.loadROM("../../../ROMS/Breakout.bin");//path to rom

	// Get the vector of legal actions
    ActionVect legal_actions = ale.getLegalActionSet();
		*/
    const int n = 1; //n agents
    //communication:
    const int sock = std::stoi(argv[1]);
   
    //communicator class, it needs a socket number sock, given by RL as first argument of execution
    //second and 3rd arguments are dimensions, not correct yet
    const int inputdim=28224; //dimension to which input gets resampled, this is for 4x84x84, uses last 4 images since dynamic is relevant
    Communicator comm(sock,inputdim,1);//3rd argument is actions per turn or available actions? 

	std::vector<double> actions;
	std::vector<unsigned char> curscreen;
	curscreen.reserve(inputdim);
	std::vector<double> resampledscreen;
	resampledscreen.reserve(inputdim);
	std::vector<double> state;
	state.reserve(inputdim);
	double reward=0;
	int info=1;
	int k=0;
	int originalsize[2]={210,160};
	int newsize[2]={84,84};

	for(int i=0;i<1;++i)
	{
		actions.push_back(0);
	}
	for (int i=0;i<inputdim;++i)
	{
		state.push_back(0);
	}
    
    while (true) {
		
		//preprocess state
		//ale.getScreenGrayscale(curscreen);
		//resampleimage(curscreen, originalsize, resampledscreen, newsize);
		//addframetostate(resampledscreen, state);
		comm.sendState(k,info, state, reward);
		comm.recvAction(actions);
		//Action a=legal_actions[actions[0]];
		//reward+=ale.act(a);
		info=0;
		/*if(ale.game_over())
		{
			ale.getScreenGrayscale(curscreen);
			resampleimage(curscreen, originalsize, resampledscreen, newsize);
			addframetostate(resampledscreen, state);
			info=2;
			comm.sendState(k,info, state, reward); //reward?
			ale.reset_game();
			info=1;
			reward=0;
		}*/
			


    }

    return 0;
}

