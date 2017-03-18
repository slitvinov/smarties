#include <iostream>
#include <ale_interface.hpp>
#include <cmath>
#include <random>
#include <cstdio>
#include <vector>
#include <functional>
#include <Communicator.h>


void addframetostate(const std::vector<unsigned char>& newframe, std::vector<double>& state)
{
	const int dim1=210;
	const int dim2=160;
	//shift state s.t the oldest image drops out and new one gets appended at the end
	state.erase(state.begin(),state.begin()+8400);
	//newframe is of size 210x160 stored rowmajor
	//for efficiency use 1 loop only
	//resize newframe to 105x80
	for(int i=0;i<dim1*dim2/4;++i)
	{
		state.push_back((1.*newframe[(i*2/dim2)*2*dim2+i*2%dim2]+newframe[(i*2/dim2)*2*dim2+i*2%dim2+1]+newframe[((i*2/dim2)*2+1)*dim1+i*2%dim2]+newframe[((i*2/dim2)*2+1)*dim2+i*2%dim2+1])/4.);
	}	
	//convert to double (maybe done??)

}


Communicator * comm;
int main(int argc, const char * argv[])
{
	ALEInterface ale;
	
    //Settings
    ale.setInt("random_seed", 42);
    ale.setFloat("repeat_action_probability", 0.25); //default
    ale.setBool("color_averaging", true);
    ale.setInt("frame_skip", 2);


	// Load the ROM file. (Also resets the system for new settings to
    // take effect.)
    ale.loadROM("../../../ROMS/breakout.bin");//path to rom

	// Get the vector of legal actions
    ActionVect legal_actions = ale.getLegalActionSet();
    
    const int n = 1; //n agents
    //communication:
    const int sock = std::stoi(argv[1]);
   
    //communicator class, it needs a socket number sock, given by RL as first argument of execution
    //second and 3rd arguments are dimensions, not correct yet
    const int inputdim=33600; //dimension to which input gets resampled, this is for 4x105x80, uses last 4 images since dynamic is relevant
    Communicator comm(sock,inputdim,legal_actions.size()); 

	std::vector<double> actions;
	std::vector<unsigned char> curscreen;
	curscreen.reserve(33600);
	std::vector<double> state;
	state.reserve(inputdim);
	double reward=0;
	int info=1;
	int k=0;

    
    while (true) {
		
		//preprocess state
		ale.getScreenGrayscale(curscreen);
		comm.sendState(k,info, state, reward);
		comm.recvAction(actions);
		Action a=legal_actions[actions[0]];
		reward+=ale.act(a);
		//use som CNN to preprocess getScreenGrayscale()
		info=0;
		if(ale.game_over())
		{
			ale.getScreenGrayscale(curscreen);
			info=2;
			comm.sendState(k,info, state, reward); //reward?
			ale.reset_game();
			info=1;
			reward=0;
		}
			


    }

    return 0;
}

