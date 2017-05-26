#include "alebotEnvironment.h"

alebotEnvironment::alebotEnvironment(const Uint _nAgents, const Uint _nActions, const string _execpath,
																 const Uint _rank, Settings & settings) :
Environment(_nAgents, _execpath, _rank, settings), legalActions(_nActions)
{
}

bool alebotEnvironment::predefinedNetwork(Builder* const net) const
{
	//indices are: feature map (color), height, width
	//this function can be used if environment requires particular network settings
	//i.e. not fully connected LSTM/FF network
	//i.e. if you want to use convolutions
	//CNN should be:
	//input 84x84x4

	//1st layer (Convolution):
	//	32 filters (8x8)
	//	stride 4
	//	applies a rectifier nonlinearity

	//2nd layer (Convolution):
	//	64 filters (4x4)
	//	stride 2
	//	applies a rectifier nonlinearity

	//3rd layer (Convolution)
	//	64 filters (3x3)
	//	stride 1
	//	applies a rectifier

	//4th layer (fully connected)
	//	512 rectifier units

	//output layer (fully connected), linear
	//	1 output per valid action (4-18) assume 18
	{
		const int inputsize[3] = {84,84,4};
		net->add2DInput(inputsize);
	}
	{
		const int filterSize[3] = {8,8,32}; //not too sure about 3rd dim (number of rectifier units)
		const int padding[2] = {0,0};
		const int outSize[3] = {20,20,32};
		const int stride[2] = {4,4};
		net->addConv2DLayer(filterSize, outSize, padding, stride, "Relu");
	}
	{
		const int filterSize[3] = {4,4,64};
		const int padding[2] = {0,0};
		const int outSize[3] = {9,9,64};
		const int stride[2] = {2,2};
		net->addConv2DLayer(filterSize, outSize, padding, stride, "Relu");
	}
	{
		const int filterSize[3] = {3,3,64};
		const int padding[2] = {0,0};
		const int outSize[3] = {7,7,64};
		const int stride[2] = {1,1};
		net->addConv2DLayer(filterSize, outSize, padding, stride, "Relu");
	}

	//AFTER CONV LAYERS, LAYER SHAPE IS CREATED BY LEARNER BY READING SETTINGS
	/*
	{
		 //add fully connected layer with 512 rectifier units
		 const int nunits=512;
		 net->addLayer(nunits, "normal");
	}

	net->addOutput(legalActions, "Normal");
	*/
	return true;
}

void alebotEnvironment::setDims() //this environment is for the cart pole test
{
    {
		//this tells which part of the input are relevant (in this case all)
        sI.inUse.clear();
        for(Uint i=0;i<84*84*4;++i)
        {
			sI.inUse.push_back(true); //ignore, leave as is

		}
    }
    {
        aI.dim = 1; //number of action that agent can perform per turn: usually 1 (eg DQN)
        aI.values.resize(aI.dim);
        for (Uint i=0; i<aI.dim; i++) {
        	for (Uint j=0;j<legalActions;++j) //should be something like: actionvec=ale.getLeagalActionSet() actionvec.length: Pass aleInterface without recreating it?
        	{
				aI.values[i].push_back(j+0.1);
			}
        }
    }
    commonSetup(); //required
}

bool alebotEnvironment::pickReward(const State & t_sO, const Action & t_a,
																 const State& t_sN, Real& reward,const int info)
{
    return info==2; //if info==2 then terminal state, reward remains unchanged
}
