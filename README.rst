smarties
**********

**smarties is a Reinforcement Learning (RL) software designed with the following
objectives**

- high-performance C++ implementations of [Remember and Forget for Experience Replay](https://arxiv.org/abs/1807.05827) and other deep RL learning algorithms including V-RACER, CMA, PPO, DQN, DPG, ACER, and NAF.

- the environment application determines at runtime the properties of the control problem to be solved. For example, the number of the agents in the environment, whether they are solving the same problem (and therefore they should all contribute to learning a common policy) or they are solving different problems (e.g. competing or collaborating). The properties of each agent's state and action spaces. Whether one or more agents are dealing with a partially-observable problem, which causes the learners to automatically use recurrent networks for function approximation. Whether the observation has to be preprocessed by convolutional layers and the properties thereof.

- the environment application is in control of the learning progress. More specifically, smarties supports applications whose API design is similar to that of OpenAI gym, where the environment is a self-contained function to be called to receive a new observation and advance the simulation in time. However, smarties also supports a more realistic API where the environment simulation dominates the structure of the application code. In this setting, it is the RL code that is called whenever new observations are available and require the agent to pick an action.

- the environment application determines the  computational resources available to train and run simulations, with support for distributed (MPI) codebases.

- minimally intrusive plug-in API that can be inserted into C++, python and Fortran simulation software.  

To cite this repository, reference the paper::

    @inproceedings{novati2019a,
        title={Remember and Forget for Experience Replay},
        author={Novati, Guido and Koumoutsakos, Petros},
        booktitle={Proceedings of the 36\textsuperscript{th} International Conference on Machine Learning},
        year={2019}
    }

.. contents:: **Contents of this document**
   :depth: 3

Install
======
Unix
------

Smarties requires gcc version 6.1 or greater, a thread-safe (at least `MPI_THREAD_SERIALIZED`) implementation of MPI, and a serial BLAS implementation with CBLAS interface. Furthermore, in order to test on the benchmark problems, OpenAI gym or the DeepMind Control Suite with python>=3.5. MPI and OpenBLAS can be installed by running the ``install_dependencies.sh`` script.

.. code:: shell

    git clone https://github.com/cselab/smarties.git
    cd smarties
    mkdir -p build
    cd build
    cmake ..
    make -j

Mac OS
------
Installation on Mac OS is a bit more laborious due to to the LLVM compiler provided by Apple not supporting OpenMP threads. First, install the required dependencies as:

.. code:: shell

    brew install llvm open-mpi openblas

Now, we have to switch from Apple's LLVM compiler to the most recent LLVM compiler as default for the user's shell:

.. code:: shell

    echo "alias cc='/usr/local/opt/llvm/bin/clang'" >> ~/.bash_profile
    echo "alias gcc='/usr/local/opt/llvm/bin/clang'" >> ~/.bash_profile
    echo "alias g++='/usr/local/opt/llvm/bin/clang++'" >> ~/.bash_profile
    echo "alias c++='/usr/local/opt/llvm/bin/clang++'" >> ~/.bash_profile
    echo "export PATH=/usr/local/opt/llvm/bin:\${PATH}" >> ~/.bash_profile

Then we are ready to get and install smarties:

.. code:: shell

    git clone https://github.com/cselab/smarties.git
    cd smarties/makefiles
    make -j


Examples
========

C++
-----
The basic structure of a C++ based application for smarties is structured as:

.. code:: shell

    #include "smarties.h"
    
    inline void app_main(smarties::Communicator*const comm, int argc, char**argv)
    {
      comm->set_state_action_dims(state_dimensionality, action_dimensionality);
      Environment env;
    
      while(true) { //train loop
        env.reset(comm->getPRNG()); // prng with different seed on each process
        comm->sendInitState(env.getState()); //send initial state
    
        while (true) { //simulation loop
          std::vector<double> action = comm->recvAction();
          bool isTerminal = env.advance(action); //advance the simulation:
    
          if(isTerminal) { //tell smarties that this is a terminal state
            comm->sendTermState(env.getState(), env.getReward());
            break;
          } else  # normal state
            comm->sendState(env.getState(), env.getReward());
        }
      }
    }
    
    int main(int argc, char**argv)
    {
      smarties::Engine e(argc, argv);
      if( e.parse() ) return 1;
      e.run( app_main );
      return 0;
    }

For compilation, the following flags should be set in order for the compiler to find smarties:

.. code:: shell

    LDFLAGS="-L${SMARTIES_LIB} -lsmarties"
    CPPFLAGS="-I${SMARTIES_INCLUDE}"


Python  
-----
smarties uses pybind11 for seamless compatibility with python. The structure of the environment application is almost the same as the C++ version:

.. code:: shell

    import smarties as rl
    
    def app_main(comm):
      comm.set_state_action_dims(state_dimensionality, action_dimensionality)
      env = Environment()
    
      while 1: #train loop
        env.reset() # (slightly) random initial conditions are best
        comm.sendInitState(env.getState())
    
        while 1: #simulation loop
          action = comm.recvAction()
          isTerminal = env.advance(action)
    
          if terminated:  # tell smarties that this is a terminal state
            comm.sendTermState(env.getState(), env.getReward())
            break
          else: # normal state
            comm.sendState(env.getState(), env.getReward())
    
    if __name__ == '__main__':
      e = rl.Engine(sys.argv)
      if( e.parse() ): exit()
      e.run( app_main )


Other examples
--------------
The ``apps`` folder contains a number of examples showing the various use-cases of smarties. Each folder contains the files required to define and run a different application. While it is generally possible to run each case as ``./exec`` or ``./exec.py``, smarties will create a number of log files, simulation folders and restart files. Therefore it is recommended to manually create a run directory or use the launch scripts contained in the ``launch`` directory.

The applications that are already included are:
- ``apps/cart_pole_cpp``: simple C++ example of a cart-pole balancing problem  

- ``apps/cart_pole_py``: simple python example of a cart-pole balancing problem  

- ``apps/cart_pole_f90``: simple fortran example of a cart-pole balancing problem  

- ``apps/cart_pole_many``: example of two cart-poles that define different decision processes: one performs the opposite of the action sent by smarties and the other hides some of the state variables from the learner (partially observable) and tehrefore requires recurrent networks.  

- ``apps/cart_pole_distribEnv``: example of a distributed environment which requires MPI. The application requests M ranks to run each simulation. If the executable is ran as ``mpirun -n N exec``, (N-1)/M teams of processes will be created, each with its own MPI communicator. Each simulation process contains one or more agents.  

- ``apps/cart_pole_distribAgent``: example of a problem where the agent themselves are distributed. Meaning that the agents exist across the team of processes that run a simulation and get the same action to perform. For example flow actuation problems where there is only one control variable (eg. some inflow parameter), but the entire simulation requires multiple CPUs to run.  

- ``apps/predator_prey``: example of agents competing.  

- ``apps/glider``: example of an ODE-based control problem that requires precise controls, used for the paper [Deep-Reinforcement-Learning for Gliding and Perching Bodies](https://arxiv.org/abs/1807.03671)  

- ``apps/func_maximization/``: example of function fitting and maximization, most naturally approached with CMA.  

- ``apps/OpenAI_gym``: code to run most gym application, including the MuJoCo based robotic benchmarks shown in [Remember and Forget for Experience Replay](https://arxiv.org/abs/1807.05827)  

- ``apps/OpenAI_gym_atari``: code to run the Atari games, which automatically creates the required convolutional pre-processing  

- ``apps/Deepmind_control``: code to run the Deepmind Control Suite control problems  

Launching
=========
The folder `launch` contains the launch scripts, some description on how to use them, and the description of the output files. Some tools to postprocess the outputs are in the folder `pytools`.  

