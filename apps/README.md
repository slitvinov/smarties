# apps

Each folder contains the files required to prepare the run directory for running an application. Multiple folders here might refer to the same environment of smarties.

When calling the launch script (eg. `launch/launch.sh`) the user specifies a folder contained here, from which the script `setup.sh` is executed.



# How to set up your application

* *(c++)* Create your application. Pseudocode:
    ```
    //communication: socket number is given by RL as first argument of execution
    const int socket = std::stoi(argv[1]);
    Communicator comm(socket, state_vars, control_vars);
    Environment env; //constructor of your environment

    while(true) { //train loop
      //reset environment to initial state:
      env.reset(comm.gen); //comm contains rng with different seed on each rank
      comm.sendInitState(env.getState()); //send initial state

      while (true) { //simulation loop
        vector<double> action = comm.recvAction();        
        bool terminated = env.advance(action); //advance the simulation
        vector<double> state = env.getState();
        double reward = env.getReward();

        if(terminated) { //tell smarties that this is a terminal state
          comm.sendTermState(state, reward);
          break;
        } else comm.sendState(state, reward);
      }
    }
    ```
    - This workflow is correct for 1 agent per simulation. Note that while sending state you have to specify whether it is an initial state, normal state, terminal state.
    - Socket number is to create a secure channel of communication between smarties and your application, will be moved inside `Communicator` in future releases.
    - `state_vars` denotes the dimensionality of the state space. Ie. if you state is `x` and `y` of an agent, `state_vars = 2`.
    - `control_vars` denotes the dimensionality of the action space. Ie. if you control the `v_x` and `v_y` of the agent, `control_vars = 2`.
    - Optional stuff. Commands to be used before sending the first state.
        * Action bounds. Useful if you know the range of meaningful actions.
        ```
        comm.set_action_scales(upper_action_bound, lower_action_bound, bounded);
        ```
        Ie. in the cart pole problem, the actions are in the range -10 to 10. With RL left with default parameters, it will start exploration with actions distributed with mean 0 and std deviation 1. If you submit this command with `vector<double> upper_action_bound{10}` and `lower_action_bound{-10};`, the action space will be rescaled to that range, and the exploration will effectively be with std deviation 10.
        If `bool bounded` is set to true, actions can ONLY be performed in the specified range.
        * Action options. This command works for discrete action spaces.
        ```
        comm.set_action_options(n_options);
        ```
        `n_options` can either be an integer or a vector of integers, with the same size as the dimensionality of the action space. This means that, if the agent can control 2 numbers, and for each number you want to provide two options: `vector<int> n_options{2, 2}`.
        * Observability of state variables. Some state variables might not be observable to the agent, or you might want to pass additional data to smarties for logging, but not include it in the state vector.
        ```
        comm.set_state_observable(b_observable);
        ```
        `b_observable` is a `vector<bool>` with dimensionality `state_vars`.
        * Input to neural networks should be normalized. If you know the range of the state variables, provide it as:
        ```
        comm.set_state_scales(upper_state_bound, lower_state_bound);
        ```
        The two arguments are `vector<double>` with dimensionality `state_vars`. If not provided, range is assumed to be -1 to 1 per variable.
    - You can provide in the constructor a number of agents. Ie:
      `Communicator comm(socket, state_vars, control_vars, number_of_agents);`
      Multi agent systems require additional care. First of all, smarties does not support environments with independent agents. This means that, while actions request can happen independently, if an agent sends a terminal state, smarties assumes that the environment is over: all other agents are about to send a terminal state and restart. However, terminal states are special states in reinforcement learning, because `V(term_state) = reward(term_state)`. If some agents reach a terminal state and the others do not, first send all the terminal states, then use the function `comm.sendCompleteTermination()`. smarties will treat the trajectories of the other agents, those which did not send a terminal state, as interrupted sequences. This means that the value of the last state will not be assigned to be equal to the last reward. E.g: in an environment with multiple cars, if 2 cars crash they reach a terminal state, but after the crash the environment will still be restarted for all cars.

* *(python)* Create your application. Pseudocode:
    ```
    comm = Communicator() # create communicator with smarties
    env = comm.get_gym_env()

    while True: #training loop
        observation = env.reset()
        comm.send_state(observation, initial=True) #send initial state

        while True: # simulation loop
            action = comm.recv_action() #receive action from smarties
            #advance the environment
            observation, reward, done, info = env.step(action)
            #send the observation to smarties
            comm.send_state(observation, reward=reward, terminal=done)
            if done: break
    ```
    All the customization options available in the *c++* version are also available in *python*. There might not be functions ready in `Communicator.py`, but you can modify class variables following the example of the function `self read_gym_env(self)`.

* You might want to create a script `launchSim.sh` that launches your executable, reads input files or execution settings.

* Create an executable script `setup.sh`. The script must:
    - Place in `${BASEPATH}${RUNFOLDER}/`  the executable/launch script that smarties needs to run to launch your application. Be careful not to name the launch script `launch_smarties.sh` or `run.sh` as that is overwritten by smarties' launch script
    - Specify the name of your executable/launch script. As default, smarties will launch the script `launchSim.sh`, if your script has a different name, or you want to launch an executable, write in `setup.sh`:
        `SETTINGS+=" --launchfile ${YOUR_EXECUTABLE}"`
    - If your application works with a non-standard environment, you need to specify its name for smarties' `ObjectFactory`. This feature is needed in two circumstances: if you application is distributed with MPI, or if you require convolutional layers as imput to the network. There is some work in progress. (Look at environment `FishDCyl_3D` for inspiration if you dare.) If you know what you are doing:
        `SETTINGS+=" --environment ${YOUR_ENVIRONMENT}"`
