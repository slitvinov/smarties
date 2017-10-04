# apps

Each folder contains the files required to prepare the run directory for running an application. Multiple folders here might refer to the same environment of smarties.

When calling the launch script (eg. `launch/launch.sh`) the user specifies a folder contained here, from which the script `setup.sh` is executed.



# How to set up your application (c++)

* Create your application. Pseudocode:
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
    - Optional stuff
    ```
    //OPTIONAL: action bounds
    bool bounded = true;
    vector<double> upper_action_bound{10}, lower_action_bound{-10};
    comm.set_action_scales(upper_action_bound, lower_action_bound, bounded);

    /*
      // ALTERNATIVE for discrete actions:
      vector<int> n_options = vector<int>{2};
      comm.set_action_options(n_options);
      // will receive either 0 or 1, app chooses resulting outcome
    */

    //OPTIONAL: hide state variables.
    // e.g. show cosine/sine but not angle
    vector<bool> b_observable = {true, true, true, false, true, true};
    comm.set_state_observable(b_observable);

    //OPTIONAL: set space bounds
    vector<double> upper_state_bound{ 1,  1,  1,  1,  1,  1};
    vector<double> lower_state_bound{-1, -1, -1, -1, -1, -1};
    comm.set_state_scales(upper_state_bound, lower_state_bound);
    ```

* Create an executable script `setup.sh`. The script must:
    - Place in `${BASEPATH}${RUNFOLDER}/`  the executable/launch script that smarties needs to run to launch your application. Be careful not to name the launch script `launch_smarties.sh` or `run.sh` as that is overwritten by smarties' launch script
    - Specify the name of your executable/launch script. As default, smarties will launch the script `launchSim.sh`, if your script has a different name, or you want to launch an executable, write in `setup.sh`:
        `SETTINGS+=" --launchfile ${YOUR_EXECUTABLE}"`
    - If your application works with a non-standard environment, you need to specify its name for smarties' `ObjectFactory`. This should never be needed, except if your application is itself distributed with MPI. (Look at environment `FishDCyl_3D` for inspiration if you dare.) If you know what you are doing:
        `SETTINGS+=" --environment ${YOUR_ENVIRONMENT}"`
