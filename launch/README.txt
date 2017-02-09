*_train.txt - TO BE USED TO FIND RL POLICY. either use "train" or "learn". Master-slave framework: Create MPI job that is only RL multiple nodes. Can specify multiple masters. Each process creates a slave sim.
*_learn.txt - 
*_client.txt - TO BE USED TO RUN SIMULATION WITH CONVERGED POLICY : launch simulation, sim creates new process which is smarties, then it gets orders from sim. Main task submitted is the sim, this forks the task that does RL.

> Keep nthreads at least 2, otherwise there is no slave. You can use this mode for training from stored transitions, but not for running a fresh training sim.
