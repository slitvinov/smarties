# Environments

This folder contains the environments implemented within smarties and the interface to make openai gym compatible.

To create an environment within openai gym refer to its documentation. Advantages include already present tools to create a rendering of the environment.

## To implement a new environment:

* Create a .cpp and .h file for your environment following those already present.

* Implement `void setDims()`:
    * Fill the vector of booleans `sI.inUse`. The number of elements must be equal to the number of components of the state vector that your application sends to smarties. Each component can be `true` if the state variable communicated to smarties is observed by the agent and therefore is used as input to the learning algorithms or `false` is the state variable is not observed by the agent. For example, in the cart-pole example, you might want to hide the velocities from the agent and train a policy only on position and angle.
		* (optional) Fill `sI.mean` and `sI.scale` with the the mean and scale (e.g. standard deviation) of each of the quantities received by smarties. Must be of equal size as `sI.inUse`.
		* Fill `aI.dim`: the dimension of the action vector.
		* (optional, for continuous action algorithms) Fill the vector of booleans `aI.bounded` with `true` for every component of the action vector that should be bounded (see next point).
		* Fill the vector of vectors `aI.values`. It must contain a vector for each dimension of the action vector. The effect of the vector depends on the learning algorithm:
				* **Continuous action algorithms**: Each vector should have at least 2 components representing the upper and lower scale of each component of the action vector. If the corresponding `aI.bounded` is `true` the upper and lower bounds are the max and min of the vector.
				* **Discrete action algorithms**: fill with the number that should be given to the learning algorithm for each of the options for the action component. For example, assuming `aI.dim = 1`, if the application expects an integer labeling each action option, fill the vector with the integers (+0.1 for safety since everything is communicated as float64). On the other hand, if the application expects a continuous value, the discretization of the action space can be done within smarties by filling each `aI.values` vector with the continuous-valued options.
		* Call `commonSetup()`
* (optional) Implement `bool pickReward(const State& sOld, const Action& a,
				const State& sNew, Real& reward, const int info)`. This function allows the user to modify the reward used for training the agent, for example by reading elements of `sNew`. If `reward` is not modified, training is performed with the reward sent by the application. Terminate the function with `return info==_AGENT_LASTCOMM;`

* (optional) Implement `bool predefinedNetwork(Builder* const net) const` and `return true;`. This function can be used if the environment calls for convolutional layers (which cannot be requested from the settings file), before the fully connected (feedforward, recurrent) layers read from the settings file.  

* Include the header file and place a forward declaration of the environment class in `source/AllSystems.h`.

* Include your environment in `ObjectFactory.cpp`. The ObjectFactory reads the `factory` in the run directory. The `factory` file must contain
    * a string identifying your environment (eg. `CartEnvironment`)
		* `exec=` a string of the command to be executed to launch your application (e.g. `./launchSim.sh`)
		* `n=` the number of active agents within the environment. smarties does not yet support each agent having a different policy.
* Add your environment to the `Makefile`
