# launch

The main files that will be maintained are `launch.sh`, `launch_openai.sh`, `launchDaint.sh`, `run.sh`.

`launch.sh` is your run-of-the-mill launch script. You must provide:
* the name of the folder to run in, which will be placed in `../runs/`.
* the number of omp-threads (>=2). The master thread handles mpi communication (therefore your mpi distribution must support thread safety, such as mpich), the others train the network by sampling transition data. This does not control the number of threads (if any) used by app.
* the number of mpi-ranks
    - Usually at least 2. Note that if `--nMasters` is greater than 1 (for most cases it _will_ be 1), then take care that the mpi ranks allow for each master to have the same number of slaves. If the environment requires multiple ranks itself then the number of mpi ranks minus `--nMasters` must be a multiple of the number of ranks required by each instance of the application.
    - 1 is acceptable in two circumstances: **1)** training exclusively from data (then either the settings file should contain the variable `--path some/path/history.txt` pointing to a data file of stored transition or smarties must find a binary dump, ie. `agent_...raw`, in the run directory), **2)** dumping the policy (happens if no data is read from file, but chances of you wanting this for now are very slim)
* the path or name of the folder in the `apps` folder containing the files defining your application.
* the path to the settings file.
* (optional) the base path to a previously saved policy (the base name is usually `path/policy`, and the policy is then composed of multiple files such as `path/policy_net.raw` and so on).
* (optional) the number of task per node, not needed for `launch.sh`.

* `launch_openai.sh` behaves much the same way, but instead of providing a path to an application provide the name of the openai environment (e.g. `CartPole-v1`)

These two scripts set up the launch environment and directory, and then call `run.sh`.

* `launchDaint.sh` .. it works on Daint. Main changes are that run folder is in `/scratch/snx3000/${MYNAME}/smarties/`, the number of threads is hardcoded to 24, and `run.sh` is not used.

# misc

* The `_client.sh` scripts will temporarily no longer be supported since their usefulness was outweighed by the confusion of the users. Also, by design it was not possible to make it work on Daint out of the box without the user writing its own `launchDaint.sh` for the application.

* To evaluate a policy:
    - Make sure `--bTrain 0`
    - (optional) `--greedyEps 0`
    - Run with at least 1 thread, one mpi-rank for the master plus the number of mpi-ranks for one instance of the application (usually 1).
    - To run a finite number of times, the option `--totNumSteps` is recycled if `bTrain==0` to be the number of sequences that are observed before terminating (instead of the maximum number of gradient steps done for the training if `bTrain==1`)
    - Make sure the policy is read correctly (eg. if code was compiled with different features or run with different algorithms, network might have different shape), by comparing the `restarted_policy...` files and the policy provided as argument of the launch script.

* For a description of the settings read `source/Settings.h`. The file follows 	an uniform pattern:
	```
	#define CHARARG_argName 'a'
	#define COMMENT_argName "Argument description"
	#define TYPEVAL_argName int
	#define TYPENUM_argName INT
	#define DEFAULT_argName 0
	int argName = DEFAULT_argName;
	```

    - The first line (`CHARARG_`) defines the `char` associated with the argument, to run `./rl -a val`
    - The second line (`COMMENT_`) contains a brief description of the argument.
    - The third line (`TYPEVAL_`) contains the variable type: must be either `int`, `string`, `Real`, or `char`.
    - The fourth line (`TYPENUM_`) specifies the enumerator in the argument parser associated with the variable type: `INT`, `STRING`, `REAL`.
    - The fifth line (`DEFAULT_`) specifies the default value for the argument.
    - The sixth and last line assigns in the constructor of `Settings` the default value to the variable.
    - Later in the text `ArgumentParser` is called to read from the arguments. The string that defines each argument is `argName`. Therefore run as `./rl --argName val`.
