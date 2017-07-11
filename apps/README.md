# apps

Each folder contains the files required to prepare the run directory for running an application. Multiple folders here might refer to the same environment of smarties.

When calling the launch script (eg. `launch/launch.sh`) the user specifies a folder contained here, from which the script `setup.sh` is executed.

The script `setup.sh` must place in `${BASEPATH}${RUNFOLDER}/` the factory file, and the executable/launch script that smarties needs to run to launch the application. Be careful not to name the launch script `launch.sh` as that is overwritten by smarties' launch script.

Note that the files are copied in the base run directory and smarties creates a separate directory for launching the application. Therefore, if the launch script is `launchSim.sh`, factory should read `../launchSim.sh`.
