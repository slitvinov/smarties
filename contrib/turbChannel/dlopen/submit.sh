bsub -n 3 -W 00:02 bash -c '. ~/.local/bin/smarties.env && mpirun ./main --workerProcessesPerEnv 2'

