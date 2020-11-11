bsub -n 5 -W 00:10 bash -c '. ~/.local/bin/smarties.env && mpirun ./main --workerProcessesPerEnv 4'
