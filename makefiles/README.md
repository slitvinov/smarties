# Dependencies
* **Euler** :
  ```
  module load new modules gcc/6.3.0 open_mpi/2.1.1 binutils/2.25 hwloc/1.11.0
  ```
* **Falcon** Have in the bashrc:
	```
	export LD_LIBRARY_PATH=/home/novatig/mpich-3.2/gcc7.1_install/lib/:/usr/local/gcc-7.1/lib64/:$LD_LIBRARY_PATH
	export PATH=/usr/local/gcc-7.1/bin/:$PATH
	```
* **Panda** Have in the bashrc:
	```
	export PATH=/opt/mpich/bin/:$PATH
	export LD_LIBRARY_PATH=/opt/mpich/lib/:$LD_LIBRARY_PATH
	```
* **Daint** Openai's gym should be installed and activated with virtualenv.
	```
	module swap PrgEnv-cray PrgEnv-gnu
	module load daint-gpu python_virtualenv/15.0.3
	```
* **MacOs** Install `mpich` with `gcc`.
