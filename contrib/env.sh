case "`hostname`" in
    eu-login-*)
	. /cluster/apps/local/env2lmod.sh
	module load gcc openmpi
	;;
    falcon.ethz.ch|panda.ethz.ch)
	module load gnu openmpi
	;;
esac
