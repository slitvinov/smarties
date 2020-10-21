OMP_NUM_THREADS=1 valgrind --show-leak-kinds=all --leak-check=full ./main --nTrainSteps 1 1>o 2>e
