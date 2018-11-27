#ifndef OPTION_H
#define OPTION_H

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

enum optionEnum {
	TRAIN_PATH, TEST_PATH, RESULT_PATH,
	TENSOR_ORDER, CACHE_SIZE, RANK_SIZE, THREAD_SIZE, ITERATION_SIZE, GRID_SIZE, LAMBDA
};

enum solverTypeEnum {
	CD_SOLVER, NN_SOLVER
};

enum buildTypeEnum {
	BUILD_ON_MEMORY, BUILD_ON_DISK, NO_BUILD
};

class Option {

public:

	static char *trainPath;
	static char *testPath;
	static char *resultPath;

	static int buildTensorFile;
	static int solverType;
	static int useSynTensor;

	static int tensorOrder;
	static int cacheSize;
	static int rankSize;
	static int threadSize;
	static int iterationSize;
	static int gridSize;
	static double lambda;
	static int scaleFactor;

	/* for scalability exp. */

	static bool parse(const int argc, char **argv);
};

#endif /* OPTION_H */
