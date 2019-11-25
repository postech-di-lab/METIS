#include "Option.h"

char *Option::trainPath = NULL;
char *Option::testPath = NULL;
char *Option::resultPath = NULL;

int Option::buildTensorFile;
int Option::solverType;

int Option::tensorOrder;
int Option::cacheSize;
int Option::rankSize;
int Option::threadSize;
int Option::iterationSize;
int Option::gridSize;
double Option::lambda;

static struct option long_options[] =
{
	{"train-path", required_argument, 0, TRAIN_PATH},	
	{"test-path", required_argument, 0, TEST_PATH},	
	{"result-path", required_argument, 0, RESULT_PATH},	

	{"build-on-memory", no_argument, &Option::buildTensorFile, BUILD_ON_MEMORY},	
	{"build-on-disk", no_argument, &Option::buildTensorFile, BUILD_ON_DISK},	
	{"no-build", no_argument, &Option::buildTensorFile, NO_BUILD},	

	{"cd-solver", no_argument, &Option::solverType, CD_SOLVER},	
	{"nn-solver", no_argument, &Option::solverType, NN_SOLVER},	

	{"tensor-order", required_argument, 0, TENSOR_ORDER},	
	{"cache-size", required_argument, 0, CACHE_SIZE},	
	{"rank-size", required_argument, 0, RANK_SIZE},	
	{"thread-size", required_argument, 0, THREAD_SIZE},	
	{"iteration-size", required_argument, 0, ITERATION_SIZE},	
	{"grid-size", required_argument, 0, GRID_SIZE},	
	{"lambda", required_argument, 0, LAMBDA},	
	{0, 0, 0, 0}

};

bool Option::parse(const int argc, char **argv) {

	int opt;

	Option::trainPath = NULL;
	Option::testPath = NULL;
	Option::resultPath = NULL;

	Option::buildTensorFile = BUILD_ON_MEMORY;
	Option::solverType = CD_SOLVER;

	Option::tensorOrder = 0;
	Option::cacheSize = 1000000;
	Option::rankSize = 10;
	Option::threadSize = 1;
	Option::iterationSize = 10;
	Option::gridSize = 1;
	Option::lambda = 0.01;

	int option_index = 0;
	while(1) {

		opt = getopt_long(argc, argv, "", long_options, &option_index);
		if (-1 == opt)
			break;

		switch (opt) {

			case TRAIN_PATH:
				Option::trainPath = optarg;
				break;		
			case TEST_PATH:
				Option::testPath = optarg;
				break;		
			case RESULT_PATH:
				Option::resultPath = optarg;
				break;		
			case TENSOR_ORDER:
				Option::tensorOrder = atoi(optarg);
				break;		
			case CACHE_SIZE:
				Option::cacheSize = atoi(optarg);
				break;
			case RANK_SIZE:
				Option::rankSize = atoi(optarg);
				break;
			case THREAD_SIZE:
				Option::threadSize = atoi(optarg);
				break;
			case ITERATION_SIZE:
				Option::iterationSize = atoi(optarg);
				break;
			case GRID_SIZE:
				Option::gridSize = atoi(optarg);
				break;
			case LAMBDA:
				Option::lambda = atof(optarg);
				break;
		}
	}

	return true; 
}
