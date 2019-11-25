/*
* @file        build-gcsc.cpp
* @author      Dongha Lee (dongha.lee@postech.ac.kr), Pohang University of Science and Technology
* @author      Jinoh Oh (joh@adobe.com), Adobe Systems Incorporated
* @author      Hwanjo Yu (hwanjoyu@postech.ac.kr), Pohang University of Science and Technology
* @date        2019-11-21
*
* OCAM: Out-of-core Coordinate Descent Algorithm for Matirx Completion
*
* This code is implemented based on the author code of
*      Yu et al., "Scalable Coordinate Descent Approaches to Parallel Matrix Factorization for Recommender Systems", in ICDM, 2012.
* This software is free of charge under research purposes.
*
* Usage:
*   - make build-gcsc
*/

#include "util.h"
#include "ocam.h"
#include "csc-idx.h"
#include <cstring>

void exit_with_help()
{
	printf(
	"Usage: omp-pmf-train [options] data_dir [model_filename]\n"
	"options:\n"
	"    -n threads : set the number of threads (default 4)\n"    
	"    -g grids : set the number of grid columns and grid rows (default 16)\n"    
	"    -d build_ondisk : build GCSC file based on disk or not (default 0)\n"
	);
	exit(1);
}

parameter parse_command_line(int argc, char **argv, char *input_filename)
{
	parameter param;   // default values have been set by the constructor 
	int i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{

			case 'n':
				param.threads = atoi(argv[i]);
				break;
			case 'g':
				param.grids = atoi(argv[i]);
				break;
			case 'd':
				param.build_ondisk = atoi(argv[i]);
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	if(i>=argc)
		exit_with_help();

	strcpy(input_filename, argv[i]);

	return param;
}

void build_gcsc(parameter &param, const char* input_filename){

	if(param.build_ondisk)
		buildGCSCFromFile(input_filename, param.grids, param.threads);	
	else 
		buildGCSCFromMatrix(input_filename, param.grids);

	system("sync");
	system("echo 3 > /proc/sys/vm/drop_caches");
	return ;
}

int main(int argc, char* argv[]){
	char input_filename[1024];
	parameter param = parse_command_line(argc, argv, input_filename); 
	build_gcsc(param, input_filename);
	return 0;
}

