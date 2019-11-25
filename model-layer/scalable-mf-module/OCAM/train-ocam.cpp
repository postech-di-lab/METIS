/*
* @file        train-ocam.cpp
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
*   - make train-ocam
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
	"    -k rank : set the rank (default 10)\n"    
	"    -n threads : set the number of threads (default 4)\n"    
	"    -l lambda : set the regularization parameter lambda (default 0.1)\n"
	"    -g grids : set the number of grid columns and grid rows (default 16)\n"    
	"    -t max_iter: set the number of iterations (default 5)\n"    
	"    -o do_opt : do two-phase optimization or not (default 0)\n"
	);
	exit(1);
}

parameter parse_command_line(int argc, char **argv, char *input_filename, char *model_filename)
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

			case 'k':
				param.k = atoi(argv[i]);
				break;
			case 'n':
				param.threads = atoi(argv[i]);
				break;
			case 'l':
				param.lambda = atof(argv[i]);
				break;
			case 't':
				param.maxiter = atoi(argv[i]);
				break;
			case 'o':
				param.do_opt = atoi(argv[i]);
				break;
			case 'g':
				param.grids = atoi(argv[i]);
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

	if(i<argc-1)
		strcpy(model_filename, argv[i+1]);
	else
	{
		sprintf(model_filename, "%s/model", input_filename);
	}
	return param;
}

void run_ocam(parameter &param, const char* input_filename, const char* model_filename=NULL){
	smat_t R;
	dmat_t W, H;
	testset_t T;
	long rows, cols, train_nnz, test_nnz;

	char buf[1024];
	char meta_filename[1024];
	char gcsc_filename[1024];
	char test_filename[1024];

	sprintf(meta_filename, "%s/meta", input_filename);
	sprintf(gcsc_filename, "%s/gcsc", input_filename);

	FILE *meta_fp = NULL, *model_fp = NULL;

	meta_fp = fopen(meta_filename, "r");
	fscanf(meta_fp, "%ld %ld", &rows, &cols);
	fscanf(meta_fp, "%ld %s", &train_nnz, buf);
	fscanf(meta_fp, "%ld %s", &test_nnz, buf);
	fclose(meta_fp);

	sprintf(test_filename, "%s/%s", input_filename, buf);
	T.load(rows, cols, test_nnz, test_filename);

	if(model_filename) {
		model_fp = fopen(model_filename, "wb");
		if(model_fp == NULL)
		{
			fprintf(stderr,"can't open output file %s\n", model_filename);
			exit(1);
		}
	}

	if(!param.do_opt)
		printf("solver type : OCAM\n");
	else
		printf("solver type : OCAM-opt\n");

	system("echo 3 > /proc/sys/vm/drop_caches");

	initial_dmat_t(W, rows, param.k, false);
	initial_dmat_t(H, cols, param.k, true);

	printf("OCAM starts!\n");
	ocam(gcsc_filename, W, H, T, param);

	if(model_fp) {
		save_dmat_t(model_fp, W, rows, param.k);
		save_dmat_t(model_fp, H, cols, param.k);
		fclose(model_fp);
	}

	free_dmat_t(W);
	free_dmat_t(H);

	return ;
}

int main(int argc, char* argv[]){
	char input_filename[1024];
	char model_filename[1024];
	parameter param = parse_command_line(argc, argv, input_filename, model_filename); 
	run_ocam(param, input_filename, model_filename);
	return 0;
}

