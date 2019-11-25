#include "util.h"
#include "Option.h"

/* 1) Initialize all factor matrices and a core tensor, and 2) Build a test tensor if there exists a test input */
void Initialize(double ***&FactorMat, Tensor &G, int *&Dims, Tensor &Xtest) {

	printf("Initializing all factor matrices and a core tensor ... ");

	int n, i, j, k, g; 

	int order = Option::tensorOrder;
	int rankSize = Option::rankSize;
	int gridSize = Option::gridSize;
	char *InputPath = Option::trainPath;

	Dims = (int *)malloc(sizeof(int)*order);

	int **gridDimCnt, **gridDims;
	Read_Grid_Info(InputPath, gridDimCnt, gridDims);
	for (i = 0; i < order; i++) Dims[i] = gridDimCnt[i][gridSize]; 
	free(gridDimCnt[0]); free(gridDimCnt);
	free(gridDims[0]); free(gridDims);

	G.buildCoreTensor(rankSize);

	int cntDims = 0, totalDims = 0;
	for (i = 0; i < order; i++) totalDims += Dims[i];

	FactorMat = (double ***)malloc(sizeof(double **)*order);
	FactorMat[0] = (double **)malloc(sizeof(double *)*totalDims);
	FactorMat[0][0] = (double *)malloc(sizeof(double)*totalDims*rankSize);

	for (i = 0; i < order; i++) {
		if (i >= 1) {
			FactorMat[i] = FactorMat[i-1] + Dims[i-1];
			cntDims += Dims[i-1];
		}

		int row = Dims[i], col = G.Dims[i];
		for (j = 0; j < row; j++) {
			FactorMat[i][j] = &FactorMat[0][0][(cntDims + j)*col];
			for (k = 0; k < col; k++) {
				if (i == 0) FactorMat[i][j][k] = 0;
				else FactorMat[i][j][k] = frand(0, 1);
			}
		}
	}

	if(Option::testPath != NULL) {
		FILE *fp;
		char tmp[1005];
		int idx; double val;	
	
		fp = fopen(Option::testPath, "r");
		while (fgets(tmp, 1005, fp)) Xtest.nnz++;
		fclose(fp);

		Xtest.val = (double *)malloc(sizeof(double)*Xtest.nnz);
		Xtest.IndexMat = (int **)malloc(sizeof(int *)*Xtest.nnz);
		Xtest.IndexMat[0] = (int *)malloc(sizeof(int)*Xtest.nnz*order);
		for (n = 0; n < Xtest.nnz; n++) Xtest.IndexMat[n] = &(Xtest.IndexMat[0][n*order]);

		fp = fopen(Option::testPath, "r");
		for (n = 0; n < Xtest.nnz; n++) {
			for (i = 0; i < order; i++) {
				fscanf(fp, "%d", &idx);
				Xtest.IndexMat[n][i] = idx-1;
			}
			fscanf(fp, "%lf", &val);
			Xtest.val[n] = val;
		}
		fclose(fp);
	}

	printf("Done ");
	printf("=> Dimensionalities: %d ", Dims[0]);
	for (i = 1; i < order; i++) printf("x %d ", Dims[i]);
	printf("\n\n");
}

/* Save the factor matrices and the core tensor in the result path */
void Print(double ***FactorMat, Tensor &G, int *Dims) {
	printf("Writing all the factor matrices and the core tensor to file ... ");

	char temp[1024];
	int i, j, k = 0;

	char *ResultPath = Option::resultPath;
	int order = Option::tensorOrder;

	for (i = 0; i < order; i++) {
		sprintf(temp, "%s/FACTOR%d", ResultPath, i);
		FILE *fp_factor = fopen(temp, "w");
		for (j = 0; j < Dims[i]; j++) {
			for (k = 0; k < G.Dims[i]; k++) {
				fprintf(fp_factor, "%e\t", FactorMat[i][j][k]);
			}
			fprintf(fp_factor, "\n");
		}
	}
	sprintf(temp, "%s/CORETENSOR", ResultPath);
	FILE *fp_core = fopen(temp, "w");
	for (i = 0; i < G.nnz; i++) {
		for (j = 0; j < order; j++) {
			fprintf(fp_core, "%d\t", G.IndexMat[i][j]);
		}
		fprintf(fp_core, "%e\n", G.val[i]);
	}
	printf("Done\n");

	free(Dims);
}

/* Allocate and free the cache table  */
void Alloc_DeltaMat(double **&DeltaMat, int tableSize, int CoreDim) {
	DeltaMat = (double **)malloc(sizeof(double *)*tableSize);
	DeltaMat[0] = (double *)malloc(sizeof(double)*tableSize*CoreDim);
	for (int n = 0; n < tableSize; n++) DeltaMat[n] = &DeltaMat[0][n*CoreDim];
} 

void Free_DeltaMat(double **DeltaMat) {
	free(DeltaMat[0]);
	free(DeltaMat);
}

/* Create the (N-1)-order permutation table for efficient access to the set of subtensors  */
void Precompute_Permutation(int **&gridPermu) {
	int i, k;
	int gridSize = Option::gridSize;
	int order = Option::tensorOrder;
	int partTensors_N = pow(gridSize, order-1);

	gridPermu = (int **)malloc(sizeof(int *)*partTensors_N);
	gridPermu[0] = (int *)malloc(sizeof(int)*partTensors_N*(order-1));

	memset(gridPermu[0], 0, sizeof(int)*(order-1));
	for (i = 1; i < partTensors_N; i++) {
		gridPermu[i] = &gridPermu[0][i*(order-1)];
		memcpy(gridPermu[i], gridPermu[i-1], sizeof(int)*(order-1));

		k = order-2;
		gridPermu[i][k]++;
		while (gridPermu[i][k] >= gridSize) {
			gridPermu[i][k] -= gridSize;
			gridPermu[i][k-1]++;
			k--;
		}
	}
}

/* Conversion operations from grid index to cell index, or from cell index to grid index */
/* for example, in case of 3-order tensor and G = 4, grid index: [1, 0, 3] <-> cell index: [49] */
int gridIdx2cellIdx(int *gridIdx) {
	int order = Option::tensorOrder;
	int gridSize = Option::gridSize;
	int cellIdx = 0;
	for (int i = 0; i < order; i++)
		cellIdx += gridIdx[i]*pow(gridSize, i);
	return cellIdx;
}

void cellIdx2gridIdx(int *gridIdx, int cellIdx) {
	int order = Option::tensorOrder;
	int gridSize = Option::gridSize;
	for (int i = 0; i < order; i++) {
		gridIdx[i] = cellIdx % gridSize;
		cellIdx /= gridSize;
	}	
}

/* Lock operations for thread-safe access to the cache table */
void acquireSpinLock(spinLock &lock){
        while (lock.test_and_set(std::memory_order_acquire) ){
                ;   
        }   
        return;
}

void releaseSpinLock(spinLock &lock){
        lock.clear(std::memory_order_release);
        return;
}

/* Basic random and arithmetic operations */
double frand(double x, double y) {
	return ((y - x)*((double)rand() / RAND_MAX)) + x;
}

double abss(double x) {
	return x > 0 ? x : -x;
}

