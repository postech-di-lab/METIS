#include "util.h"
#include "Option.h"
#include "Tensor.h"

/* Build a Tensor object from the raw text file */
void Tensor::buildTensorFromRawFile(const char *InputPath) { 

	FILE *fp;
	int n, i, j, idx;
	double v;
	char tmp[1005];

	order = Option::tensorOrder;
	Dims = (int *)malloc(sizeof(int)*order);
	memset(Dims, 0, sizeof(int)*order);

	fp = fopen(InputPath, "r");
	while (fgets(tmp, 1005, fp)) nnz++;
	fclose(fp);

	IndexMat = (int **)malloc(sizeof(int*)*nnz);
	IndexMat[0] = (int *)malloc(sizeof(int)*nnz*order);
	for(n = 0; n < nnz; n++) IndexMat[n] = &IndexMat[0][n*order];

	val = (double *)malloc(sizeof(double)*nnz);

	fp = fopen(InputPath, "r");
	for (n = 0; n < nnz; n++) {
		for (i = 0; i < order; i++) {
			fscanf(fp, "%d", &idx);
			IndexMat[n][i] = idx - 1;
			if (Dims[i] < idx) Dims[i] = idx;
		}
		fscanf(fp, "%lf", &v);
		val[n] = v;
	}
	fclose(fp);

	int cntSize = 0;
	for (i = 0; i < order; i++)
		cntSize += Dims[i] + 1;

	WhereMat = (int **)malloc(sizeof(int*)*order);
	WhereMat[0] = (int *)malloc(sizeof(int)*order*nnz);
	for(i = 0; i < order; i++) WhereMat[i] = &WhereMat[0][i*nnz];

	CountMat = (int **)malloc(sizeof(int*)*order);
	CountMat[0] = (int *)malloc(sizeof(int)*cntSize);
	for(i = 1; i < order; i++) CountMat[i] = CountMat[i-1] + Dims[i-1] + 1;

	assignTensorIndex();
}

/* Build a core tensor */
void Tensor::buildCoreTensor(int CoreDim) {

	int n, i, j, k;

	order = Option::tensorOrder;	
	nnz = pow(CoreDim, order);
	Dims = (int *)malloc(sizeof(int)*order); 
	for (i = 0; i < order; i++) Dims[i] = CoreDim;

	IndexMat = (int **)malloc(sizeof(int*)*nnz);
	IndexMat[0] = (int *)malloc(sizeof(int)*nnz*order);
	for(n = 0; n < nnz; n++) IndexMat[n] = &IndexMat[0][n*order];

	val = (double *)malloc(sizeof(double)*nnz);

	for (i = 0; i < nnz; i++) {
		val[i] = frand(0, 1);
		if (i == 0) {
			for (j = 0; j < order; j++) IndexMat[i][j] = 0;
		}
		else {
			for (j = 0; j < order; j++) {
				IndexMat[i][j] = IndexMat[i-1][j];
			}
			k = order - 1;
			IndexMat[i][k]++;
			while (IndexMat[i][k] >= Dims[k]) {
				IndexMat[i][k] -= Dims[k];
				IndexMat[i][k-1]++;
				k--;
			}
		}
	}
}

/* Complete all the internal structures of a tensor */
void Tensor::assignTensorIndex() {

	int n, i, j;

	int cntSize = 0;
	for (i = 0; i < order; i++)
		cntSize += Dims[i] + 1;

	int **tempMat = (int **)malloc(sizeof(int *)*order);
	tempMat[0] = (int *)malloc(sizeof(int)*cntSize);
	for(i = 1; i < order; i++) tempMat[i] = tempMat[i-1] + Dims[i-1] + 1;

	memset(CountMat[0], 0, sizeof(int)*cntSize);
	memset(tempMat[0], 0, sizeof(int)*cntSize);

	for (n = 0; n < nnz; n++) {
		for (i = 0; i < order; i++) {
			j = IndexMat[n][i];
			CountMat[i][j+1]++;
			tempMat[i][j+1]++;
		}
	}

	for (i = 0; i < order; i++) {
		for (j = 1; j < Dims[i] + 1; j++) {
			CountMat[i][j] += CountMat[i][j-1];
			tempMat[i][j] += tempMat[i][j-1];
		}
	}

	for (n = 0; n < nnz; n++) {
		for (i = 0; i < order; i++) {
			j = IndexMat[n][i];
			int now = tempMat[i][j];
			WhereMat[i][now] = n;
			tempMat[i][j]++;
		}
	}

	free(tempMat[0]); free(tempMat);
}

/* Complete all the internal structures of a subtensor */
void SubTensor::assignTensorIndex(int *gridIdx, int **gridDims) {

	int n, i, j;

	int cntSize = 0;
	for (i = 0; i < order; i++)
		cntSize += gridDims[i][gridIdx[i]] + 1;

	int **tempMat = (int **)malloc(sizeof(int *)*order);
	tempMat[0] = (int *)malloc(sizeof(int)*cntSize);
	for(i = 1; i < order; i++) tempMat[i] = tempMat[i-1] + gridDims[i-1][gridIdx[i-1]] + 1;

	memset(CountMat[0], 0, sizeof(int)*cntSize);
	memset(tempMat[0], 0, sizeof(int)*cntSize);

	for (n = 0; n < nnz; n++) {
		for (i = 0; i < order; i++) {
			j = IndexMat[n][i];
			CountMat[i][j+1]++;
			tempMat[i][j+1]++;
		}
	}

	for (i = 0; i < order; i++) {
		for (j = 1; j < gridDims[i][gridIdx[i]] + 1; j++) {
			CountMat[i][j] += CountMat[i][j-1];
			tempMat[i][j] += tempMat[i][j-1];
		}
	}

	for (n = 0; n < nnz; n++) {
		for (i = 0; i < order; i++) {
			j = IndexMat[n][i];
			int now = tempMat[i][j];
			WhereMat[i][now] = n;
			tempMat[i][j]++;
		}
	}

	free(tempMat[0]); free(tempMat);
}

/* Read the subtensor (i.e., X_g[g_1, g_2, ..., g_N]) from a grid-based tensor file */
void SubTensor::readSubTensorFromGTF(const int fd, int cellIdx, int *gridIdx, int **gridDims) {

	int gridSize, order, cntSize = 0;
	off_t gridOffset;
	off_t readSize;

	pread(fd, &gridSize, sizeof(int), 0);
	pread(fd, &order, sizeof(int), sizeof(int));
	pread(fd, &gridOffset, sizeof(off_t), sizeof(int)*2 + sizeof(int)*order*(gridSize+1) + sizeof(off_t)*cellIdx);

	for (int i = 0; i < order; i++)
		cntSize += gridDims[i][gridIdx[i]] + 1;

	lseek(fd, gridOffset, SEEK_SET);
	read(fd, &nnz, sizeof(int));

	ptr = (int *)malloc(sizeof(int)*(nnz*order*2 + cntSize));
	val = (double *)malloc(sizeof(double)*nnz);	

	IndexPtr = ptr;
	WherePtr = IndexPtr + nnz*order;
	CountPtr = WherePtr + nnz*order;

	IndexMat = (int **)malloc(sizeof(int *)*nnz);
	WhereMat = (int **)malloc(sizeof(int *)*order);
	CountMat = (int **)malloc(sizeof(int *)*order);
	
	for (int i = 0; i < nnz; i++) IndexMat[i] = &IndexPtr[i*order];
	for (int i = 0; i < order; i++) WhereMat[i] = &WherePtr[i*nnz];
	for (int i = 0; i < order; i++) {
		if (i == 0) CountMat[i] = CountPtr;
		else CountMat[i] = CountMat[i-1] + gridDims[i-1][gridIdx[i-1]] + 1;
	}

	read(fd, ptr, sizeof(int)*(nnz*order*2 + cntSize));
	read(fd, val, sizeof(double)*nnz);
}

/* Write the subtensor (i.e., X_g[g_1, g_2, ..., g_N]) to a grid-based tensor file */
off_t SubTensor::writeSubTensorToGTF(const int fd, int cellIdx, int *gridIdx, int **gridDims) {

	int gridSize, order, cntSize = 0;
	off_t gridOffset;
	off_t writeSize;

	pread(fd, &gridSize, sizeof(int), 0);
	pread(fd, &order, sizeof(int), sizeof(int));
	pread(fd, &gridOffset, sizeof(off_t), sizeof(int)*2 + sizeof(int)*order*(gridSize+1) + sizeof(off_t)*cellIdx);

	for (int i = 0; i < order; i++)
		cntSize += gridDims[i][gridIdx[i]] + 1;

	lseek(fd, gridOffset, SEEK_SET);

	write(fd, &nnz, sizeof(int));
	write(fd, ptr, sizeof(int)*(nnz*order*2 + cntSize));
	write(fd, val, sizeof(double)*nnz);

	writeSize = sizeof(int)*(1 + nnz*order*2 + cntSize) + sizeof(double)*nnz;
	return writeSize;
}

/* Build a grid-based tensor file */
void Build_GTF_From_Memory() {

	printf("Building a grid-based tensor file based on memory ... ");

	/* build an entire input tensor on memory */
	Tensor X;
	X.buildTensorFromRawFile(Option::trainPath);

	int i, j, n, g, cellIdx;
	int gridSize = Option::gridSize;
	int order = Option::tensorOrder;
	int nnz = X.nnz;

	int *gridIdx;
	int **gridDims, **gridDimCnt;

	int gridTensors_N = pow(gridSize, order);
	int gridEntries_N = nnz/gridSize + 1;


	/* array of sub-tensors (grid-based tensor) */
	SubTensor *GT = new SubTensor[gridTensors_N];

	gridIdx = (int *)malloc(sizeof(int)*order);
	gridDims = (int **)malloc(sizeof(int *)*order);
	gridDims[0] = (int *)malloc(sizeof(int)*order*gridSize);
	gridDimCnt = (int **)malloc(sizeof(int *)*order);
	gridDimCnt[0] = (int *)malloc(sizeof(int)*order*(gridSize+1));

	memset(gridIdx, 0, sizeof(int)*order);
	memset(gridDimCnt[0], 0, sizeof(int)*order*(gridSize+1));

	/* partition each dimensionality of a data tensor so that the number of entries in a single grid unit must be the same */
	for (i = 0; i < order; i++) {
		gridDims[i] = &gridDims[0][i*gridSize];
		gridDimCnt[i] = &gridDimCnt[0][i*(gridSize+1)];

		if (X.Dims[i] <= gridSize) {
			for (g = 0; g < X.Dims[i]; g++) gridDimCnt[i][g] = g;
			for (g = X.Dims[i]; g < gridSize+1; g++) gridDimCnt[i][g] = X.Dims[i];
		}
		else {
			for (j = 0; j < X.Dims[i]; j++) {
				if (X.CountMat[i][j] >= gridEntries_N*gridIdx[i])
					gridDimCnt[i][gridIdx[i]++] = j;
			}
			for (g = gridIdx[i]; g < gridSize+1; g++) gridDimCnt[i][g] = X.Dims[i];
		}

		for (g = 0; g < gridSize; g++) gridDims[i][g] = gridDimCnt[i][g+1] - gridDimCnt[i][g];
	}

	/* count the number of observed entries in each grid cell */
	for (n = 0; n < nnz; n++) {
		memset(gridIdx, 0, sizeof(int)*order);
		for (i = 0; i < order; i++) {
			int targetIndex = X.IndexMat[n][i];
			while(targetIndex >= gridDimCnt[i][gridIdx[i]+1]) gridIdx[i]++;
		}
		cellIdx = gridIdx2cellIdx(gridIdx);
		GT[cellIdx].nnz++;
	}

	/* allocate the space for each grid cell */
	for (cellIdx = 0; cellIdx < gridTensors_N; cellIdx++) {
		cellIdx2gridIdx(gridIdx, cellIdx);

		int matSize = GT[cellIdx].nnz*order;
		int cntSize = 0;
		for (i = 0; i < order; i++)
			cntSize += gridDims[i][gridIdx[i]] + 1;			 

		GT[cellIdx].order = order;		
		GT[cellIdx].ptr = (int *)malloc(sizeof(int)*(matSize*2 + cntSize));
		GT[cellIdx].val = (double *)malloc(sizeof(double)*GT[cellIdx].nnz);	

		GT[cellIdx].IndexPtr = GT[cellIdx].ptr;
		GT[cellIdx].WherePtr = GT[cellIdx].IndexPtr + matSize;
		GT[cellIdx].CountPtr = GT[cellIdx].WherePtr + matSize;

		GT[cellIdx].IndexMat = (int **)malloc(sizeof(int *)*GT[cellIdx].nnz);
		GT[cellIdx].WhereMat = (int **)malloc(sizeof(int *)*order);
		GT[cellIdx].CountMat = (int **)malloc(sizeof(int *)*order);

		for (i = 0; i < GT[cellIdx].nnz; i++)
			GT[cellIdx].IndexMat[i] = &GT[cellIdx].IndexPtr[i*order];
		for (i = 0; i < order; i++)
			GT[cellIdx].WhereMat[i] = &GT[cellIdx].WherePtr[i*GT[cellIdx].nnz];
		for (i = 0; i < order; i++) {
			if (i == 0) GT[cellIdx].CountMat[i] = GT[cellIdx].CountPtr;
			else GT[cellIdx].CountMat[i] = GT[cellIdx].CountMat[i-1] + gridDims[i-1][gridIdx[i-1]] + 1;
		}

		GT[cellIdx].nnz = 0;
	}

	/* complete the space of each grid cell */
	for (n = 0; n < nnz; n++) {
		memset(gridIdx, 0, sizeof(int)*order);
		for (i = 0; i < order; i++) {
			int targetIndex = X.IndexMat[n][i];
			while(targetIndex >= gridDimCnt[i][gridIdx[i]+1]) gridIdx[i]++;
		}

		cellIdx = gridIdx2cellIdx(gridIdx);			
		int pos = GT[cellIdx].nnz;
		memcpy(GT[cellIdx].IndexMat[pos], X.IndexMat[n], sizeof(int)*order);

		for (i = 0; i < order; i++)
			GT[cellIdx].IndexMat[pos][i] -= gridDimCnt[i][gridIdx[i]];

		GT[cellIdx].val[pos] = X.val[n];
		GT[cellIdx].nnz++;
	}

	for (cellIdx = 0; cellIdx < gridTensors_N; cellIdx++) {
		cellIdx2gridIdx(gridIdx, cellIdx);
		GT[cellIdx].assignTensorIndex(gridIdx, gridDims);		
	}

	/* write grid-based tensor file to disk */
	char filename[1024];
	sprintf(filename, "%s_GTF", Option::trainPath);
	int fd = open(filename, O_CREAT|O_RDWR, S_IRWXU);

	// meta information for a data tensor
	write(fd, &gridSize, sizeof(int));
	write(fd, &order, sizeof(int));
	write(fd, gridDimCnt[0], sizeof(int)*order*(gridSize+1));

	off_t gridTensorOffset = sizeof(int)*2 + sizeof(int)*order*(gridSize+1) + sizeof(off_t)*gridTensors_N;

	for (cellIdx = 0; cellIdx < gridTensors_N; cellIdx++) {
		cellIdx2gridIdx(gridIdx, cellIdx);
		pwrite(fd, &gridTensorOffset, sizeof(off_t), sizeof(int)*2 + sizeof(int)*order*(gridSize+1) + sizeof(off_t)*cellIdx);
		gridTensorOffset += GT[cellIdx].writeSubTensorToGTF(fd, cellIdx, gridIdx, gridDims);
	}
	close(fd);

	system("sync");
	system("echo 3 > /proc/sys/vm/drop_caches");

	/* free all local variables */
	delete []GT;
	free(gridIdx);
	free(gridDims[0]); free(gridDims);
	free(gridDimCnt[0]); free(gridDimCnt);

	printf("Done\n");
}

/* Build a grid-based tensor file using the disk (for big tensors which exceed the size of the memory) */
void Build_GTF_From_Disk() {

	printf("Building a grid-based tensor file based on disk ...\n");

	int i, j, g, k, idx, cellIdx;
	int gridSize = Option::gridSize;
	int order = Option::tensorOrder;
	long n, nnz = 0;
	double val;
	FILE *fp;
	char tmp[1005];

	/* pre-compute the gridPermu array */
	int **gridPermu;
	Precompute_Permutation(gridPermu);

	/* read an input file to obtain meta information (nnz, Dims, CountMat) */
	int *Dims = (int *)malloc(sizeof(int)*order);
	memset(Dims, 0, sizeof(int)*order);

	printf("\t1. Reading Input for nnz ... ");

	fp = fopen(Option::trainPath, "r");
	while (fgets(tmp, 1005, fp)) nnz++;
	fclose(fp);
	
	printf("Done \n");
	printf("\t2. Reading Input for Dims[] ... ");

	fp = fopen(Option::trainPath, "r");
	for (n = 0; n < nnz; n++) {
		for (i = 0; i < order; i++) {
			fscanf(fp, "%d", &idx);
			if (Dims[i] < idx) Dims[i] = idx;
		}
		fscanf(fp, "%lf", &val);
	}
	fclose(fp);

	int cntSize = 0;
	for (i = 0; i < order; i++)
		cntSize += Dims[i] + 1;

	long **CountMat;
	CountMat = (long **)malloc(sizeof(long*)*order);
	CountMat[0] = (long *)malloc(sizeof(long)*cntSize);
	memset(CountMat[0], 0, sizeof(long)*cntSize);
	for (i = 1; i < order; i++) CountMat[i] = CountMat[i-1] + Dims[i-1] + 1;
	
	printf("Done \n");
	printf("\t3. Reading Input for CountMat[] ... ");

	fp = fopen(Option::trainPath, "r");
	for (n = 0; n < nnz; n++) {
		for (i = 0; i < order; i++) {
			fscanf(fp, "%d", &idx);
			CountMat[i][idx]++;
		}
		fscanf(fp, "%lf", &val);
	}
	fclose(fp);

	for (i = 0; i < order; i++) {
		for (j = 1; j < Dims[i] + 1; j++) {
			CountMat[i][j] += CountMat[i][j-1];
		}
	}

	int *tempIdx, *gridIdx;
	int **gridDims, **gridDimCnt;
	int partTensors_N = pow(gridSize, order-1);
	int gridTensors_N = pow(gridSize, order);
	long gridEntries_N = nnz/gridSize + 1;

	int *nnzCnt = (int *)malloc(sizeof(int)*gridTensors_N);

	tempIdx = (int *)malloc(sizeof(int)*order);
	gridIdx = (int *)malloc(sizeof(int)*order);
	gridDims = (int **)malloc(sizeof(int *)*order);
	gridDims[0] = (int *)malloc(sizeof(int)*order*gridSize);
	gridDimCnt = (int **)malloc(sizeof(int *)*order);
	gridDimCnt[0] = (int *)malloc(sizeof(int)*order*(gridSize+1));

	memset(nnzCnt, 0, sizeof(int)*gridTensors_N);
	memset(gridIdx, 0, sizeof(int)*order);
	memset(gridDimCnt[0], 0, sizeof(int)*order*(gridSize+1));

	/* partition each dimensionality of a data tensor so that the number of entries in a single grid unit must be the same */
	for (i = 0; i < order; i++) {
		gridDims[i] = &gridDims[0][i*gridSize];
		gridDimCnt[i] = &gridDimCnt[0][i*(gridSize+1)];

		if (Dims[i] <= gridSize) {
			for (g = 0; g < Dims[i]; g++) gridDimCnt[i][g] = g;
			for (g = Dims[i]; g < gridSize+1; g++) gridDimCnt[i][g] = Dims[i];
		}
		else {
			for (j = 0; j < Dims[i]; j++) {
				if (CountMat[i][j] >= gridEntries_N*gridIdx[i])
					gridDimCnt[i][gridIdx[i]++] = j;
			}
			for (g = gridIdx[i]; g < gridSize+1; g++) gridDimCnt[i][g] = Dims[i];
		}

		for (g = 0; g < gridSize; g++) gridDims[i][g] = gridDimCnt[i][g+1] - gridDimCnt[i][g];
	}

	printf("Done \n");
	/* count the number of observed entries in each grid cell */
	printf("\t4. Reading Input for nnz of each cell ... ");

	fp = fopen(Option::trainPath, "r");
	for (n = 0; n < nnz; n++) {
		memset(gridIdx, 0, sizeof(int)*order);
		for (i = 0; i < order; i++) {
			fscanf(fp, "%d", &idx);
			while (idx-1 >= gridDimCnt[i][gridIdx[i]+1]) gridIdx[i]++;
		}
		fscanf(fp, "%lf", &val);
		cellIdx = gridIdx2cellIdx(gridIdx);
		nnzCnt[cellIdx]++;
	}
	fclose(fp);
	printf("Done \n");

	/* open a file and write meta-information  */
	char filename[1024];
	sprintf(filename, "%s_GTF", Option::trainPath);
	int fd = open(filename, O_CREAT|O_RDWR, S_IRWXU);

	write(fd, &gridSize, sizeof(int));
	write(fd, &order, sizeof(int));
	write(fd, gridDimCnt[0], sizeof(int)*order*(gridSize+1));

	off_t writeSize;
	off_t gridTensorOffset = sizeof(int)*2 + sizeof(int)*order*(gridSize+1) + sizeof(off_t)*gridTensors_N;
	for (cellIdx = 0; cellIdx < gridTensors_N; cellIdx++) {
		write(fd, &gridTensorOffset, sizeof(off_t));

		int cntSize = 0;
		cellIdx2gridIdx(gridIdx, cellIdx);
		for (i = 0; i < order; i++)
			cntSize += gridDims[i][gridIdx[i]] + 1;
		writeSize = sizeof(int)*(1 + nnzCnt[cellIdx]*order*2 + cntSize) + sizeof(double)*nnzCnt[cellIdx];
		gridTensorOffset += writeSize;
	}


	printf("\t5. Reading Input for "); 		
	SubTensor *GT = new SubTensor[gridTensors_N];

	/* build a part tensor X^(1)_g one at a time because of a memory limitation */
	for (g = 0; g < gridSize; g++) {

		printf("%d-th ", g);
		int gg;
		gridIdx[0] = g;

		/* allocate the space for each grid cell */
		for (gg = 0; gg < partTensors_N; gg++) {
			for (i = 0; i < order-1; i++)
				gridIdx[i+1] = gridPermu[gg][i];
			cellIdx = gridIdx2cellIdx(gridIdx);

			int matSize = nnzCnt[cellIdx]*order;
			int cntSize = 0;
			for (i = 0; i < order; i++)
				cntSize += gridDims[i][gridIdx[i]] + 1;			 

			GT[cellIdx].nnz = 0;
			GT[cellIdx].order = order;		
			GT[cellIdx].ptr = (int *)malloc(sizeof(int)*(matSize*2 + cntSize));
			GT[cellIdx].val = (double *)malloc(sizeof(double)*nnzCnt[cellIdx]);	

			GT[cellIdx].IndexPtr = GT[cellIdx].ptr;
			GT[cellIdx].WherePtr = GT[cellIdx].IndexPtr + matSize;
			GT[cellIdx].CountPtr = GT[cellIdx].WherePtr + matSize;

			GT[cellIdx].IndexMat = (int **)malloc(sizeof(int *)*nnzCnt[cellIdx]);
			GT[cellIdx].WhereMat = (int **)malloc(sizeof(int *)*order);
			GT[cellIdx].CountMat = (int **)malloc(sizeof(int *)*order);

			for (i = 0; i < nnzCnt[cellIdx]; i++)
				GT[cellIdx].IndexMat[i] = &GT[cellIdx].IndexPtr[i*order];
			for (i = 0; i < order; i++)
				GT[cellIdx].WhereMat[i] = &GT[cellIdx].WherePtr[i*nnzCnt[cellIdx]];
			for (i = 0; i < order; i++) {
				if (i == 0) GT[cellIdx].CountMat[i] = GT[cellIdx].CountPtr;
				else GT[cellIdx].CountMat[i] = GT[cellIdx].CountMat[i-1] + gridDims[i-1][gridIdx[i-1]] + 1;
			}
		}

		/* complete the space of each grid cell, and write it to disk */
		fp = fopen(Option::trainPath, "r");
		for (n = 0; n < nnz; n++) {
			memset(gridIdx, 0, sizeof(int)*order);
			for (i = 0; i < order; i++) {
				fscanf(fp, "%d", &idx);
				tempIdx[i] = idx-1;
				while (idx-1 >= gridDimCnt[i][gridIdx[i]+1]) gridIdx[i]++;
			}
			fscanf(fp, "%lf", &val);
			if(gridIdx[0] != g) continue;

			cellIdx = gridIdx2cellIdx(gridIdx);
			int pos = GT[cellIdx].nnz;
			memcpy(GT[cellIdx].IndexMat[pos], tempIdx, sizeof(int)*order);

			for (i = 0; i < order; i++)
				GT[cellIdx].IndexMat[pos][i] -= gridDimCnt[i][gridIdx[i]];

			GT[cellIdx].val[pos] = val;
			GT[cellIdx].nnz++;
			
		}		
		fclose(fp);

		gridIdx[0] = g;
		for (gg = 0; gg < partTensors_N; gg++) {
			for (i = 0; i < order-1; i++)
				gridIdx[i+1] = gridPermu[gg][i];
			cellIdx = gridIdx2cellIdx(gridIdx);
			GT[cellIdx].assignTensorIndex(gridIdx, gridDims);		
			GT[cellIdx].writeSubTensorToGTF(fd, cellIdx, gridIdx, gridDims);
		}
	}
	close(fd);

	printf("cell ... Done\n");

	system("sync");
	system("echo 3 > /proc/sys/vm/drop_caches");

	/* free all local variables */
	delete []GT;
	free(tempIdx); free(gridIdx); free(nnzCnt);
	free(gridPermu[0]); free(gridPermu);
	free(gridDims[0]); free(gridDims);
	free(gridDimCnt[0]); free(gridDimCnt);
}

/* Read meta-information from a grid-based tensor file */
void Read_Grid_Info(char *InputPath, int **&gridDimCnt, int **&gridDims) {

	int i, g;
	int gridSize = Option::gridSize;
	int order = Option::tensorOrder;

	char filename[1024];
	sprintf(filename, "%s_GTF", InputPath);
	int fd = open(filename, O_RDWR, S_IRWXU);

	gridDimCnt = (int **)malloc(sizeof(int *)*order);
	gridDimCnt[0] = (int *)malloc(sizeof(int)*order*(gridSize+1));
	gridDims = (int **)malloc(sizeof(int *)*order);
	gridDims[0] = (int *)malloc(sizeof(int)*order*gridSize);

	pread(fd, gridDimCnt[0], sizeof(int)*order*(gridSize+1), sizeof(int)*2);
	close(fd);

	for (i = 0; i < order; i++) {
		gridDims[i] = &gridDims[0][i*gridSize];
		gridDimCnt[i] = &gridDimCnt[0][i*(gridSize+1)];
		for (g = 0; g < gridSize; g++)
			gridDims[i][g] = gridDimCnt[i][g+1] - gridDimCnt[i][g];
	} 
}
