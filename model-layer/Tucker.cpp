#include "util.h"
#include "Option.h"
#include "Tensor.h"
#include "Tucker.h"

double Fit = -1;
double Error, RMSE, TestRMSE;
spinLock mapLock;

#define ARMA_USE_LAPACK
#define ARMA_USE_BLAS

using namespace std;
using namespace arma;

/* Compute the delta vector (i.e., \delta_(alpha)^(targetOrder)) and store it in result */
void Compute_Delta_Vector(double *result, int *alpha, int targetOrder, double ***FactorMat, Tensor &G) {
	int i, l;
	int order = Option::tensorOrder;
	int CorePos, mulrow, mulcol;
	double res;
	for (l = 0; l < G.nnz; l++) {
		CorePos = G.IndexMat[l][targetOrder];
		res = G.val[l];
		for (i = 0; i < order; i++) {
			if (i != targetOrder) {
				mulrow = alpha[i];
				mulcol = G.IndexMat[l][i];
				res *= FactorMat[i][mulrow][mulcol];
			}
		}
		result[CorePos] += res;
	}
}

/* Read the partition of the data tensor (i.e., X_g^(targetOrder)[targetIndex]) from GTF */
void Load_Part_SubTensors(const char *filename, int partTensors_N, SubTensor *partTensor, int *cellIdxArray, int targetOrder, int targetIdx, int **gridDims, int **gridPermu) {
	int g, order = Option::tensorOrder;

#pragma omp parallel for schedule(static)
	for (g = 0; g < partTensors_N; g++) {
		int i, cellIdx;
		int *gridIdx = (int *)malloc(sizeof(int)*order);

		gridIdx[targetOrder] = targetIdx;
		for (i = 0; i < order-1; i++) {
			if (i < targetOrder) gridIdx[i] = gridPermu[g][i];
			else gridIdx[i+1] = gridPermu[g][i];
		}	
		cellIdx = gridIdx2cellIdx(gridIdx);
		cellIdxArray[g] = cellIdx;

		int fd = open(filename, O_RDWR, S_IRWXU);
		partTensor[g].readSubTensorFromGTF(fd, cellIdx, gridIdx, gridDims);		
		close(fd);

		free(gridIdx);
	}
}

/* Update all factor matrices using GTF */
void Update_Factor_Matrices_On_Disk(double ***FactorMat, Tensor &G, int **gridDimCnt, int **gridDims, int **gridPermu) {

	omp_set_num_threads(Option::threadSize);

	int i, j, g;
	int order = Option::tensorOrder;
	int cacheSize = Option::cacheSize;
	int gridSize = Option::gridSize;
	double **DeltaMat;
	double lambda = Option::lambda;
	int partTensors_N = pow(gridSize, order-1);

	char filename[1024];
	sprintf(filename, "%s_GTF", Option::trainPath);

	// update the i-th factor matrix
	for (i = 0; i < order; i++) {
		double start = omp_get_wtime();
		int column_size = G.Dims[i];

		// initialize the cache table (=DeltaMat) and the mapping table (=IdxMap)
		Alloc_DeltaMat(DeltaMat, cacheSize, G.Dims[i]);
		map<DeltaIdx, int> IdxMap;
		int curloc = 0;

		int **ValidTable = (int **)malloc(sizeof(int *)*cacheSize);
		ValidTable[0] = (int *)malloc(sizeof(int)*cacheSize*order);
		for (j = 0; j < cacheSize; j++) ValidTable[j] = &ValidTable[0][j*order];
		memset(ValidTable[0], -1, sizeof(int)*cacheSize*order);		

		// update the g-th part of the i-th factor matrix
		for (g = 0; g < gridSize; g++) {
			SubTensor *partTensor = new SubTensor[partTensors_N];
			int *cellIdxArray = (int *)malloc(sizeof(int)*partTensors_N);
			Load_Part_SubTensors(filename, partTensors_N, partTensor, cellIdxArray, i, g, gridDims, gridPermu);

			int rows = gridDims[i][g];
			int rowOffset = gridDimCnt[i][g];
			if (rows == 0) continue;

#pragma omp parallel for schedule(dynamic)
			for (j = 0; j < rows; j++) {
				int k, t, l, ii, jj, gg;
				int pos, current_input_entry, cellnnz, tempnnz = 0, nnz = 0;
				int targetRow = j + gridDimCnt[i][g];

				for (gg = 0; gg < partTensors_N; gg++)
					nnz += partTensor[gg].CountMat[i][j+1] - partTensor[gg].CountMat[i][j];
				if (nnz == 0) continue;

				map<DeltaIdx, int>::iterator it;

				int *alpha = (int *)malloc(sizeof(int)*order);
				int *gridIdx = (int *)malloc(sizeof(int)*order);

				double *DeltaArray = (double *)malloc(sizeof(double)*nnz*column_size);
				memset(DeltaArray, 0, sizeof(double)*nnz*column_size);

				for (gg = 0; gg < partTensors_N; gg++) {
					pos = partTensor[gg].CountMat[i][j];
					cellnnz = partTensor[gg].CountMat[i][j+1] - partTensor[gg].CountMat[i][j];
					cellIdx2gridIdx(gridIdx, cellIdxArray[gg]);

					for (k = 0; k < cellnnz; k++) {
						current_input_entry = partTensor[gg].WhereMat[i][pos+k];
						memcpy(alpha, partTensor[gg].IndexMat[current_input_entry], sizeof(int)*order);
						for (ii = 0; ii < order; ii++) alpha[ii] += gridDimCnt[ii][gridIdx[ii]];

						/* you can ``explicitly'' specify the mode where the caching algorithm is not helpful */
						/* if you uncomment the following code, each delta vector is always computed without being stored in the cache table*/
//						if (i == specific mode) {
//							Compute_Delta_Vector(&DeltaArray[(tempnnz+k)*column_size], alpha, i, FactorMat, G);
//						} else {

						/* check if the corresponding delta vector has been already computed before */
						DeltaIdx gamma;
						memset(&gamma[0], 0, sizeof(int)*MAX_KEY_ELEM);
						memcpy(&gamma[0], alpha, sizeof(int)*order);
						gamma[i] = 1;

						acquireSpinLock(mapLock);
						it = IdxMap.find(gamma);
						releaseSpinLock(mapLock);

						/* if there exists the delta vector in the cache table, */
						if (it != IdxMap.end()) {
							int loc = it->second;
							memcpy(&DeltaArray[(tempnnz+k)*column_size], DeltaMat[loc], sizeof(double)*column_size);
						}
						/* otherwise, */
						else {
							Compute_Delta_Vector(&DeltaArray[(tempnnz+k)*column_size], alpha, i, FactorMat, G);
	
							/* store the computed delta vector in the cache table, and update the idxmap (:gamma -> assignedID) */
							acquireSpinLock(mapLock);
							it = IdxMap.find(gamma);

							if (it == IdxMap.end()) {
								/* if we have to evict the existing vector at assignedID, */
								if (ValidTable[curloc][0] != -1) {
									DeltaIdx tmpgamma;
									memset(&tmpgamma[0], 0, sizeof(int)*MAX_KEY_ELEM);
									memcpy(&tmpgamma[0], ValidTable[curloc], sizeof(int)*order);
									IdxMap.erase(tmpgamma);
								}
								memcpy(ValidTable[curloc], &gamma[0], sizeof(int)*order);

								IdxMap[gamma] = curloc;
								memcpy(DeltaMat[curloc], &DeltaArray[(tempnnz+k)*column_size], sizeof(double)*column_size);
								if (++curloc == cacheSize) curloc = 0;
							}
							releaseSpinLock(mapLock);
						}
//						}
					}
					tempnnz += cellnnz;
				}

				for (t = 0; t < column_size; t++) {
					double g = 0, h = lambda*nnz;
					tempnnz = 0;

					for (gg = 0; gg < partTensors_N; gg++) {
						pos = partTensor[gg].CountMat[i][j];
						cellnnz = partTensor[gg].CountMat[i][j+1] - partTensor[gg].CountMat[i][j];

						for (k = 0; k < cellnnz; k++) { 
							current_input_entry = partTensor[gg].WhereMat[i][pos+k];
							double Residual = partTensor[gg].val[current_input_entry];

							for (l = 0; l < column_size; l++) {
								if (l != t) Residual -= DeltaArray[(tempnnz+k)*column_size + l] * FactorMat[i][targetRow][l];
							}

							double Delta = DeltaArray[(tempnnz+k)*column_size + t];
							g += Delta * Residual;
							h += Delta * Delta; 
						}
						tempnnz += cellnnz;
					}

					if (Option::solverType == NN_SOLVER) {
						if (g/h < 0) FactorMat[i][targetRow][t] = 0;
						else FactorMat[i][targetRow][t] = g/h;
					}
					else
						FactorMat[i][targetRow][t] = g/h;
				}

				free(alpha); free(gridIdx); free(DeltaArray);
			}

			delete[] partTensor;
			free(cellIdxArray);
		}

		Free_DeltaMat(DeltaMat);
		IdxMap.clear();

		free(ValidTable[0]); free(ValidTable);

//		system("sync");
//		system("echo 3 > /proc/sys/vm/drop_caches");
	}
}

/* Calculate 1) the fit and 2) the training RMSE by reconstructing the original tensor */
void Reconstruct_On_Disk(double ***FactorMat, Tensor &G, int **gridDimCnt, int **gridDims) {
	int order = Option::tensorOrder;
	int gridSize = Option::gridSize;
	int cellIdx, gridTensors_N = pow(gridSize, order);
	long nnz = 0;
	double Norm = 0;
	RMSE = Error = 0;

	char filename[1024];
	sprintf(filename, "%s_GTF", Option::trainPath);

	for (cellIdx = 0; cellIdx < gridTensors_N; cellIdx++) {
		int n;
		int fd = open(filename, O_RDWR, S_IRWXU);
		int *gridIdx = (int *)malloc(sizeof(int)*order);

		SubTensor GT;
		cellIdx2gridIdx(gridIdx, cellIdx);
		GT.readSubTensorFromGTF(fd, cellIdx, gridIdx, gridDims);
		close(fd);

		nnz += GT.nnz;
#pragma omp parallel for schedule(static) reduction(+:Error,Norm,nnz)
		for (n = 0; n < GT.nnz; n++) {
			int i, l;
			int mulrow, mulcol;
			double ans = 0;
			int *alpha = (int *)malloc(sizeof(int)*order);

			memcpy(alpha, GT.IndexMat[n], sizeof(int)*order);

			for (l = 0; l < G.nnz; l++) {
				double temp = G.val[l];
				for (i = 0; i < order; i++) {
					mulrow = alpha[i] + gridDimCnt[i][gridIdx[i]]; 
					mulcol = G.IndexMat[l][i];
					temp *= FactorMat[i][mulrow][mulcol];
				}
				ans += temp;
			}

			Norm += pow(GT.val[n], 2);
			Error += pow(GT.val[n] - ans, 2);
			free(alpha);
		}
		free(gridIdx);
	}

	RMSE = sqrt(Error/nnz);
	if (Norm == 0) Fit = 1;
	else Fit = 1 - sqrt(Error)/sqrt(Norm);
}

/* Calculate the test RMSE using a tensor with test entries (=Xtest) */
double Evaluate(double ***FactorMat, Tensor &G, Tensor &Xtest) {
	int n, order = Option::tensorOrder;
	double val, RMSE = 0;

#pragma omp parallel for schedule(static) reduction(+:RMSE)
	for (n = 0; n < Xtest.nnz; n++) {
		int ii, l, mulrow, mulcol;
		double ans = 0;

		int *alpha = (int *)malloc(sizeof(int)*order);
		memcpy(alpha, Xtest.IndexMat[n], sizeof(int)*order);

		for (l = 0; l < G.nnz; l++) {
			double temp = G.val[l];
			for (ii = 0; ii < order; ii++) {
				mulrow = alpha[ii]; 
				mulcol = G.IndexMat[l][ii];
				temp *= FactorMat[ii][mulrow][mulcol];
			}
			ans += temp;
		}

		RMSE += pow(Xtest.val[n] - ans, 2);
		free(alpha);
	}

	RMSE = sqrt(RMSE/Xtest.nnz);
	return RMSE;
}

/* Orthogonalize all the factor matrices and update the core tensor */
void Orthogonalize(double ***FactorMat, Tensor &G, int *Dims) {

	int i, j, k, l;
	int order = Option::tensorOrder;

	int *Mul = (int *)malloc(sizeof(int)*order);
	Mul[order - 1] = 1;
	for (i = order - 2; i >= 0; i--) {
		Mul[i] = Mul[i + 1] * G.Dims[i + 1];
	}
	int pos = 0;
	for (i = 0; i < order; i++) {
		mat Q, R;
		mat P = mat(Dims[i], G.Dims[i]);
		for (k = 0; k < Dims[i]; k++) {
			for (l = 0; l < G.Dims[i]; l++) {
				P(k, l) = FactorMat[i][k][l];
			}
		}
		qr_econ(Q, R, P);
		for (k = 0; k < Dims[i]; k++) {
			for (l = 0; l < G.Dims[i]; l++) {
				FactorMat[i][k][l] = Q(k, l);
			}
		}

		double *tempCore = (double *)malloc(sizeof(double)*G.nnz);
		int *tempPermu = (int *)malloc(sizeof(int)*order);
		memset(tempCore, 0, sizeof(double)*G.nnz);

		for (j = 0; j < G.nnz; j++) {
			for (k = 0; k <= i - 1; k++) {
				tempPermu[k] = G.IndexMat[j][k];
			}
			for (k = i + 1; k < order; k++) {
				tempPermu[k] = G.IndexMat[j][k];
			}
			for (k = 0; k < G.Dims[i]; k++) {
				tempPermu[i] = k;
				int cur = j + (k - G.IndexMat[j][i])*Mul[i];
				tempCore[cur] += G.val[j] * R(k, G.IndexMat[j][i]);
			}
		}
		for (j = 0; j < G.nnz; j++) {
			G.val[j] = tempCore[j];
		}
		free(tempCore); free(tempPermu);
	}
	free(Mul);
}

/* The main process of Tucker factorization */
void FTcom(double ***FactorMat, Tensor &G, int *Dims, Tensor &Xtest) {

	int iter = 0;
	double factorTime = 0, evalTime = 0;
	int **gridPermu, **gridDimCnt, **gridDims;

	Precompute_Permutation(gridPermu);
	Read_Grid_Info(Option::trainPath, gridDimCnt, gridDims);

	for (iter = 0; iter < Option::iterationSize; iter++) {
		double startTime = omp_get_wtime();

		Update_Factor_Matrices_On_Disk(FactorMat, G, gridDimCnt, gridDims, gridPermu);
		factorTime = omp_get_wtime() - startTime;

		Reconstruct_On_Disk(FactorMat, G, gridDimCnt, gridDims);
		if (Option::testPath != NULL)
			TestRMSE = Evaluate(FactorMat, G, Xtest);
		evalTime = omp_get_wtime() - startTime - factorTime;

		if (Option::testPath == NULL)
			printf("Iter: %d, Fit: %lf, Training RMSE: %lf, Training Time: %lf, Reconst. Time: %lf\n", iter+1, Fit, RMSE, factorTime, evalTime);
		else
			printf("Iter: %d, Fit: %lf, Training RMSE: %lf, Test RMSE: %lf, Training Time: %lf, Reconst.&Test Time: %lf\n", iter+1, Fit, RMSE, TestRMSE, factorTime, evalTime);
	}

	printf("\nOrthogonalizing all factor matrices and updating the core tensor ... ");
	Orthogonalize(FactorMat, G, Dims);
	printf("Done\n");

	free(gridPermu[0]); free(gridPermu);
	free(gridDimCnt[0]); free(gridDimCnt);
	free(gridDims[0]); free(gridDims);
}
