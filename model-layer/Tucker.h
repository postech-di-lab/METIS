
#ifndef TUCKER_H
#define TUCKER_H

#include "Tensor.h"

/* MAX_KEY_ELEM has to be defined at least the order of an input tensor */
#define MAX_KEY_ELEM 4
typedef std::array<int, MAX_KEY_ELEM> DeltaIdx;

void Compute_Delta_Vector(double *result, int *alpha, int targetOrder, double ***FactorMat, Tensor &G);
void Load_Part_SubTensors(const char *filename, int partTensors_N, SubTensor *partTensor, int *cellIdxArray, int targetOrder, int targetIdx, int **gridDims, int **gridPermu);
void Update_Factor_Matrices_On_Disk(double ***FactorMat, Tensor &G, int **gridDims, int **gridPermu);
void Reconstruct_On_Disk(double ***FactorMat, Tensor &G, int **gridDimCnt, int **gridDims);
double Evaluate(double ***FactorMat, Tensor &G, Tensor &Xtest);
void Orthogonalize(double ***FactorMat, Tensor &G, int *Dims);
void FTcom(double ***FactorMat, Tensor &G, int *Dims, Tensor &Xtest);

#endif /* TUCKER_H */
