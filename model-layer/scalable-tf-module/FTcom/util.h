#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <map>
#include <atomic>
#include <armadillo>
#include <omp.h>

#include "Tensor.h"

typedef std::atomic_flag spinLock;

void Initialize(double ***&FactorMat, Tensor &G, int *&Dims, Tensor &Xtest);
void Print(double ***FactorMat, Tensor &G, int *Dims);

void Alloc_DeltaMat(double **&DeltaMat, int tableSize, int CoreDim);
void Free_DeltaMat(double **DeltaMat);

void Precompute_Permutation(int **&gridPermu);
int gridIdx2cellIdx(int *gridIdx);
void cellIdx2gridIdx(int *gridIdx, int cellIdx);

void acquireSpinLock(spinLock &lock);
void releaseSpinLock(spinLock &lock);

double frand(double x, double y);
double abss(double x);

#endif /* UTIL_H */
