#include "util.h"
#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))

void alloc_dmat_t(dmat_t &X, long n, long k) {
	X = MALLOC(RVALTYPE*, n);
	X[0] = MALLOC(RVALTYPE, n*k);
	for(long i = 0; i < n; i++)
		X[i] = &X[0][i*k];
}

void free_dmat_t(dmat_t &X) {
	free(X[0]);
	free(X);
}

void save_dmat_t(FILE *fp, dmat_t A, long n, long k){
	if (fp == NULL) 
		fprintf(stderr, "output stream is not valid.\n");

	fwrite(&n, sizeof(long), 1, fp);
	fwrite(&k, sizeof(long), 1, fp);

	for(long i=0; i<n; ++i)
		fwrite(A[i], sizeof(RVALTYPE), k, fp);
}

void initial_dmat_t(dmat_t &X, long n, long k, bool initZero) {
	alloc_dmat_t(X, n, k);
	srand48(0L);
	if(initZero) {
		for(long i = 0; i < n; ++i)
			for(long j = 0; j < k; ++j)
				X[i][j] = 0;
	}
	else {
		for(long i = 0; i < n; ++i)
			for(long j = 0; j < k; ++j)
				X[i][j] = 0.1*drand48();
	}
}

double dot(const dvec_t a, const dvec_t b, long k){
	double ret = 0;
	for(int i = k-1; i >=0; --i)
		ret+=a[i]*b[i];
	return ret;
}
double dot(const dmat_t W, const int i, const dmat_t H, const int j, long k){
	double ret = 0;
	for(int t = 0; t < k; ++t)
		ret+=W[t][i]*H[t][j];
	return ret;
}
double dot(const dmat_t W, const int i, const dvec_t Hj, long k){
	double ret = 0;
	for(int t = 0; t < k; ++t)
		ret+=W[t][i]*Hj[t];
	return ret;
}
double norm(const dvec_t a, long k){
	double ret = 0;
	for(int i = k-1; i >=0; --i)
		ret+=a[i]*a[i];
	return ret;
}
double norm(const dmat_t M, long n, long k) {
	double reg = 0;
	for(int i = n-1; i>=0; --i) reg += norm(M[i], k);
	return reg;
}

double calrmse(testset_t &testset, const dmat_t W, const dmat_t H, long k, bool iscol){
	size_t nnz = testset.nnz;
	double rmse = 0, err;
#pragma omp parallel for private(err) reduction(+:rmse)
	for(size_t idx = 0; idx < nnz; ++idx){
		err = -testset[idx].v;
		if(iscol)
			err += dot(W, testset[idx].i, H, testset[idx].j, k);
		else 
			err += dot(W[testset[idx].i], H[testset[idx].j], k);
		rmse += err*err;
	}
	return sqrt(rmse/nnz);
}

