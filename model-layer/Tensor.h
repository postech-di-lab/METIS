
#ifndef TENSOR_H
#define TENSOR_H

class Tensor {
public:
	int order, nnz;

	int *ptr;
	double *val;

	int *IndexPtr, **IndexMat;
	int *WherePtr, **WhereMat;
	int *CountPtr, **CountMat;
	int *Dims;

	Tensor(): order(0), nnz(0), ptr(NULL), val(NULL), IndexMat(NULL), WhereMat(NULL), CountMat(NULL), Dims(NULL) { }
	Tensor(const Tensor& st) { *this = st; }
	~Tensor() {
		if(ptr) { free(ptr); ptr = NULL; } 
		if(val) { free(val); val = NULL; }
		if(IndexMat) { free(IndexMat); IndexMat = NULL; }
		if(WhereMat) { free(WhereMat); WhereMat = NULL; }
		if(CountMat) { free(CountMat); CountMat = NULL; }
		if(Dims) { free(Dims); Dims = NULL; }
	}

	void buildTensorFromRawFile(const char *InputPath);
	void buildCoreTensor(int CoreDim);
	void assignTensorIndex(); 
};

class SubTensor : public Tensor {
public:
	SubTensor() { }
	SubTensor(const SubTensor& st) { *this = st; }
	~SubTensor() { }

	void assignTensorIndex(int *gridIdx, int **gridDims);
	void readSubTensorFromGTF(const int fd, int cellIdx, int *gridIdx, int **gridDims);
	off_t writeSubTensorToGTF(const int fd, int cellIdx, int *gridIdx, int **gridDims);
};

void Build_GTF_From_Memory();
void Build_GTF_From_Disk();
void Read_Grid_Info(char *InputPath, int **&gridDimCnt, int **&gridDims);

#endif /* TENSOR_H  */
