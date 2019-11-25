#include "util.h"
#include "ocam.h"
#define coarsegrained dynamic,1
#define finegrained dynamic,500

double Tstart = 0, Ttime = 0;

/*
	Update a single row
*/
inline void SingleRowUpdate(csc_idx **partIdx, RVALTYPE **partVal, const int gridSize, const int j, const dmat_t H, dvec_t wj, const RVALTYPE lambda, int dims) {
	long nnz = 0;
	for(int i=0; i<gridSize; i++)
		nnz += partIdx[i]->row_ptr[j+1] - partIdx[i]->row_ptr[j];		
	if(!nnz) return;

	dmat_t subH;
	alloc_dmat_t(subH, dims, nnz);
	long vBegin, vEnd, vi; 	
	long idxCol;
	int locCid = 0, i = 0, t = 0;	

	for(i=0; i<gridSize; i++) {
		vBegin = partIdx[i]->row_ptr[j];
		vEnd = partIdx[i]->row_ptr[j+1];
		for(vi=vBegin; vi<vEnd; vi++) {
			idxCol = partIdx[i]->col_offset + partIdx[i]->col_idx[vi];
			for(t=0; t<dims; t++) 
				subH[t][locCid] = H[idxCol][t];
			locCid++;
		}
	}

	for(t=0; t<dims; t++){
		RVALTYPE g=0, h=lambda*nnz;
		RVALTYPE oldwj = wj[t];
		RVALTYPE *ptrVal;

		locCid = 0;
		RVALTYPE locHVal;
		for(i=0; i<gridSize; i++) {
			vBegin = partIdx[i]->row_ptr[j];
			vEnd = partIdx[i]->row_ptr[j+1];
			ptrVal = partVal[i];
			for(vi=vBegin; vi<vEnd; vi++){
				locHVal = subH[t][locCid++];
				g += locHVal*(ptrVal[vi] + locHVal*oldwj);
				h += locHVal*locHVal;
			}
		}

		RVALTYPE newwj = g/h, tmp = 0, delta = 0;
		if(false && newwj < 0) {
			newwj = 0;
			delta = oldwj; // old - new
		} else {
			delta = oldwj - newwj;
		}

		locCid = 0;
		for(i=0; i<gridSize; i++) {
			vBegin = partIdx[i]->row_ptr[j];
			vEnd = partIdx[i]->row_ptr[j+1];
			ptrVal = partVal[i];
			for(vi=vBegin; vi<vEnd; vi++) {
				ptrVal[vi] += delta*subH[t][locCid++];
			}
		}
		wj[t] = newwj;
	}

	free_dmat_t(subH);
}

/*
	Update a single column 
*/
inline void SingleColumnUpdate(csc_idx **partIdx, RVALTYPE **partVal, const int gridSize, const int j, const dmat_t W, dvec_t hj, const RVALTYPE lambda, int dims) {
	long nnz = 0;
	for(int i=0; i<gridSize; i++)
		nnz += partIdx[i]->col_ptr[j+1] - partIdx[i]->col_ptr[j];		
	if(!nnz) return;

	dmat_t subW;
	alloc_dmat_t(subW, dims, nnz);
	long vBegin, vEnd, vi;	
	long idxRow;
	int locRid = 0, i = 0, t = 0;

	for(i=0; i<gridSize; i++) {
		vBegin = partIdx[i]->col_ptr[j];
		vEnd = partIdx[i]->col_ptr[j+1];
		for(vi=vBegin; vi<vEnd; vi++) {
			idxRow = partIdx[i]->row_offset + partIdx[i]->row_idx[vi];
			for(int t=0; t<dims; t++) 
				subW[t][locRid] = W[idxRow][t];
			locRid++;
		}
	}

	for(t=0; t<dims; t++) {
		RVALTYPE g=0, h=lambda*nnz;
		RVALTYPE oldhj = hj[t];
		RVALTYPE *ptrVal;

		locRid = 0;
		RVALTYPE locWVal;
		for(i=0; i<gridSize; i++) {
			vBegin = partIdx[i]->col_ptr[j];
			vEnd = partIdx[i]->col_ptr[j+1];
			ptrVal = partVal[i];
			for(vi=vBegin; vi<vEnd; vi++) {
				locWVal = subW[t][locRid++];
				g += locWVal*(ptrVal[vi] + locWVal*oldhj);
				h += locWVal*locWVal;
			}
		}

		RVALTYPE newhj = g/h, tmp = 0, delta = 0;
		if(false && newhj < 0) {
			newhj = 0;
			delta = oldhj;
		} else {
			delta = oldhj - newhj;
		}

		locRid = 0;
		for(i=0; i<gridSize; i++) {
			vBegin = partIdx[i]->col_ptr[j];
			vEnd = partIdx[i]->col_ptr[j+1];
			ptrVal = partVal[i];
			for(vi=vBegin; vi<vEnd; vi++) {
				ptrVal[vi] += delta*subW[t][locRid++];
			}
		}
		hj[t] = newhj;
	}

	free_dmat_t(subW);
}


/*
	Update a single grid row or grid column 
*/
inline void PartitionUpdate(const char *gcsc_filename, const int gridSize, const int idx, dmat_t W, dmat_t H, const RVALTYPE lambda, int dims, bool isCol) {

	csc_idx **partIdx = MALLOC(csc_idx*, gridSize);
	RVALTYPE **partVal = MALLOC(RVALTYPE*, gridSize);

	int fd = open(gcsc_filename, O_RDWR, S_IRWXU);
	int fileGridSize;
	read(fd, &fileGridSize, sizeof(int));
	assert(gridSize == fileGridSize);	

	// Grid row case	
	if(!isCol) {
		RVALTYPE **transVal = MALLOC(RVALTYPE*, gridSize);
		for(int i=0; i<gridSize; i++) {
			partIdx[i] = new csc_idx();
			partVal[i] = partIdx[i]->readGCSCFromFile(fd, idx + i*gridSize);
			transVal[i] = partIdx[i]->transC2R(partVal[i]);
		}

		long row_offset = partIdx[0]->row_offset;
		long rows = partIdx[0]->rows;

		for(long r=0; r<rows; r++)
			SingleRowUpdate(partIdx, transVal, gridSize, r, H, W[row_offset+r], lambda, dims);
			
		for(int i=0; i<gridSize; i++) {
			partIdx[i]->transR2C(transVal[i], partVal[i]);
			partIdx[i]->writeGValToFile(fd, idx + i*gridSize, partVal[i]);

			free(partIdx[i]->ptr);
			delete partIdx[i];
			free(partVal[i]);
		}
	}

	// Grid column case
	else {
		for(int i=0; i<gridSize; i++) {
			partIdx[i] = new csc_idx();
			partVal[i] = partIdx[i]->readGCSCFromFile(fd, i + idx*gridSize);
		}

		long col_offset = partIdx[0]->col_offset;
		long cols = partIdx[0]->cols;

		for(long c=0; c<cols; c++)
			SingleColumnUpdate(partIdx, partVal, gridSize, c, W, H[col_offset+c], lambda, dims);

		for(int i=0; i<gridSize; i++) {
			partIdx[i]->writeGValToFile(fd, i + idx*gridSize, partVal[i]);

			free(partIdx[i]->ptr);
			delete partIdx[i];
			free(partVal[i]);	
		}
	}

	free(partIdx);
	free(partVal);
	close(fd);
}

/*
	Out-of-core Cyclic Coordinate Descent (CCD) for matrix completion 
*/
void ocam(const char *gcsc_filename, dmat_t W, dmat_t H, testset_t &T, parameter &param){

	const bool isRow = true;

	int k = param.k;
	int maxiter = param.maxiter;
	int gridSize = param.grids;
	int num_threads_old = omp_get_num_threads();
	RVALTYPE lambda = param.lambda;
	long num_updates = 0;

	int dims = k;

	omp_set_num_threads(param.threads);

	for(int oiter = 1; oiter <= maxiter; oiter++) {

		Tstart = omp_get_wtime();

		if(param.do_opt) {
			if(oiter <= 3)	dims = (int)k/8;
			else	dims = k;
		}

		// Update the factor matrix H
#pragma omp parallel for schedule(coarsegrained) shared(W,H)
		for(int i=0; i<gridSize; i++) {
			PartitionUpdate(gcsc_filename, gridSize, i, W, H, lambda, dims, isRow);

			system("sync");
			system("echo 3 > /proc/sys/vm/drop_caches");
		}

		// Update the factor matrix W
#pragma omp parallel for schedule(coarsegrained) shared(W,H)
		for(int i=0; i<gridSize; i++) {
			PartitionUpdate(gcsc_filename, gridSize, i, W, H, lambda, dims, !isRow);

			system("sync");
			system("echo 3 > /proc/sys/vm/drop_caches");
		}
		
		Ttime += omp_get_wtime() - Tstart;

		printf("%d\t%.10g\t", oiter, Ttime);
		if(T.nnz != 0){
			printf("%.10g", calrmse(T, W, H, k, false));
		}
		puts("");
		fflush(stdout);
	}

	omp_set_num_threads(num_threads_old);
}

