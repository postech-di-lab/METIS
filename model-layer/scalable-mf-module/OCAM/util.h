#ifndef MATUTIL
#define MATUTIL

#include "macros.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <set>
#include <vector>
#include <cmath>
#include <omp.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))

enum {ROWMAJOR, COLMAJOR};

using namespace std;
class rate_t;
class smat_t;
class testset_t;
typedef RVALTYPE* dvec_t;
typedef RVALTYPE** dmat_t;

void alloc_dmat_t(dmat_t &X, long n, long k);
void free_dmat_t(dmat_t &X);
void initial_dmat_t(dmat_t &X, long n, long k, bool initZero);
void save_dmat_t(FILE *fp, dmat_t A, long n, long k);

double dot(const dvec_t a, const dvec_t b, long k);
double dot(const dmat_t W, const int i, const dmat_t H, const int j, long k);
double dot(const dmat_t W, const int i, const dvec_t Hj, long k);
double norm(const dvec_t a, long k);
double norm(const dmat_t M, long n, long k);
double calrmse(testset_t &testset, const dmat_t W, const dmat_t H, long k, bool iscol=false);

class rate_t{
	public:
		int i, j; double v;
		rate_t(int ii=0, int jj=0, double vv=0): i(ii), j(jj), v(vv){}
};

class entry_iterator_t{
	private:
		FILE *fp;
		char buf[1000];
	public:
		size_t nnz;
		entry_iterator_t():nnz(0),fp(NULL){}
		entry_iterator_t(size_t nnz_, const char* filename) {
			nnz = nnz_;
			fp = fopen(filename,"r");
		}
		size_t size() {return nnz;}
		virtual rate_t next() {
			int i = 1, j = 1;
			double v = 0;
			if (nnz > 0) {
				fgets(buf, 1000, fp);
				sscanf(buf, "%d %d %lf", &i, &j, &v);
				--nnz;
			} else {
				fprintf(stderr,"Error: no more entry to iterate !!\n");
			}
			return rate_t(i,j,v);
		}
		virtual ~entry_iterator_t(){
			if (fp) fclose(fp);
		}
};

/*
	Comparator for sorting entries into row/column compression storage 
*/
class SparseComp {
	public:
		const long *row_idx;
		const long *col_idx;
		SparseComp(const long *row_idx_, const long *col_idx_, bool isRCS_=true) {
			row_idx = (isRCS_)? row_idx_: col_idx_;
			col_idx = (isRCS_)? col_idx_: row_idx_;
		}
		bool operator()(size_t x, size_t y) const {
			return  (row_idx[x] < row_idx[y]) || ((row_idx[x] == row_idx[y]) && (col_idx[x]<= col_idx[y]));
		}
};

/*
	Sparse matrix format (i.e., CSR & CSC) for a training matrix 
*/
class smat_t{
	public:
		long rows, cols;
		long nnz, max_row_nnz, max_col_nnz;
		double *val, *val_t;
		long *col_ptr, *row_ptr;
		long *row_idx, *col_idx;  
		bool mem_alloc_by_me;

		smat_t():mem_alloc_by_me(false){ }
		smat_t(const smat_t& m){ *this = m; mem_alloc_by_me = false;}

		smat_t(long m, long n, long *ir, long *jc, double *v, long *ir_t, long *jc_t, double *v_t):
			rows(m), cols(n), mem_alloc_by_me(false), 
			row_idx(ir), col_ptr(jc), val(v), col_idx(ir_t), row_ptr(jc_t), val_t(v_t) {
			if(col_ptr[n] != row_ptr[m]) 
				fprintf(stderr,"Error occurs! two nnz do not match (%ld, %ld)\n", col_ptr[n], row_ptr[n]);
			nnz = col_ptr[n];
			max_row_nnz = max_col_nnz = 0;
			for(long r=1; r<=rows; ++r)
				max_row_nnz = max(max_row_nnz, row_ptr[r]);
			for(long c=1; c<=cols; ++c) 
				max_col_nnz = max(max_col_nnz, col_ptr[c]);
		}

		void load(long _rows, long _cols, long _nnz, const char* filename){
			entry_iterator_t entry_it(_nnz, filename);
			load_from_iterator(_rows, _cols, _nnz, &entry_it);
		}

		void load_from_iterator(long _rows, long _cols, long _nnz, entry_iterator_t* entry_it) {
			rows =_rows,cols=_cols,nnz=_nnz;
			mem_alloc_by_me = true;
			val = MALLOC(double, nnz); val_t = MALLOC(double, nnz);
			row_idx = MALLOC(long, nnz); col_idx = MALLOC(long, nnz);  // switch to this for memory
			row_ptr = MALLOC(long, rows+1); col_ptr = MALLOC(long, cols+1);
			memset(row_ptr,0,sizeof(long)*(rows+1));
			memset(col_ptr,0,sizeof(long)*(cols+1));

			// a trick here to utilize the space the have been allocated 
			vector<size_t> perm(_nnz);
			long *tmp_row_idx = col_idx;
			long *tmp_col_idx = row_idx;
			double *tmp_val = val;
			for(size_t idx = 0; idx < _nnz; idx++){
				rate_t rate = entry_it->next();
				// for shuffling indices of rows and cols
				int i = rate.i, j = rate.j;
				row_ptr[rate.i+1]++;
				col_ptr[rate.j+1]++;
				tmp_row_idx[idx] = rate.i; 
				tmp_col_idx[idx] = rate.j;
				tmp_val[idx] = rate.v;
				perm[idx] = idx;
			}
			// sort entries into row-majored ordering
			sort(perm.begin(), perm.end(), SparseComp(tmp_row_idx, tmp_col_idx, true));
			// Generate CSR format
			for(size_t idx = 0; idx < _nnz; idx++) {
				val_t[idx] = tmp_val[perm[idx]];
				col_idx[idx] = tmp_col_idx[perm[idx]];
			}

			// Calculate nnz for each row and col
			max_row_nnz = max_col_nnz = 0;
			for(long r=1; r<=rows; ++r) {
				max_row_nnz = max(max_row_nnz, row_ptr[r]);
				row_ptr[r] += row_ptr[r-1];
			}
			for(long c=1; c<=cols; ++c) {
				max_col_nnz = max(max_col_nnz, col_ptr[c]);
				col_ptr[c] += col_ptr[c-1];
			}
			// Transpose CSR into CSC matrix
			for(long r=0; r<rows; ++r){
				for(long i = row_ptr[r]; i < row_ptr[r+1]; ++i){
					long c = col_idx[i];
					row_idx[col_ptr[c]] = r; 
					val[col_ptr[c]] = val_t[i];
					col_ptr[c]++;
				}
			}
			for(long c=cols; c>0; --c) col_ptr[c] = col_ptr[c-1];
			col_ptr[0] = 0;
		}

		long nnz_of_row(int i) const {return (row_ptr[i+1]-row_ptr[i]);}
		long nnz_of_col(int i) const {return (col_ptr[i+1]-col_ptr[i]);}

		void free(void *ptr) {if(ptr) ::free(ptr);}
		~smat_t(){
			if(mem_alloc_by_me) {
				//puts("Warnning: Somebody just free me.");
				free(val); free(val_t);
				free(row_ptr);free(row_idx); 
				free(col_ptr);free(col_idx);
			}
		}

		void clear_space() {
			free(val); free(val_t);
			free(row_ptr);free(row_idx); 
			free(col_ptr);free(col_idx);
			mem_alloc_by_me = false;

		}

		smat_t transpose(){
			smat_t mt;
			mt.cols = rows; mt.rows = cols; mt.nnz = nnz;
			mt.val = val_t; mt.val_t = val;
			mt.col_ptr = row_ptr; mt.row_ptr = col_ptr;
			mt.col_idx = row_idx; mt.row_idx = col_idx;
			mt.max_col_nnz=max_row_nnz; mt.max_row_nnz=max_col_nnz;
			return mt;
		}
};

/*
	Row-major iterator 
*/
class smat_iterator_t: public entry_iterator_t{
	private:
		long *col_idx;
		long *row_ptr;
		double *val_t;
		size_t	rows, cols, cur_idx, cur_row;
	public:
		smat_iterator_t(const smat_t& M, int major = ROWMAJOR) {
			nnz = M.nnz;
			col_idx = (major == ROWMAJOR)? M.col_idx: M.row_idx;
			row_ptr = (major == ROWMAJOR)? M.row_ptr: M.col_ptr;
			val_t = (major == ROWMAJOR)? M.val_t: M.val;
			rows = (major==ROWMAJOR)? M.rows: M.cols;
			cols = (major==ROWMAJOR)? M.cols: M.rows;
			cur_idx = cur_row = 0;
		}
		~smat_iterator_t() {}
		rate_t next() {
			int i = 1, j = 1;
			double v = 0;
			while (cur_idx >= row_ptr[cur_row+1]) ++cur_row;
			if (nnz > 0) --nnz;
			else fprintf(stderr,"Error: no more entry to iterate !!\n");
			rate_t ret(cur_row, col_idx[cur_idx], val_t[cur_idx]);
			cur_idx++;
			return ret;
		}
};

/*
	Sparse matrix format for a test matrix 
*/
class testset_t{
	public:
	long rows, cols, nnz;
	vector<rate_t> T;
	testset_t(): rows(0), cols(0), nnz(0){}
	inline rate_t& operator[](const long &idx) {return T[idx];}
	void load(long _rows, long _cols, long _nnz, const char *filename) {
		int r, c; 
		double v;
		rows = _rows; cols = _cols; nnz = _nnz;
		T = vector<rate_t>(nnz);
		FILE *fp = fopen(filename, "r");
		for(long idx = 0; idx < nnz; ++idx){
			fscanf(fp, "%d %d %lg", &r, &c, &v); 
			T[idx] = rate_t(r,c,v);
		}
		fclose(fp);
	}
	void load_from_iterator(long _rows, long _cols, long _nnz, entry_iterator_t* entry_it){ 
		rows =_rows,cols=_cols,nnz=_nnz;
		T = vector<rate_t>(nnz);
		for(size_t idx=0; idx < nnz; ++idx) 
			T[idx] = entry_it->next();
	}
};


#endif
