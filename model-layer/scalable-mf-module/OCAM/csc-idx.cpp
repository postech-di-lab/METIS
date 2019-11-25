#include "csc-idx.h"
#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))

/*
	Transpose a CSC sub-matrix into a CSR sub-matrix 
*/
RVALTYPE* csc_idx::transC2R(RVALTYPE *blockVal) {
	RVALTYPE *transVal = MALLOC(RVALTYPE, nnz);
	col_idx = MALLOC(long, nnz);

	for(long c=0; c<cols; c++) {
		for(long i=col_ptr[c]; i<col_ptr[c+1]; i++) {
			long r = row_idx[i];
			col_idx[row_ptr[r]] = c;
			transVal[row_ptr[r]] = blockVal[i];
			row_ptr[r]++;
		}
	}
	for(long r=rows; r>0; r--) row_ptr[r] = row_ptr[r-1];
	row_ptr[0] = 0;
	return transVal;
}

/*
	Transpose a CSR sub-matrix into a CSC sub-matrix 
*/
void csc_idx::transR2C(RVALTYPE *transVal, RVALTYPE *blockVal) {
	for(long r=0; r<rows; r++) {
		for(long i=row_ptr[r]; i<row_ptr[r+1]; i++) {
			long c = col_idx[i];
			blockVal[col_ptr[c]] = transVal[i];
			col_ptr[c]++;
		}
	}
	for(long c=cols; c>0; c--) col_ptr[c] = col_ptr[c-1];
	col_ptr[0] = 0;

	free(col_idx);
	free(transVal);
}

/*
	Read and Write functions for CSC file
*/
RVALTYPE* csc_idx::readCSCFromFile(const char *filename){
	int fd = open(filename, O_CREAT|O_RDWR, S_IRWXU);

	// read cols, rows, nnz, col_offset, row_offset from a file
	read(fd, this, sizeof(long)*5);

	ptr = MALLOC(long, cols+rows+nnz+2);
	RVALTYPE *val = MALLOC(RVALTYPE, nnz);

	read(fd, ptr, sizeof(long)*(cols+rows+nnz+2));
	read(fd, val, sizeof(RVALTYPE)*nnz);

	col_ptr = ptr;
	row_ptr = col_ptr+cols+1;
	row_idx = row_ptr+rows+1;
	col_idx = NULL;	

	close(fd);

	return val;
}

void csc_idx::writeCSCToFile(const char *filename, RVALTYPE *val){
	int fd = open(filename, O_CREAT|O_RDWR, S_IRWXU);

	write(fd, this, sizeof(long)*5);
	write(fd, ptr, sizeof(long)*(cols+rows+2));

	size_t offset_ptr = sizeof(long)*(cols+rows+7);
	size_t offset_val = sizeof(long)*(cols+rows+nnz+7);
	size_t num = 0;

	for(size_t c=0; c<cols; c++) {
		num = col_ptr[c+1] - col_ptr[c];

		pwrite(fd, &row_idx[col_ptr[c]], sizeof(long)*num, offset_ptr);
		pwrite(fd, &val[col_ptr[c]], sizeof(RVALTYPE)*num, offset_val);

		offset_ptr += sizeof(long)*num;
		offset_val += sizeof(RVALTYPE)*num;
	}

	close(fd);
}

void csc_idx::writeValToFile(const char *filename, RVALTYPE *val){
	int fd = open(filename, O_CREAT|O_RDWR, S_IRWXU);

	pwrite(fd, val, sizeof(RVALTYPE)*nnz, sizeof(long)*(cols+rows+nnz+7));

	close(fd);
}

/*
	Read and Write functions for GCSC file
*/
RVALTYPE* csc_idx::readGCSCFromFile(const int fd, int cellIdx){

	off_t offset_gcsc;
	pread(fd, &offset_gcsc, sizeof(off_t), sizeof(int)+sizeof(off_t)*cellIdx);
	lseek(fd, offset_gcsc, SEEK_SET);

	// read cols, rows, nnz, col_offset, row_offset from a file
	read(fd, this, sizeof(long)*5);

	ptr = MALLOC(long, cols+rows+nnz+2);
	RVALTYPE *val = MALLOC(RVALTYPE, nnz);

	read(fd, ptr, sizeof(long)*(cols+rows+nnz+2));
	read(fd, val, sizeof(RVALTYPE)*nnz);

	col_ptr = ptr;
	row_ptr = col_ptr+cols+1;
	row_idx = row_ptr+rows+1;
	col_idx = NULL;	

	return val;
}

void csc_idx::writeGCSCToFile(const int fd, int cellIdx, RVALTYPE *val){
	
	off_t offset_cell;
	pread(fd, &offset_cell, sizeof(off_t), sizeof(int)+sizeof(off_t)*cellIdx);
	lseek(fd, offset_cell, SEEK_SET);
	
	write(fd, this, sizeof(long)*5);
	write(fd, ptr, sizeof(long)*(cols+rows+2));

	off_t offset_ptr = offset_cell + sizeof(long)*(cols+rows+7);
	off_t offset_val = offset_cell + sizeof(long)*(cols+rows+nnz+7);
	size_t num = 0;

	for(long c=0; c<cols; c++) {
		num = col_ptr[c+1] - col_ptr[c];

		pwrite(fd, &row_idx[col_ptr[c]], sizeof(long)*num, offset_ptr);
		pwrite(fd, &val[col_ptr[c]], sizeof(RVALTYPE)*num, offset_val);

		offset_ptr += sizeof(long)*num;
		offset_val += sizeof(RVALTYPE)*num;
	}

}

void csc_idx::writeGValToFile(const int fd, int cellIdx, RVALTYPE *val){

	off_t offset_cell;
	pread(fd, &offset_cell, sizeof(off_t), sizeof(int)+sizeof(off_t)*cellIdx);

	off_t offset_val = offset_cell + sizeof(long)*(cols+rows+nnz+7);
	pwrite(fd, val, sizeof(RVALTYPE)*nnz, offset_val);

}

/*
    This function writes constructed GCSC files to a disk, after loading the raw file as the GCSC structure
    It is a disk-based process, so it can handle any large-scale data whose size exceeds a main memory
*/
void buildGCSCFromFile(const char *srcdir, int gridSize, int threadSize){
	char meta_filename[1024], input_filename[1024], filename[1024], buf[1024];
	sprintf(meta_filename, "%s/meta", srcdir);
	FILE *fp = fopen(meta_filename, "r");

	long rows, cols;
	size_t nnz;
	fscanf(fp, "%ld %ld", &rows, &cols);
	fscanf(fp, "%ld %s", &nnz, buf);
	fclose(fp);

	sprintf(input_filename, "%s/%s", srcdir, buf);

	// allocate and initialize the space

	long *col_ptr = MALLOC(long, cols+1);
	long *row_ptr = MALLOC(long, rows+1);
	memset(col_ptr, 0, sizeof(long)*(cols+1));
	memset(row_ptr, 0, sizeof(long)*(rows+1));

	long *colGrid = MALLOC(long, gridSize+1);
	long *rowGrid = MALLOC(long, gridSize+1);
	memset(colGrid, 0, sizeof(long)*(gridSize+1));
	memset(rowGrid, 0, sizeof(long)*(gridSize+1));

	// scan the input file, and obtain the col_ptr, row_ptr 

	int _i, _j;
	double _v;
	fp = fopen(input_filename, "r");
	for(size_t idx=0; idx<nnz; idx++) {
#ifndef HUGEWIKI
		fscanf(fp, "%d %d %lf", &_i, &_j, &_v);
#else
		fscanf(fp, "%d %d %lf", &_j, &_i, &_v);
#endif
		row_ptr[_i+1]++;
		col_ptr[_j+1]++;
	}	

	for(long c=1; c<=cols; c++)
		col_ptr[c] += col_ptr[c-1];
	for(long r=1; r<=rows; r++)
		row_ptr[r] += row_ptr[r-1];

	fclose(fp);

	// determine grid widths and heights for a balanced grid file

	int grid_col_idx = 0, grid_row_idx = 0;
	long entryNum = (long)nnz/gridSize+1;

	for(long j=0; j<cols; j++) {
		if(col_ptr[j] >= entryNum*grid_col_idx)
			colGrid[grid_col_idx++] = j;
	}
	colGrid[grid_col_idx] = cols;

	for(long i=0; i<rows; i++) {
		if(row_ptr[i] >= entryNum*grid_row_idx)
			rowGrid[grid_row_idx++] = i;		
	}
	rowGrid[grid_row_idx] = rows;

	// count nnz of each grid cell

	size_t *nnzCnt = MALLOC(size_t, gridSize*gridSize);
	memset(nnzCnt, 0, sizeof(size_t)*gridSize*gridSize);
	fp = fopen(input_filename, "r");
	for(size_t idx=0; idx<nnz; idx++) {
		grid_col_idx = 0;
		grid_row_idx = 0;
#ifndef HUGEWIKI
		fscanf(fp, "%d %d %lf", &_i, &_j, &_v);
#else
		fscanf(fp, "%d %d %lf", &_j, &_i, &_v);
#endif
		while(_j >= colGrid[grid_col_idx+1]) grid_col_idx++;
		while(_i >= rowGrid[grid_row_idx+1]) grid_row_idx++;
		nnzCnt[grid_row_idx + grid_col_idx*gridSize]++;		
	}
	fclose(fp);

	// open a file and write meta-information

	sprintf(filename, "%s/gcsc", srcdir);

	int fd = open(filename, O_CREAT|O_RDWR, S_IRWXU);
	write(fd, &gridSize, sizeof(int));

	off_t offset_cell = sizeof(int) + sizeof(off_t)*gridSize*gridSize;
	for(grid_col_idx=0; grid_col_idx<gridSize; grid_col_idx++)
		for(grid_row_idx=0; grid_row_idx<gridSize; grid_row_idx++) {  
			write(fd, &offset_cell, sizeof(off_t));
			offset_cell += sizeof(long)*(colGrid[grid_col_idx+1] - colGrid[grid_col_idx]);
			offset_cell += sizeof(long)*(rowGrid[grid_row_idx+1] - rowGrid[grid_row_idx]);
			offset_cell += sizeof(long)*(nnzCnt[grid_row_idx + grid_col_idx*gridSize] + 7);
			offset_cell += sizeof(RVALTYPE)*nnzCnt[grid_row_idx + grid_col_idx*gridSize];
		}

	close(fd);

	// build a grid column at a time because of a memory limitation

	int num_threads_old = omp_get_num_threads();
	omp_set_num_threads(threadSize);

#pragma omp parallel for schedule(dynamic)
	for(int g=0; g<gridSize; g++) {
		csc_idx *gridIdx = new csc_idx[gridSize];
		RVALTYPE **gridVal = MALLOC(RVALTYPE*, gridSize);  

		// allocate the space for each csc-idx and csc-val
		// tmp_col_idx, tmp_row_idx, tmp_val, permutation are temporary space for sorting

		vector<vector<size_t> > perm(gridSize);
		long **tmp_col_idx = MALLOC(long*, gridSize);
		long **tmp_row_idx = MALLOC(long*, gridSize);
		RVALTYPE **tmp_val = MALLOC(RVALTYPE*, gridSize);

		for(int i=0; i<gridSize; i++) {
			size_t tmp_nnz = nnzCnt[i + g*gridSize];

			perm[i] = vector<size_t>(tmp_nnz);
			tmp_col_idx[i] = MALLOC(long, tmp_nnz);
			tmp_row_idx[i] = MALLOC(long, tmp_nnz);
			tmp_val[i] = MALLOC(RVALTYPE, tmp_nnz);

			gridIdx[i].cols = colGrid[g+1] - colGrid[g];
			gridIdx[i].rows = rowGrid[i+1] - rowGrid[i];

			gridIdx[i].col_offset = colGrid[g];
			gridIdx[i].row_offset = rowGrid[i];

			long ptrSize = gridIdx[i].cols + gridIdx[i].rows + tmp_nnz + 2;

			gridIdx[i].ptr = MALLOC(long, ptrSize); 
			gridVal[i] = MALLOC(RVALTYPE, tmp_nnz);

			memset(gridIdx[i].ptr, 0, sizeof(long)*ptrSize);
			memset(gridVal[i], 0, sizeof(RVALTYPE)*tmp_nnz);

			gridIdx[i].col_ptr = gridIdx[i].ptr;
			gridIdx[i].row_ptr = gridIdx[i].col_ptr + gridIdx[i].cols + 1;
			gridIdx[i].row_idx = gridIdx[i].row_ptr + gridIdx[i].rows + 1;
			gridIdx[i].nnz = 0;

		}

		// fill in each csc-idx and csc-val

		long sub_col_idx = 0, sub_row_idx = 0, sub_val_idx = 0;
		FILE *t_fp = fopen(input_filename, "r");
		int t_i, t_j;
		double t_v; 
		for(size_t idx=0; idx<nnz; idx++) {
			int t_grid_col_idx = 0;
			int t_grid_row_idx = 0;
#ifndef HUGEWIKI
			fscanf(t_fp, "%d %d %lf", &t_i, &t_j, &t_v);
#else
			fscanf(t_fp, "%d %d %lf", &t_j, &t_i, &t_v);
#endif
			if(!(t_j >= colGrid[g] && t_j < colGrid[g+1])) continue;
			while(t_i >= rowGrid[t_grid_row_idx+1]) t_grid_row_idx++;

			sub_col_idx = t_j - gridIdx[t_grid_row_idx].col_offset;	
			sub_row_idx = t_i - gridIdx[t_grid_row_idx].row_offset;

			perm[t_grid_row_idx][gridIdx[t_grid_row_idx].nnz] = gridIdx[t_grid_row_idx].nnz;
			tmp_col_idx[t_grid_row_idx][gridIdx[t_grid_row_idx].nnz] = sub_col_idx;
			tmp_row_idx[t_grid_row_idx][gridIdx[t_grid_row_idx].nnz] = sub_row_idx;
			tmp_val[t_grid_row_idx][gridIdx[t_grid_row_idx].nnz] = t_v;

			gridIdx[t_grid_row_idx].col_ptr[sub_col_idx+1]++;
			gridIdx[t_grid_row_idx].row_ptr[sub_row_idx+1]++;
			gridIdx[t_grid_row_idx].nnz++;

		}
		fclose(t_fp);

		// sort entries into column-majored ordering

		for(int i=0; i<gridSize; i++) {
			sort(perm[i].begin(), perm[i].end(), SparseComp(tmp_col_idx[i], tmp_row_idx[i], true));
			for(size_t idx=0; idx<gridIdx[i].nnz; idx++) {
				gridIdx[i].row_idx[idx] = tmp_row_idx[i][perm[i][idx]];
				gridVal[i][idx] = tmp_val[i][perm[i][idx]];
			}

			for(long c=1; c<=gridIdx[i].cols; c++)
				gridIdx[i].col_ptr[c] += gridIdx[i].col_ptr[c-1];
			for(long r=1; r<=gridIdx[i].rows; r++)
				gridIdx[i].row_ptr[r] += gridIdx[i].row_ptr[r-1];

			vector<size_t>().swap(perm[i]);
			free(tmp_col_idx[i]);
			free(tmp_row_idx[i]);
			free(tmp_val[i]);
		}

		vector<vector<size_t> >().swap(perm);
		free(tmp_col_idx);
		free(tmp_row_idx);
		free(tmp_val);

		// write the constructed GCSC files to a disk

		int t_fd = open(filename, O_RDWR, S_IRWXU);

		for(int i=0; i<gridSize; i++){
			gridIdx[i].writeGCSCToFile(t_fd, i + g*gridSize, gridVal[i]);

			free(gridIdx[i].ptr);
			free(gridVal[i]);
		}

		close(t_fd);

		delete[] gridIdx;
		free(gridVal);
		
		printf("Build-GCSC progress ... %d/%d\n", g+1, gridSize);
	}

	omp_set_num_threads(num_threads_old);

	free(col_ptr);
	free(row_ptr);
	free(nnzCnt);

	free(colGrid);
	free(rowGrid);

	close(fd);

	printf("Grid-CSC file is created\n");
}

/*
    Constructing Grid-CSC structure from the full matrix
    This function completes each csc-idx and csc-val, and returns the array of csc-val
*/
void buildGCSCFromMatrix(const char *srcdir, int gridSize){
	char meta_filename[1024], input_filename[1024], filename[1024], buf[1024];
	sprintf(meta_filename, "%s/meta", srcdir);
	FILE *fp = fopen(meta_filename, "r");

	long rows, cols;
	size_t nnz;
	fscanf(fp, "%ld %ld", &rows, &cols);
	fscanf(fp, "%ld %s", &nnz, buf);
	fclose(fp);

	sprintf(input_filename, "%s/%s", srcdir, buf);

	smat_t R;
	R.load(rows, cols, nnz, input_filename);

	int grid_col_idx = 0, grid_row_idx = 0;

	csc_idx *gridIdx = new csc_idx[gridSize*gridSize];
	RVALTYPE **gridVal = MALLOC(RVALTYPE*, gridSize*gridSize);  

	long entryNum = (long)R.nnz/gridSize+1;

	long *colGrid = MALLOC(long, gridSize+1);
	long *rowGrid = MALLOC(long, gridSize+1);
	memset(colGrid, 0, sizeof(long)*(gridSize+1));
	memset(rowGrid, 0, sizeof(long)*(gridSize+1));

	for(long j=0; j<R.cols; j++) {
		if(R.col_ptr[j] >= entryNum*grid_col_idx)
			colGrid[grid_col_idx++] = j;
	}
	colGrid[grid_col_idx] = R.cols;

	for(long i=0; i<R.rows; i++) {
		if(R.row_ptr[i] >= entryNum*grid_row_idx)
			rowGrid[grid_row_idx++] = i;		
	}
	rowGrid[grid_row_idx] = R.rows;

	for(grid_col_idx=0; grid_col_idx<gridSize; grid_col_idx++) {
		for(long col_idx=colGrid[grid_col_idx]; col_idx<colGrid[grid_col_idx+1]; col_idx++) {
			grid_row_idx = 0;	
			for(long val_idx=R.col_ptr[col_idx]; val_idx<R.col_ptr[col_idx+1]; val_idx++) {
				while(R.row_idx[val_idx] >= rowGrid[grid_row_idx+1]) grid_row_idx++;
				gridIdx[grid_row_idx + grid_col_idx*gridSize].nnz++;
			}
		}
	}

	for(int j=0; j<gridSize; j++) {
		for(int i=0; i<gridSize; i++) {
			int block_idx = i + j*gridSize;

			gridIdx[block_idx].cols = colGrid[j+1] - colGrid[j];
			gridIdx[block_idx].rows = rowGrid[i+1] - rowGrid[i];

			gridIdx[block_idx].col_offset = colGrid[j];
			gridIdx[block_idx].row_offset = rowGrid[i];

			long ptrSize = gridIdx[block_idx].cols + gridIdx[block_idx].rows + gridIdx[block_idx].nnz + 2;

			gridIdx[block_idx].ptr = MALLOC(long, ptrSize); 
			gridVal[block_idx] = MALLOC(RVALTYPE, gridIdx[block_idx].nnz);

			memset(gridIdx[block_idx].ptr, 0, sizeof(long)*ptrSize);
			memset(gridVal[block_idx], 0, sizeof(RVALTYPE)*gridIdx[block_idx].nnz);

			gridIdx[block_idx].col_ptr = gridIdx[block_idx].ptr;
			gridIdx[block_idx].row_ptr = gridIdx[block_idx].col_ptr + gridIdx[block_idx].cols + 1;
			gridIdx[block_idx].row_idx = gridIdx[block_idx].row_ptr + gridIdx[block_idx].rows + 1;
			gridIdx[block_idx].nnz = 0;
		}
	}

	for(grid_col_idx=0; grid_col_idx<gridSize; grid_col_idx++) {
		long sub_col_idx = 0, sub_row_idx = 0, sub_val_idx = 0;

		for(long col_idx=colGrid[grid_col_idx]; col_idx<colGrid[grid_col_idx+1]; col_idx++) {
			grid_row_idx = 0;
			sub_col_idx = col_idx - colGrid[grid_col_idx];

			for(long val_idx=R.col_ptr[col_idx]; val_idx<R.col_ptr[col_idx+1]; val_idx++) {
				while(R.row_idx[val_idx] >= rowGrid[grid_row_idx+1]) grid_row_idx++;

				int block_idx = grid_row_idx + grid_col_idx*gridSize;
				sub_row_idx = R.row_idx[val_idx] - rowGrid[grid_row_idx];
				sub_val_idx = gridIdx[block_idx].nnz;

				gridIdx[block_idx].col_ptr[sub_col_idx+1]++;
				gridIdx[block_idx].row_ptr[sub_row_idx+1]++;
				gridIdx[block_idx].nnz++;

				gridIdx[block_idx].row_idx[sub_val_idx] = sub_row_idx;
				gridVal[block_idx][sub_val_idx] = R.val[val_idx];
			}
		}
	}

	for(int block_idx=0; block_idx<gridSize*gridSize; block_idx++) {
		for(long c=1; c<=gridIdx[block_idx].cols; c++)
			gridIdx[block_idx].col_ptr[c] += gridIdx[block_idx].col_ptr[c-1];
		for(long r=1; r<=gridIdx[block_idx].rows; r++)
			gridIdx[block_idx].row_ptr[r] += gridIdx[block_idx].row_ptr[r-1];
	}

	free(colGrid);
	free(rowGrid);

	// Write GCSC file from constructed Grid-CSC structure
	sprintf(filename, "%s/gcsc", srcdir);

	int fd = open(filename, O_CREAT|O_RDWR, S_IRWXU);
	write(fd, &gridSize, sizeof(int));

	off_t offset_cell = sizeof(int) + sizeof(off_t)*gridSize*gridSize;
	for(int i=0; i<gridSize*gridSize; i++){ 
		write(fd, &offset_cell, sizeof(off_t));
		offset_cell += sizeof(long)*(gridIdx[i].cols + gridIdx[i].rows + gridIdx[i].nnz + 7) + sizeof(RVALTYPE)*gridIdx[i].nnz;
	}	

	for(int i=0; i<gridSize*gridSize; i++){
		gridIdx[i].writeGCSCToFile(fd, i, gridVal[i]);

		free(gridIdx[i].ptr);
		free(gridVal[i]);
	}

	close(fd);
	delete[] gridIdx;
	free(gridVal);

	printf("Grid-CSC file is created\n");
}

