#ifndef GRIDCSC
#define GRIDCSC

#include "ocam.h"
#include "util.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

class csc_idx;

void buildGCSCFromFile(const char *srcdir, int gridSize, int threadSize);
void buildGCSCFromMatrix(const char *srcdir, int gridSize);

class csc_idx{
	public:
		long cols, rows, nnz;
		long col_offset, row_offset;

		long *ptr;
		long *col_ptr, *row_ptr;
		long *row_idx, *col_idx;

		csc_idx(): nnz(0) { }
		csc_idx(const csc_idx& m){ *this = m;}

		RVALTYPE *readCSCFromFile(const char *filename);
		void writeCSCToFile(const char *filename, RVALTYPE *val);
		void writeValToFile(const char *filename, RVALTYPE *val);

		RVALTYPE *readGCSCFromFile(const int fd, int cellIdx);
		void writeGCSCToFile(const int fd, int cellIdx, RVALTYPE *val);
		void writeGValToFile(const int fd, int cellIdx, RVALTYPE *val);

		RVALTYPE* transC2R(RVALTYPE *blockVal);
		void transR2C(RVALTYPE *transVal, RVALTYPE *blockVal);
};

#endif
