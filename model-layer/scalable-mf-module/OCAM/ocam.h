#ifndef _PMF_H_
#define _PMF_H_

#include "macros.h"
#include "util.h"
#include "csc-idx.h"

class parameter {
	public:
		int k;
		int threads;
		int maxiter;
		int grids;

		RVALTYPE lambda;
		int do_opt;  // flag for two-phase optimization
		int build_ondisk; // flag for disk-based GCSC construction

		parameter() {
			k = 40;
			threads = 4;
			maxiter = 5;
			grids = 16;
			lambda = 0.1;
			do_opt = 0;
			build_ondisk = 0;
		}
};

void ocam(const char *gcsc_filename, dmat_t W, dmat_t H, testset_t &T, parameter &param);


#endif
