/*
* @file        main.cpp
* @author      Dongha Lee (dongha.lee@postech.ac.kr), Pohang University of Science and Technology
* @author      Jaehyung Lee (anthi7@postech.ac.kr), Pohang University of Science and Technology
* @author      Hwanjo Yu (hwanjoyu@postech.ac.kr), Pohang University of Science and Technology
* @date        2018-09-28
*
* Fast Tucker Factorization for Large-scale Tensor Completion
*
* This code is implemented based on the author code of
*      Oh et. al., "Scalable Tucker Factorization for Sparse Tensors - Algorithms and Discoveries", in ICDE, 2018.
* This software is free of charge under research purposes.
*
* Usage:
*   - make all
*/

#include "util.h"
#include "Option.h"
#include "Tensor.h"
#include "Tucker.h"

int main(int argc, char* argv[]) {

	Tensor G;	/* Core tensor */
	Tensor Xtest; 	/* Data tensor with test entries */
	double ***FactorMat;	/* N Factor matrices  */
	int* Dims;	/* N dimensionalities of the data tensor */

	Option::parse(argc, argv);

	if (Option::buildTensorFile == BUILD_ON_MEMORY)
		Build_GTF_From_Memory();
	if (Option::buildTensorFile == BUILD_ON_DISK)
		Build_GTF_From_Disk();

	Initialize(FactorMat, G, Dims, Xtest);
	FTcom(FactorMat, G, Dims, Xtest);
	Print(FactorMat, G, Dims);

	return 0;
}
