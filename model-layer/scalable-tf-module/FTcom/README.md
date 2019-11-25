# Fast Tucker Factorization for Large-Scale Tensor Completion

### Overview
> Tensor completion is the task of completing multiaspect data represented as a tensor by accurately predicting missing entries in the tensor. It is mainly solved by tensor factorization methods, and among them, Tucker factorization has attracted considerable interests due to its powerful ability to learn latent factors and even their interactions. Although several Tucker methods have been developed to reduce the memory and computational complexity, the state-of-the-art method still 1) generates redundant computations and 2) cannot factorize a large tensor that exceeds the size of memory. This paper proposes FTcom, a fast and scalable Tucker factorization method for tensor completion. FTcom performs element-wise updates for factor matrices based on coordinate descent, and adopts a novel caching algorithm which stores frequently-required intermediate data. It also uses a tensor file for disk-based data processing and loads only a small part of the tensor at a time into the memory. Experimental results show that FTcom is much faster and more scalable compared to all other competitors. It significantly shortens the training time of Tucker factorization, especially on real-world tensors, and it can be executed on a billion-scale tensor which is bigger than the memory capacity within a single machine.

### Paper
- Fast Tucker Factorization for Large-Scale Tensor Completion (*ICDM 2018*)
  - **Dongha Lee**, Jaehyung Lee, Hwanjo Yu

### Requirements
- OpenMP library 
- Armadillo library (It needs LAPACK and BLAS libraries)

### Configuration
- **Path**

Parameter | Default | Description
--- | :---: | ---
`--train-path <path>` | { } | A raw text file of observed tensor entries
`--test-path <path>` | { } | A raw text file of observed tensor entries used for testing *(optional)*
`--result-path <path>` | { } | A directory where obtained core tensor and factor matrices are stored as text files

- **File construction types**

Parameter | Default | Description
--- | :---: | ---
`--build-on-memory` | `default` |  In case that an input tensor fits into your memory (fast construction)
`--build-on-disk` | | In case that an input tensor is larger than your memory (slow construction)
`--no-build` | | In case that a grid-based tensor file (GTF) is already built

- **Solver types**

Parameter | Default | Description
--- | :---: | ---
`--cd-solver` | `default` |  The coordinate descent-based update rule 
`--nn-solver` | |  The coordinate descent-based update rule that enforces the non-negativity constraint

- **Parameters**

Parameter | Default | Description
--- | :---: | ---
`--tensor-order <int>` | { } | The order of an input tensor
`--rank-size <int>` | 10 | The rank of factor matrices 
`--iteration-size <int>` | 10 | The number of total iterations
`--cache-size <int>` | 1000000 | The number of delta vectors that can be stored in the cache table at a time
`--grid-size <int>` | 1 | The size of a grid 
`--thread-size <int>` | 1 | The number of concurrent threads
`--lambda <float>` | 0.01 | The regularization coefficient

### Note
- This code is implemented based on the author code of <a href="https://datalab.snu.ac.kr/ptucker/" target="_blank">Oh et. al., "Scalable Tucker Factorization for Sparse Tensors - Algorithms and Discoveries", in ICDE, 2018</a>.
- We recommend you to use a solid-state disk (SSD) for efficient and scalable Tucker factorization.
