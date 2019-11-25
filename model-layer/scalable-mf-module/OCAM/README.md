# OCAM: Out-of-core Coordinate Descent Algorithm for Matrix Completion

### Overview
> Recently, there are increasing reports that most datasets can be actually stored in disks of a single off-the-shelf workstation, and utilizing out-of-core methods is much cheaper and even faster than using a distributed system. For these reasons, out-of-core methods have been actively developed for machine learning and graph processing. The goal of this paper is to develop an efficient out-of-core matrix completion method based on coordinate descent approach. Coordinate descent-based matrix completion (CD-MC) has two strong benefits over other approaches: 1) it does not involve heavy computation such as matrix inversion and 2) it does not have step-size hyper-parameters, which reduces the effort for hyper-parameter tuning. Existing solutions for CD-MC have been developed and analyzed for in-memory setting and they do not take disk-I/O into account. Thus, we propose OCAM, a novel out-of-core coordinate descent algorithm for matrix completion. Our evaluation results and cost analyses provide sound evidences supporting the following benefits of OCAM: (1) Scalability – OCAM is a truly scalable out-of-core method and thus decomposes a matrix larger than the size of memory, (2) Efficiency – OCAM is super fast. OCAM is **up to 10x faster** than the state-of-the-art out-of-core method, and **up to 4.1x faster** than a competing distributed method when using eight machines.

### Paper
- OCAM: Out-of-core Coordinate Descent Algorithm for Matrix Completion (*Information Sciences, 2019*)
  - **Dongha Lee**, Jinoh Oh, Hwanjo Yu

### Requirements
- OpenMP library 

### Configuration

- **Parameters for build-gcsc**

Parameter | Default | Description
--- | :---: | ---
`-n <int>` | 4 | The number of concurrent threads
`-g <int>` | 16 | The size of a grid
`-d <int>` | 0 | 0 : build the GCSC file on memory, 1 : build the GCSC file on disk

- **Parameters for train-ocam**

Parameter | Default | Description
--- | :---: | ---
`-k <int>` | 40 | The rank size of factor matrices
`-l <float>` | 0.01 | The regularization coefficient
`-t <int>` | 5 | The number of total iterations
`-n <int>` | 4 | The number of concurrent threads
`-g <int>` | 16 | The size of a grid
`-o <int>` | 0 | 0 : OCAM, 1 : OCAM-opt

### Note
- This code is implemented based on the author code of <a href="http://www.cs.utexas.edu/~rofuyu/libpmf/" target="_blank">Yu et al., "Scalable Coordinate Descent Approaches to Parallel Matrix Factorization for Recommender Systems", in ICDM, 2012</a>.
- We recommend you to use a solid-state disk (SSD) for efficient and scalable matrix factorization.
