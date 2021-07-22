# Topology Distillation for Recommender System
This repository provides the source code of "Topology Distillation for Recommender System" accepted in KDD2021 as a research paper.

### 1. Overview
We develop a general topology distillation approach for Recommender System.
The topology distillation guides the learning of the student model by the topological structure built upon the relational knowledge in representation space of the teacher model.

Concretely, we propose two topology distillation methods: 

  <ol type="a">
  <li><b>Full Topology Distillation (FTD)</b>. 
      FTD transfers the full topology, and it is used in the scenario where the student has enough capacity to learn all the teacher’s knowledge.</li>
  
  <li><b>Hierarchical Topology Distillation (HTD)</b>. HTD transfers the decomposed topology hierarchically, and it is adopted in the classical KD scenario where the student has a very limited capacity compared to the teacher.</li>
  </ol>


### 2. Main Results

- When the capacity of the student model is highly limited, the student model learns best with HTD.

  ![TD1](https://user-images.githubusercontent.com/68782810/124361145-8a4ddc00-dc68-11eb-8a2f-1e93efd26184.jpg)

- As the capacity gap between the teacher model and student model decreases, the student model takes more benefits from FTD.

  ![TD2](https://user-images.githubusercontent.com/68782810/124361147-8b7f0900-dc68-11eb-8ccd-bfe93131c7b0.jpg)


### 3. Requirements
- Python version: 3.6.10
- Pytorch version: 1.5.0

### 4. How to Run
```
Please refer to 'Guide to using topology distillation.ipynb' file.
```
