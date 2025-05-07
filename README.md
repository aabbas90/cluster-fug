# ClusterFuG: Clustering Fully connected Graphs by Multicut [(arxiv)](https://arxiv.org/abs/2301.12159)
Solve multicut problem on complete graph alleviating need for graph structure design. Similarity between a pair of nodes is computed by inner product of input node features.

## Requirements
We use `GCC 10`. Other combinations might also work but not tested. `CMake` is required for compilation.

## Installation

### C++ solver:
```bash
git clone git@github.com:aabbas90/cluster-fug.git
cd cluster-fug
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make -j 4
```

### Python bindings:
We also provide python bindings using [pybind](https://github.com/pybind/pybind11). Simply run the following command:

```bash
python -m pip install git+https://github.com/aabbas90/cluster-fug.git
```

## Usage

### C++ solver:
We require multicut instance stored in a (.txt) file in the following format:
```
FACTORIZED COMPLETE MULTICUT
a
n d
feature_node_1_dim_1 ... feature_node_1_dim_d
...
feature_node_n_dim_1 ... feature_node_n_dim_d
```
which corresponds to a graph with `n` nodes, and `d`-dimensional node features. The value of `a` controls affinity strength where increasing the value helps in creating more clusters and vice versa.
Afterwards run (for our **DAppLAEC** algorithm):
```bash
./src/dense_multicut_text_input <PATH_TO_PROBLEM_INSTANCE_TXT> HNSW laec_bf_later
```
Other algorithms mentioned in the paper can be run as:

```bash
./src/dense_multicut_text_input <PATH_TO_PROBLEM_INSTANCE_TXT> <INDEX_TYPE> <CONTRACTION_TYPE>
```
where `INDEX_TYPE` and `CONTRACTION_TYPE` can be choosen as:
| Algorithm  | INDEX_TYPE  | CONTRACTION_TYPE |
|---|---|---|
| GAEC |  brute_force | gaec  |
| DGAEC | faiss_brute_force | dense_gaec   |
| DGAECInc | faiss_brute_force | dense_gaec_inc   |
| DLAEC | faiss_brute_force | dense_laec   |
| DAppLAEC | HNSW | dense_laec_bf_later   |

For example to run **GAEC**:
```bash
./src/dense_multicut_text_input <PATH_TO_PROBLEM_INSTANCE_TXT> brute_force gaec
```
 For more information run: 
```bash
./src/dense_multicut_text_input --help
```

### Python solver:
An example to compute multicut on a set of features:
```python
import dense_multicut_py
import numpy as np

num_nodes = 100
dim = 16
affinity_strength = 5.0
k = 5
k_cap = k
# DAppLAEC:
INDEX_TYPE, CONTRACTION_TYPE = "HNSW", "dense_laec_bf_later"
features = np.random.rand(num_nodes, dim).astype(np.float32)
node_labels = dense_multicut_py.dense_multicut(features.flatten(), num_nodes, dim, affinity_strength, k, INDEX_TYPE, CONTRACTION_TYPE, k_cap)
```

### Benchmark instances:
Instances used in the paper can be obtained from [structured-prediction-prob-archive](https://arxiv.org/abs/2202.03574).

## References
If you use this work please cite as
```
@inproceedings{abbas2023clusterfug,
  title={ClusterFuG: clustering fully connected graphs by multicut},
  author={Abbas, Ahmed and Swoboda, Paul},
  booktitle={International Conference on Machine Learning},
  pages={19--30},
  year={2023},
  organization={PMLR}
}
```
