## HGNN-based Job Execution Time Prediction

This directory contains the code for a heterogeneous graph neural network (HGNN) for predicting job execution time in distributed computing environments.

### Overview

This work proposes a heterogeneous GraphSAGE-based model that learns from the complex relationships between tasks, job execution time, and computing clusters. By representing these entities and their interactions as a heterogeneous graph, the model effectively captures the structural patterns that influence application performance.

---

### Directory Structure

- `main.py`: Main training and evaluation pipeline with k-fold cross-validation
- `GNN_architecture.py`: Defines the `HeteroGraphSage` model architecture
- `utils.py`: Utility functions for data processing, graph construction, and training routines
- `datasets/`: Directory containing input data files:
  - `clusters.csv`: Cluster node features
  - `runtimes.csv`: Runtime edge data with execution times
  - `tasks.csv`: Task node features and labels

---

### Requirements

**Dependencies:**

```
dgl==2.4.0+cu121
torch==2.4.0
pandas==2.3.2
numpy==1.26.4
scikit-learn==1.5.2
```

Install DGL following the [official installation guide](https://www.dgl.ai/pages/start.html) based on your platform and CUDA version.

---

### Dataset Structure

The dataset models distributed computing workloads as a heterogeneous graph with three node types:

**cluster.csv:**

```csv
cluster_id,label,feat
0,normal_cluster,"[200, 120]"
```

- `cluster_id`: Numeric cluster identifier
- `label`: Cluster name
- `feat`: Resource configuration features

**runtime.csv:**

```csv
appMasterHost,appId,runtime
normal_cluster,application_1758535431786_10168,0 days 00:01:50.593000
normal_cluster,application_1758535431786_10173,0 days 00:00:53.889000
```

- `appMasterHost`: Cluster where the application executed
- `appId`: Application identifier
- `runtime`: Execution time in format "X days HH:MM:SS.microseconds"

**task.csv:**

```csv
task_id,label,feat
0,application_1758535431786_10168,"[5841285, 4224]"
1,application_1758535431786_10173,"[169088, 82]"
```

- `task_id`: Numeric task identifier
- `label`: Application ID (matches `appId` in runtime.csv)
- `feat`: Task workload characteristics

**Graph Structure:**

- **Nodes**: Clusters, Tasks, Runtime instances
- **Edges**: cluster→runtime, runtime→task, and reverse edges
- Runtime instances connect tasks to the clusters where they executed

---

### Usage

Run the training script with hyperparameters:

```bash
python main.py <seed> <n_hidden> <n_layers> <dropout>
```

**Example:**

```bash
python main.py 42 64 2 0.2
```

Results will be saved in the `results/` directory.

