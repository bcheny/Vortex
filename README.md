# Vortex: Efficient Geo-Distributed Task Scheduling

This repository contains the source code for the paper:

> **Vortex: Efficient Geo-Distributed Task Scheduling with Spatio-Temporal Resource Awareness and Data Gravity Optimization**

**Vortex** addresses key challenges in existing schedulers by integrating spatio-temporal resource modeling, HGNN-based job execution time (JET) prediction, physics-inspired data gravity optimization, and multi-objective scheduling.

---

## Usage

### 1. Quick Start

Clone the repository:

```bash
git clone https://github.com/bcheny/Vortex.git
cd Vortex
```

### 2. HGNN-based Prediction

Job execution time [prediction](model/README.md) using heterogeneous graph neural networks:
```bash
cd model
```

### 3. Scheduling Process

[Deploy](scheduler/README.md) and run the Vortex scheduler:

```bash
cd scheduler
```

#### Installation

###### 1. Copy the Vortex plugin files to your Volcano scheduler plugins directory:

```bash
cp vortex.go $VOLCANO_ROOT/pkg/scheduler/plugins/vortex/
cp vortex_spatiotemporal.go $VOLCANO_ROOT/pkg/scheduler/plugins/vortex/
cp vortex_predictor_gravity.go $VOLCANO_ROOT/pkg/scheduler/plugins/vortex/
cp vortex_optimizer.go $VOLCANO_ROOT/pkg/scheduler/plugins/vortex/
```

###### 2. Register the plugin in the Volcano scheduler plugin registry.

###### 3. Rebuild the Volcano scheduler:

```bash
cd $VOLCANO_ROOT
make
```

#### Configuration

##### Basic Configuration

Add Vortex to your Volcano scheduler configuration:

```yaml
actions: "enqueue, allocate, backfill"
tiers:
  - plugins:
      - name: priority
      - name: gang
      - name: conformance
  - plugins:
      - name: drf
      - name: predicates
      - name: vortex
        arguments:
          # Weight between JCT and migration cost (0-1)
          vortex.alpha: 0.5
          
          # Gravity model parameters
          vortex.gravity.scaling: 1.0
          vortex.gravity.decay: 2.0
          
          # NSGA-II optimization parameters
          vortex.nsga.population: 50
          vortex.nsga.generations: 100
          
          # Time slot duration in minutes
          vortex.timeslot.duration: 60
      - name: nodeorder
      - name: binpack
```

##### Advanced Configuration

###### Node Availability Windows

Annotate nodes with availability windows:

```yaml
apiVersion: v1
kind: Node
metadata:
  name: edge-site-1
  annotations:
    vortex.scheduler/availability-window: "00:00-06:00"
  labels:
    topology.kubernetes.io/zone: "us-west-1"
spec:
  # ... node spec
```

###### Job Priority Configuration

Configure jobs with appropriate priorities:

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: data-processing-job
spec:
  schedulerName: volcano
  priorityClassName: high-priority
  queue: default
  minAvailable: 3
  tasks:
    - replicas: 3
      name: worker
      template:
        spec:
          containers:
            - name: processor
              image: myapp:latest
              resources:
                requests:
                  cpu: "2000m"
                  memory: "4Gi"
                limits:
                  cpu: "4000m"
                  memory: "8Gi"
```

#### Usage Examples

###### Example 1: CPU-Intensive Workload

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: wordcount-job
spec:
  schedulerName: volcano
  minAvailable: 5
  tasks:
    - replicas: 5
      name: map
      template:
        spec:
          containers:
            - name: mapper
              image: wordcount:v1
              resources:
                requests:
                  cpu: "4000m"
                  memory: "8Gi"
```

###### Example 2: GPU-Accelerated ML Training

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: ml-training
spec:
  schedulerName: volcano
  minAvailable: 2
  tasks:
    - replicas: 1
      name: trainer
      template:
        spec:
          containers:
            - name: trainer
              image: ml-trainer:v1
              resources:
                requests:
                  cpu: "8000m"
                  memory: "32Gi"
                  nvidia.com/gpu: "2"
                limits:
                  nvidia.com/gpu: "2"
```

###### Example 3: Geo-Distributed Analytics

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: distributed-analytics
  annotations:
    vortex.scheduler/data-locality: "required"
spec:
  schedulerName: volcano
  minAvailable: 10
  tasks:
    - replicas: 10
      name: analyzer
      template:
        spec:
          containers:
            - name: analyzer
              image: analytics:v1
              resources:
                requests:
                  cpu: "2000m"
                  memory: "16Gi"
          volumes:
            - name: data
              persistentVolumeClaim:
                claimName: analytics-data-pvc
```
