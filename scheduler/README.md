## Scheduling Process

This directory contains deployment procedures for multi-cluster environments and implementations of scheduling algorithms.

### Overview

This project implements two typical computational tasks—WordCount and PageRank—in a multi-cluster environment. We run these tasks using the Volcano scheduling framework and evaluate their execution time and utilization performance under different scheduling algorithm configurations.

---

### Deploy Steps

#### 1. Deploy the Karmada

`Karmada` Version: **v1.15.0**

##### Prerequisites

- [Go](https://golang.org/) version v1.18+
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) version v1.19+
- [kind](https://kind.sigs.k8s.io/) version v0.14.0+

##### Install the Karmada control plane

###### 1. Clone this repo to your machine:

```text
git clone https://github.com/karmada-io/karmada
```

###### 2. Change to the karmada directory:

```text
cd karmada
```

###### 3. Deploy and run Karmada control plane:

```text
hack/local-up-karmada.sh
```

#### 2. Deploy the Volcano to member clusters

`Volcano` Version: **1.12.0**

Install `Volcano` to all member cluster:

```
# Switch to the member clusters, you need install the Volcano to the all member cluster.
export KUBECONFIG=$HOME/.kube/members.config

# Deploy Volcano to the member clusters.
kubectl --context member1 apply -f https://raw.githubusercontent.com/volcano-sh/volcano/release-1.12/installer/volcano-development.yaml
kubectl --context member2 apply -f https://raw.githubusercontent.com/volcano-sh/volcano/release-1.12/installer/volcano-development.yaml
kubectl --context member3 apply -f https://raw.githubusercontent.com/volcano-sh/volcano/release-1.12/installer/volcano-development.yaml
```

#### 3. View and Switch Scheduling Algorithms

###### 1. Edit the `configmap` file and locate the `volcano-scheduler.conf` field:

    kubectl -n volcano-system edit configmap volcano-scheduler-configmap

###### 2. After making the changes, restart the volcano-scheduler:

    kubectl rollout restart deployment volcano-scheduler -n volcano-system

###### 3. View the current scheduling algorithm:

    kubectl logs -n volcano-system deploy/volcano-scheduler | grep plugin

#### 4. Run Jobs

###### 1. Run the WordCount Job:

    cd wordcount/deploy
    kubectl apply -f wordcount-job.yaml

###### 2. Run the PageRank Job:

    cd pagerank/deploy
    kubectl apply -f pagerank.yaml

###### 3. Check job execution status:

    kubectl get vcjob -n default
