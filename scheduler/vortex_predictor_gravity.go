package vortex

import (
	"math"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/framework"
)

// JETPredictor implements HGNN-based job execution time prediction
type JETPredictor struct {
	// Historical execution records
	executionHistory map[string][]ExecutionRecord
	
	// Feature embeddings (simplified HGNN representation)
	jobEmbeddings      map[string][]float64
	resourceEmbeddings map[string][]float64
	
	// Model parameters
	weights      [][]float64
	biases       []float64
	embeddingDim int
	
	mutex sync.RWMutex
}

// ExecutionRecord stores historical execution data
type ExecutionRecord struct {
	TaskID       string
	JobID        string
	NodeName     string
	StartTime    time.Time
	EndTime      time.Time
	Duration     float64
	CPUUsage     float64
	MemoryUsage  float64
	ResourceType string
}

// NewJETPredictor creates a new JET predictor
func NewJETPredictor(ssn *framework.Session) *JETPredictor {
	predictor := &JETPredictor{
		executionHistory:   make(map[string][]ExecutionRecord),
		jobEmbeddings:      make(map[string][]float64),
		resourceEmbeddings: make(map[string][]float64),
		embeddingDim:       64,
	}
	
	// Initialize with simple neural network weights
	predictor.initializeModel()
	
	// Load historical data if available
	predictor.loadHistoricalData(ssn)
	
	return predictor
}

// initializeModel initializes the prediction model
func (jp *JETPredictor) initializeModel() {
	// Simple 2-layer network
	inputDim := jp.embeddingDim * 2 // job + resource embeddings
	hiddenDim := 32
	
	// Initialize weights with Xavier initialization
	jp.weights = make([][]float64, 2)
	jp.weights[0] = make([]float64, inputDim*hiddenDim)
	jp.weights[1] = make([]float64, hiddenDim)
	jp.biases = make([]float64, hiddenDim+1)
	
	scale := math.Sqrt(2.0 / float64(inputDim))
	for i := range jp.weights[0] {
		jp.weights[0][i] = (2.0*randomFloat() - 1.0) * scale
	}
	
	scale = math.Sqrt(2.0 / float64(hiddenDim))
	for i := range jp.weights[1] {
		jp.weights[1][i] = (2.0*randomFloat() - 1.0) * scale
	}
}

// randomFloat returns a random float in [0, 1)
func randomFloat() float64 {
	// Simplified random number generation
	return float64(time.Now().UnixNano()%1000) / 1000.0
}

// loadHistoricalData loads past execution records
func (jp *JETPredictor) loadHistoricalData(ssn *framework.Session) {
	// In practice, load from persistent storage
	// For now, initialize with defaults based on current jobs
	
	for _, job := range ssn.Jobs {
		jobKey := string(job.UID)
		jp.jobEmbeddings[jobKey] = jp.generateJobEmbedding(job)
	}
	
	for nodeName, node := range ssn.Nodes {
		jp.resourceEmbeddings[nodeName] = jp.generateResourceEmbedding(node)
	}
	
	klog.V(4).Infof("Vortex JETPredictor: Initialized with %d job embeddings, %d resource embeddings",
		len(jp.jobEmbeddings), len(jp.resourceEmbeddings))
}

// generateJobEmbedding creates an embedding vector for a job
func (jp *JETPredictor) generateJobEmbedding(job *api.JobInfo) []float64 {
	embedding := make([]float64, jp.embeddingDim)
	
	// Feature engineering: encode job characteristics
	// [0-15]: Resource requirements (normalized)
	totalCPU := 0.0
	totalMem := 0.0
	taskCount := float64(len(job.Tasks))
	
	for _, task := range job.Tasks {
		totalCPU += float64(task.Resreq.MilliCPU)
		totalMem += float64(task.Resreq.Memory)
	}
	
	embedding[0] = totalCPU / 100000.0  // Normalize
	embedding[1] = totalMem / 100000.0  // Normalize
	embedding[2] = taskCount / 100.0    // Normalize
	embedding[3] = float64(job.Priority) / 100.0
	
	// [4-7]: Job type indicators (simplified)
	if job.Name != "" {
		// Simple hash-based encoding
		hash := 0
		for _, c := range job.Name {
			hash = (hash*31 + int(c)) % 1000
		}
		embedding[4] = float64(hash) / 1000.0
	}
	
	// [8-63]: Random features (in practice, learned from HGNN)
	for i := 8; i < jp.embeddingDim; i++ {
		embedding[i] = (2.0*randomFloat() - 1.0) * 0.1
	}
	
	return embedding
}

// generateResourceEmbedding creates an embedding vector for a resource node
func (jp *JETPredictor) generateResourceEmbedding(node *api.NodeInfo) []float64 {
	embedding := make([]float64, jp.embeddingDim)
	
	// Feature engineering: encode node characteristics
	embedding[0] = float64(node.Allocatable.MilliCPU) / 100000.0
	embedding[1] = float64(node.Allocatable.Memory) / 100000.0
	embedding[2] = float64(node.Allocatable.Storage) / 1000000.0
	
	// GPU indicator
	if node.Allocatable.ScalarResources != nil {
		if gpu, ok := node.Allocatable.ScalarResources["nvidia.com/gpu"]; ok {
			embedding[3] = gpu / 10.0
		}
	}
	
	// Node state
	embedding[4] = float64(node.Idle.MilliCPU) / float64(node.Allocatable.MilliCPU+1)
	embedding[5] = float64(node.Idle.Memory) / float64(node.Allocatable.Memory+1)
	
	// Random features (in practice, learned from HGNN)
	for i := 6; i < jp.embeddingDim; i++ {
		embedding[i] = (2.0*randomFloat() - 1.0) * 0.1
	}
	
	return embedding
}

// PredictJET predicts job execution time for a task on a node
func (jp *JETPredictor) PredictJET(task *api.TaskInfo, node *api.NodeInfo, ssn *framework.Session) (float64, float64) {
	jp.mutex.RLock()
	defer jp.mutex.RUnlock()
	
	job := ssn.Jobs[task.Job]
	if job == nil {
		return 300.0, 0.5 // Default: 5 minutes, low confidence
	}
	
	// Get or create embeddings
	jobEmbed := jp.getJobEmbedding(string(job.UID), job)
	nodeEmbed := jp.getResourceEmbedding(node.Name, node)
	
	// Concatenate embeddings
	input := append(jobEmbed, nodeEmbed...)
	
	// Forward pass through network
	predicted := jp.forwardPass(input)
	
	// Calculate confidence based on historical data similarity
	confidence := jp.calculateConfidence(string(job.UID), node.Name)
	
	// Add base time estimates based on resource requirements
	baseTime := jp.estimateBaseTime(task, node)
	
	// Combine model prediction with base estimate
	finalPrediction := predicted*0.6 + baseTime*0.4
	
	klog.V(5).Infof("Vortex JETPredictor: task <%s/%s> on node <%s>: predicted=%.2fs, confidence=%.2f",
		task.Namespace, task.Name, node.Name, finalPrediction, confidence)
	
	return finalPrediction, confidence
}

// getJobEmbedding retrieves or generates job embedding
func (jp *JETPredictor) getJobEmbedding(jobID string, job *api.JobInfo) []float64 {
	if embed, ok := jp.jobEmbeddings[jobID]; ok {
		return embed
	}
	return jp.generateJobEmbedding(job)
}

// getResourceEmbedding retrieves or generates resource embedding
func (jp *JETPredictor) getResourceEmbedding(nodeName string, node *api.NodeInfo) []float64 {
	if embed, ok := jp.resourceEmbeddings[nodeName]; ok {
		return embed
	}
	return jp.generateResourceEmbedding(node)
}

// forwardPass performs neural network forward pass
func (jp *JETPredictor) forwardPass(input []float64) float64 {
	// First layer
	hiddenDim := len(jp.biases) - 1
	hidden := make([]float64, hiddenDim)
	
	for i := 0; i < hiddenDim; i++ {
		sum := jp.biases[i]
		for j := 0; j < len(input); j++ {
			if i*len(input)+j < len(jp.weights[0]) {
				sum += input[j] * jp.weights[0][i*len(input)+j]
			}
		}
		hidden[i] = relu(sum)
	}
	
	// Output layer
	output := jp.biases[hiddenDim]
	for i := 0; i < hiddenDim; i++ {
		if i < len(jp.weights[1]) {
			output += hidden[i] * jp.weights[1][i]
		}
	}
	
	// Apply activation and scale to reasonable time range
	return math.Max(10.0, math.Min(3600.0, relu(output)*10.0))
}

// relu activation function
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// estimateBaseTime provides baseline execution time estimate
func (jp *JETPredictor) estimateBaseTime(task *api.TaskInfo, node *api.NodeInfo) float64 {
	// Simple heuristic based on resource requirements
	cpuTime := float64(task.Resreq.MilliCPU) / float64(node.Allocatable.MilliCPU+1) * 100.0
	memTime := float64(task.Resreq.Memory) / float64(node.Allocatable.Memory+1) * 50.0
	
	return math.Max(cpuTime, memTime) + 30.0 // Add base overhead
}

// calculateConfidence estimates prediction confidence
func (jp *JETPredictor) calculateConfidence(jobID, nodeName string) float64 {
	// Check if we have historical data
	if records, ok := jp.executionHistory[jobID]; ok {
		for _, record := range records {
			if record.NodeName == nodeName {
				return 0.9 // High confidence with exact match
			}
		}
		if len(records) > 0 {
			return 0.7 // Medium confidence with similar job
		}
	}
	return 0.5 // Low confidence for new job
}

// RecordExecution records actual execution for model training
func (jp *JETPredictor) RecordExecution(task *api.TaskInfo, nodeName string) {
	jp.mutex.Lock()
	defer jp.mutex.Unlock()
	
	record := ExecutionRecord{
		TaskID:    string(task.UID),
		JobID:     string(task.Job),
		NodeName:  nodeName,
		StartTime: time.Now(),
	}
	
	jobID := string(task.Job)
	jp.executionHistory[jobID] = append(jp.executionHistory[jobID], record)
}

// Persist saves model state (placeholder)
func (jp *JETPredictor) Persist() {
	klog.V(4).Info("Vortex JETPredictor: Persisting model state")
	// In practice, save to database or file
}

// DataGravityModel implements physics-inspired data gravity calculation
type DataGravityModel struct {
	gravityScaling float64
	gravityDecay   float64
	spatioTemporal *SpatioTemporalModel
	
	// Data location tracking
	dataLocations map[string]string // data ID -> node name
	dataSize      map[string]float64 // data ID -> size in GB
	
	mutex sync.RWMutex
}

// NewDataGravityModel creates a new data gravity model
func NewDataGravityModel(scaling, decay float64, stm *SpatioTemporalModel) *DataGravityModel {
	return &DataGravityModel{
		gravityScaling: scaling,
		gravityDecay:   decay,
		spatioTemporal: stm,
		dataLocations:  make(map[string]string),
		dataSize:       make(map[string]float64),
	}
}

// CalculateGravityStrength computes gravitational attraction between task and node
func (dgm *DataGravityModel) CalculateGravityStrength(
	task *api.TaskInfo,
	node *api.NodeInfo,
	job *api.JobInfo,
	ssn *framework.Session,
) float64 {
	// Data mass (S): sum of input data sizes
	dataMass := dgm.calculateDataMass(task, job)
	
	// Compute mass (C): available computational capacity
	computeMass := dgm.calculateComputeMass(node)
	
	// Network distance (D): transfer cost considering latency and bandwidth
	networkDistance := dgm.calculateNetworkDistance(task, node, job, ssn)
	
	// Apply gravity formula: F = G * (S * C) / D^k
	if networkDistance < 0.1 {
		networkDistance = 0.1 // Avoid division by zero
	}
	
	gravity := dgm.gravityScaling * (dataMass * computeMass) / 
		math.Pow(networkDistance, dgm.gravityDecay)
	
	// Apply temporal and compatibility modulation
	temporalFactor := dgm.calculateTemporalFactor(node)
	compatibilityFactor := dgm.calculateCompatibilityFactor(task, node)
	
	effectiveGravity := gravity * temporalFactor * compatibilityFactor
	
	return effectiveGravity
}

// calculateDataMass computes data mass for a task
func (dgm *DataGravityModel) calculateDataMass(task *api.TaskInfo, job *api.JobInfo) float64 {
	dgm.mutex.RLock()
	defer dgm.mutex.RUnlock()
	
	// Estimate data size from task and job characteristics
	// In practice, parse from task annotations or PVC sizes
	mass := 1.0 // Base mass
	
	// Check for volumes
	if task.Pod != nil {
		for _, vol := range task.Pod.Spec.Volumes {
			if vol.PersistentVolumeClaim != nil {
				// Add PVC size (simplified)
				mass += 10.0 // Assume 10 GB per PVC
			}
		}
	}
	
	// Add mass based on task resource requirements (proxy for data)
	mass += float64(task.Resreq.Memory) / 10000.0
	
	return mass
}

// calculateComputeMass computes computational capacity mass
func (dgm *DataGravityModel) calculateComputeMass(node *api.NodeInfo) float64 {
	// Weighted sum of available resources
	wCPU := 1.0
	wGPU := 3.0
	wMem := 0.5
	wNet := 0.5
	
	cpuMass := float64(node.Idle.MilliCPU) / 1000.0
	memMass := float64(node.Idle.Memory) / 10000.0
	
	gpuMass := 0.0
	if node.Idle.ScalarResources != nil {
		if gpu, ok := node.Idle.ScalarResources["nvidia.com/gpu"]; ok {
			gpuMass = gpu
		}
	}
	
	netMass := 1.0 // Simplified network capacity
	
	return wCPU*cpuMass + wGPU*gpuMass + wMem*memMass + wNet*netMass
}

// calculateNetworkDistance computes effective network distance
func (dgm *DataGravityModel) calculateNetworkDistance(
	task *api.TaskInfo,
	node *api.NodeInfo,
	job *api.JobInfo,
	ssn *framework.Session,
) float64 {
	// Find where task's data currently resides
	dataNodes := dgm.findDataLocations(task, job, ssn)
	
	if len(dataNodes) == 0 {
		return 1.0 // Local data
	}
	
	// Calculate minimum transfer cost across all data sources
	minDistance := math.MaxFloat64
	
	for _, dataNode := range dataNodes {
		latency := dgm.spatioTemporal.GetLatency(dataNode, node.Name)
		bandwidth := dgm.spatioTemporal.GetBandwidth(dataNode, node.Name)
		
		// Distance combines latency and bandwidth constraints
		// Higher latency = farther, lower bandwidth = farther
		distance := latency + 1000.0/bandwidth
		
		if distance < minDistance {
			minDistance = distance
		}
	}
	
	return minDistance
}

// findDataLocations identifies where task data is located
func (dgm *DataGravityModel) findDataLocations(task *api.TaskInfo, job *api.JobInfo, ssn *framework.Session) []string {
	locations := make([]string, 0)
	
	// Check if this is a follow-up task in a workflow
	// Look for predecessor tasks in the same job
	for _, t := range job.Tasks {
		if t.UID != task.UID && t.Status == api.Succeeded {
			if t.NodeName != "" {
				locations = append(locations, t.NodeName)
			}
		}
	}
	
	// If no predecessor, data is likely distributed or needs to be fetched
	if len(locations) == 0 {
		// Assume data is on all nodes (worst case for gravity)
		for nodeName := range ssn.Nodes {
			locations = append(locations, nodeName)
			break // Use just one for simplicity
		}
	}
	
	return locations
}

// calculateTemporalFactor applies temporal availability constraints
func (dgm *DataGravityModel) calculateTemporalFactor(node *api.NodeInfo) float64 {
	// Check if node is in its availability window
	availableNodes := dgm.spatioTemporal.GetAvailableNodes(time.Now())
	
	for _, availNode := range availableNodes {
		if availNode == node.Name {
			return 1.0 // Fully available
		}
	}
	
	// Node not currently available - apply penalty
	return 0.1
}

// calculateCompatibilityFactor measures resource type compatibility
func (dgm *DataGravityModel) calculateCompatibilityFactor(task *api.TaskInfo, node *api.NodeInfo) float64 {
	compatibility := 1.0
	
	// Check GPU requirements
	needsGPU := false
	if task.Resreq.ScalarResources != nil {
		if gpu, ok := task.Resreq.ScalarResources["nvidia.com/gpu"]; ok && gpu > 0 {
			needsGPU = true
		}
	}
	
	hasGPU := false
	if node.Allocatable.ScalarResources != nil {
		if gpu, ok := node.Allocatable.ScalarResources["nvidia.com/gpu"]; ok && gpu > 0 {
			hasGPU = true
		}
	}
	
	// Perfect match
	if needsGPU == hasGPU {
		compatibility = 1.0
	} else if needsGPU && !hasGPU {
		// Task needs GPU but node doesn't have it - major penalty
		compatibility = 0.1
	} else {
		// Task doesn't need GPU but node has it - minor penalty for waste
		compatibility = 0.8
	}
	
	return compatibility
}

// CalculateMigrationCost estimates data migration cost
func (dgm *DataGravityModel) CalculateMigrationCost(task *api.TaskInfo, job *api.JobInfo) float64 {
	// Estimate migration cost based on data size and network constraints
	dataMass := dgm.calculateDataMass(task, job)
	
	// Assume average bandwidth of 1 Gbps = 125 MB/s
	avgBandwidth := 125.0 // MB/s
	avgLatency := 50.0    // ms
	
	// Cost = transfer time + latency overhead
	transferTime := (dataMass * 1024.0) / avgBandwidth // seconds
	cost := transferTime + avgLatency/1000.0
	
	return cost
}
