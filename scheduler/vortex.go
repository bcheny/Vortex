package vortex

import (
	"context"
	"fmt"
	"math"
	"time"

	"k8s.io/klog/v2"

	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/framework"
	"volcano.sh/volcano/pkg/scheduler/plugins/util"
)

const (
	// PluginName indicates name of volcano scheduler plugin
	PluginName = "vortex"

	// Weight parameter balancing completion time and migration cost
	AlphaWeight = "vortex.alpha"
	
	// Gravity model parameters
	GravityScalingFactor = "vortex.gravity.scaling"
	GravityDecayExponent = "vortex.gravity.decay"
	
	// NSGA-II parameters
	PopulationSize = "vortex.nsga.population"
	MaxGenerations = "vortex.nsga.generations"
	
	// Time slot duration in minutes
	TimeSlotDuration = "vortex.timeslot.duration"
)

// vortexPlugin implements the Vortex scheduling algorithm
type vortexPlugin struct {
	pluginArguments framework.Arguments
	
	// Spatio-temporal resource model
	spatioTemporal *SpatioTemporalModel
	
	// HGNN-based JET predictor
	jetPredictor *JETPredictor
	
	// Data gravity calculator
	gravityModel *DataGravityModel
	
	// Multi-objective optimizer
	optimizer *STNSGAIIOptimizer
	
	// Configuration parameters
	alpha              float64
	gravityScaling     float64
	gravityDecay       float64
	populationSize     int
	maxGenerations     int
	timeSlotDuration   time.Duration
}

// New returns a new Vortex plugin instance
func New(arguments framework.Arguments) framework.Plugin {
	vp := &vortexPlugin{
		pluginArguments: arguments,
		alpha:           0.5, // Default: equal weight to JCT and migration cost
		gravityScaling:  1.0,
		gravityDecay:    2.0,
		populationSize:  50,
		maxGenerations:  100,
		timeSlotDuration: 60 * time.Minute,
	}
	
	// Parse configuration arguments
	arguments.GetFloat64(&vp.alpha, AlphaWeight)
	arguments.GetFloat64(&vp.gravityScaling, GravityScalingFactor)
	arguments.GetFloat64(&vp.gravityDecay, GravityDecayExponent)
	arguments.GetInt(&vp.populationSize, PopulationSize)
	arguments.GetInt(&vp.maxGenerations, MaxGenerations)
	
	var duration int
	if arguments.GetInt(&duration, TimeSlotDuration) {
		vp.timeSlotDuration = time.Duration(duration) * time.Minute
	}
	
	return vp
}

func (vp *vortexPlugin) Name() string {
	return PluginName
}

func (vp *vortexPlugin) OnSessionOpen(ssn *framework.Session) {
	klog.V(4).Infof("Vortex plugin session opened")
	
	// Initialize spatio-temporal resource model
	vp.spatioTemporal = NewSpatioTemporalModel(ssn, vp.timeSlotDuration)
	
	// Initialize JET predictor
	vp.jetPredictor = NewJETPredictor(ssn)
	
	// Initialize data gravity model
	vp.gravityModel = NewDataGravityModel(
		vp.gravityScaling,
		vp.gravityDecay,
		vp.spatioTemporal,
	)
	
	// Initialize multi-objective optimizer
	vp.optimizer = NewSTNSGAIIOptimizer(
		vp.populationSize,
		vp.maxGenerations,
		vp.alpha,
		vp.jetPredictor,
		vp.gravityModel,
		vp.spatioTemporal,
	)
	
	// Register job ordering function based on gravity and predicted completion time
	jobOrderFn := func(l, r interface{}) int {
		lv := l.(*api.JobInfo)
		rv := r.(*api.JobInfo)
		
		// Calculate scores for both jobs
		lScore := vp.calculateJobScore(ssn, lv)
		rScore := vp.calculateJobScore(ssn, rv)
		
		klog.V(4).Infof("Vortex JobOrderFn: <%v/%v> score: %f, <%v/%v> score: %f",
			lv.Namespace, lv.Name, lScore, rv.Namespace, rv.Name, rScore)
		
		if math.Abs(lScore-rScore) < 1e-6 {
			return 0
		}
		
		// Higher score = higher priority
		if lScore > rScore {
			return -1
		}
		return 1
	}
	ssn.AddJobOrderFn(vp.Name(), jobOrderFn)
	
	// Register node ordering function based on data gravity
	nodeOrderFn := func(task *api.TaskInfo, node *api.NodeInfo) (float64, error) {
		job := ssn.Jobs[task.Job]
		if job == nil {
			return 0, fmt.Errorf("job not found for task %s/%s", task.Namespace, task.Name)
		}
		
		// Calculate gravity strength for this task-node pair
		gravityStrength := vp.gravityModel.CalculateGravityStrength(
			task, node, job, ssn,
		)
		
		// Predict execution time
		predictedJET, confidence := vp.jetPredictor.PredictJET(task, node, ssn)
		
		// Calculate resource utilization score
		utilizationScore := vp.calculateResourceUtilization(task, node)
		
		// Combined score: balance gravity, predicted performance, and utilization
		// Higher gravity = better data locality
		// Lower predicted time = better performance
		// Higher utilization = better resource efficiency
		score := gravityStrength * 100.0 - predictedJET/60.0 + utilizationScore*10.0
		
		// Penalize low-confidence predictions
		score *= confidence
		
		klog.V(5).Infof("Vortex NodeOrder: task <%s/%s> on node <%s>: "+
			"gravity=%.2f, JET=%.2fs, util=%.2f, confidence=%.2f, score=%.2f",
			task.Namespace, task.Name, node.Name,
			gravityStrength, predictedJET, utilizationScore, confidence, score)
		
		return score, nil
	}
	ssn.AddNodeOrderFn(vp.Name(), nodeOrderFn)
	
	// Register preemptable function considering data gravity
	preemptableFn := func(preemptor *api.TaskInfo, preemptees []*api.TaskInfo) ([]*api.TaskInfo, int) {
		preemptorJob := ssn.Jobs[preemptor.Job]
		if preemptorJob == nil {
			return nil, util.Reject
		}
		
		var victims []*api.TaskInfo
		preemptorScore := vp.calculateJobScore(ssn, preemptorJob)
		
		for _, preemptee := range preemptees {
			preempteeJob := ssn.Jobs[preemptee.Job]
			if preempteeJob == nil {
				continue
			}
			
			preempteeScore := vp.calculateJobScore(ssn, preempteeJob)
			
			// Allow preemption if preemptor has higher score
			// and considering data migration cost
			migrationCost := vp.gravityModel.CalculateMigrationCost(
				preemptee, preemptorJob,
			)
			
			// Only preempt if benefit outweighs migration cost
			if preemptorScore > preempteeScore && migrationCost < 1000.0 {
				victims = append(victims, preemptee)
			}
		}
		
		klog.V(4).Infof("Vortex preemption: %d victims selected", len(victims))
		return victims, util.Permit
	}
	ssn.AddPreemptableFn(vp.Name(), preemptableFn)
	
	// Register job ready function considering temporal availability
	jobReadyFn := func(obj interface{}) bool {
		ji := obj.(*api.JobInfo)
		
		// Check if job meets minimum availability requirements
		if !ji.IsReady() {
			return false
		}
		
		// Check if suitable nodes are available in current time window
		availableNodes := vp.spatioTemporal.GetAvailableNodes(time.Now())
		if len(availableNodes) == 0 {
			klog.V(4).Infof("Job <%s/%s> not ready: no available nodes in current time window",
				ji.Namespace, ji.Name)
			return false
		}
		
		return true
	}
	ssn.AddJobReadyFn(vp.Name(), jobReadyFn)
	
	// Register event handlers for maintaining state
	ssn.AddEventHandler(&framework.EventHandler{
		AllocateFunc: func(event *framework.Event) {
			// Update spatio-temporal model with allocation
			vp.spatioTemporal.UpdateAllocation(event.Task, event.Task.NodeName, true)
			
			// Update predictor with actual execution data
			vp.jetPredictor.RecordExecution(event.Task, event.Task.NodeName)
			
			klog.V(4).Infof("Vortex allocated task <%s/%s> to node <%s>",
				event.Task.Namespace, event.Task.Name, event.Task.NodeName)
		},
		DeallocateFunc: func(event *framework.Event) {
			// Update spatio-temporal model with deallocation
			vp.spatioTemporal.UpdateAllocation(event.Task, event.Task.NodeName, false)
			
			klog.V(4).Infof("Vortex deallocated task <%s/%s> from node <%s>",
				event.Task.Namespace, event.Task.Name, event.Task.NodeName)
		},
	})
}

// calculateJobScore computes a composite score for job priority
func (vp *vortexPlugin) calculateJobScore(ssn *framework.Session, job *api.JobInfo) float64 {
	if job == nil {
		return 0.0
	}
	
	// Base score from job priority
	score := float64(job.Priority)
	
	// Factor in data locality - jobs with better data locality get higher scores
	var totalGravity float64
	taskCount := 0
	for _, task := range job.Tasks {
		for _, node := range ssn.Nodes {
			gravity := vp.gravityModel.CalculateGravityStrength(task, node, job, ssn)
			if gravity > totalGravity {
				totalGravity = gravity
			}
		}
		taskCount++
	}
	if taskCount > 0 {
		score += totalGravity * 10.0
	}
	
	// Factor in predicted execution time - jobs with shorter predicted time get higher scores
	var totalPredictedTime float64
	predictionCount := 0
	for _, task := range job.Tasks {
		if task.Status == api.Pending {
			for _, node := range ssn.Nodes {
				predictedTime, _ := vp.jetPredictor.PredictJET(task, node, ssn)
				if predictedTime > 0 {
					totalPredictedTime += predictedTime
					predictionCount++
					break
				}
			}
		}
	}
	if predictionCount > 0 {
		avgPredictedTime := totalPredictedTime / float64(predictionCount)
		// Invert so shorter time = higher score
		score += 1000.0 / (avgPredictedTime + 1.0)
	}
	
	// Factor in resource demand matching
	score += vp.calculateResourceMatchingScore(job, ssn)
	
	return score
}

// calculateResourceUtilization computes how well a task matches node resources
func (vp *vortexPlugin) calculateResourceUtilization(task *api.TaskInfo, node *api.NodeInfo) float64 {
	if node.Idle.MilliCPU <= 0 || node.Idle.Memory <= 0 {
		return 0.0
	}
	
	// Calculate CPU utilization
	cpuUtil := float64(task.Resreq.MilliCPU) / float64(node.Idle.MilliCPU)
	
	// Calculate memory utilization
	memUtil := float64(task.Resreq.Memory) / float64(node.Idle.Memory)
	
	// Prefer balanced utilization (around 0.7-0.8)
	targetUtil := 0.75
	cpuScore := 1.0 - math.Abs(cpuUtil-targetUtil)
	memScore := 1.0 - math.Abs(memUtil-targetUtil)
	
	// Average the scores
	return (cpuScore + memScore) / 2.0
}

// calculateResourceMatchingScore evaluates how well job requirements match available resources
func (vp *vortexPlugin) calculateResourceMatchingScore(job *api.JobInfo, ssn *framework.Session) float64 {
	var matchScore float64
	
	// Check GPU requirements
	needsGPU := false
	for _, task := range job.Tasks {
		if task.Resreq.ScalarResources != nil {
			if gpu, ok := task.Resreq.ScalarResources["nvidia.com/gpu"]; ok && gpu > 0 {
				needsGPU = true
				break
			}
		}
	}
	
	// Count available GPU nodes
	gpuNodes := 0
	for _, node := range ssn.Nodes {
		if node.Allocatable.ScalarResources != nil {
			if gpu, ok := node.Allocatable.ScalarResources["nvidia.com/gpu"]; ok && gpu > 0 {
				gpuNodes++
			}
		}
	}
	
	// Reward good matching
	if needsGPU && gpuNodes > 0 {
		matchScore += 20.0
	} else if !needsGPU && gpuNodes == 0 {
		matchScore += 10.0
	}
	
	return matchScore
}

func (vp *vortexPlugin) OnSessionClose(ssn *framework.Session) {
	klog.V(4).Infof("Vortex plugin session closed")
	
	// Cleanup and persist any necessary state
	if vp.jetPredictor != nil {
		vp.jetPredictor.Persist()
	}
}
