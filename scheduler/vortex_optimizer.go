package vortex

import (
	"math"
	"math/rand"
	"sort"
	"time"

	"k8s.io/klog/v2"

	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/framework"
)

// STNSGAIIOptimizer implements Spatio-Temporal NSGA-II optimization
type STNSGAIIOptimizer struct {
	populationSize int
	maxGenerations int
	alpha          float64 // Weight for JCT vs migration cost
	
	jetPredictor   *JETPredictor
	gravityModel   *DataGravityModel
	spatioTemporal *SpatioTemporalModel
	
	random *rand.Rand
}

// Individual represents a solution in the population
type Individual struct {
	// 4-tuple encoding: TaskAssignment, TimeSlot, ResourceAllocation, DataPlacement
	taskAssignment map[string]string    // task ID -> node name
	timeSlot       map[string]int       // task ID -> time slot index
	dataPlacement  map[string]string    // data ID -> node name
	
	// Objective values
	makespan       float64  // Maximum job completion time
	migrationCost  float64  // Total data migration cost
	
	// NSGA-II specific
	rank           int      // Pareto rank
	crowdingDist   float64  // Crowding distance
	dominatedCount int      // Number of solutions that dominate this
	dominates      []int    // Indices of solutions this dominates
}

// NewSTNSGAIIOptimizer creates a new optimizer
func NewSTNSGAIIOptimizer(
	popSize, maxGen int,
	alpha float64,
	predictor *JETPredictor,
	gravity *DataGravityModel,
	stm *SpatioTemporalModel,
) *STNSGAIIOptimizer {
	return &STNSGAIIOptimizer{
		populationSize: popSize,
		maxGenerations: maxGen,
		alpha:          alpha,
		jetPredictor:   predictor,
		gravityModel:   gravity,
		spatioTemporal: stm,
		random:         rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// Optimize runs the ST-NSGA-II algorithm
func (opt *STNSGAIIOptimizer) Optimize(ssn *framework.Session) *Individual {
	klog.V(4).Infof("Vortex ST-NSGA-II: Starting optimization with population=%d, generations=%d",
		opt.populationSize, opt.maxGenerations)
	
	// Initialize population
	population := opt.initializePopulation(ssn)
	
	// Evaluate initial population
	for i := range population {
		opt.evaluateIndividual(&population[i], ssn)
	}
	
	bestSolution := &population[0]
	
	// Evolution loop
	for gen := 0; gen < opt.maxGenerations; gen++ {
		// Generate offspring through selection, crossover, and mutation
		offspring := opt.generateOffspring(population, ssn)
		
		// Combine parent and offspring populations
		combined := append(population, offspring...)
		
		// Fast non-dominated sorting
		fronts := opt.fastNonDominatedSort(combined)
		
		// Calculate crowding distance for each front
		for _, front := range fronts {
			opt.calculateCrowdingDistance(combined, front)
		}
		
		// Select next generation
		population = opt.selectNextGeneration(combined, fronts)
		
		// Track best solution (lowest rank, highest crowding distance)
		for i := range population {
			if population[i].rank < bestSolution.rank ||
				(population[i].rank == bestSolution.rank && 
				 population[i].crowdingDist > bestSolution.crowdingDist) {
				bestSolution = &population[i]
			}
		}
		
		if gen%10 == 0 {
			klog.V(5).Infof("Vortex ST-NSGA-II: Generation %d, best makespan=%.2f, migration=%.2f",
				gen, bestSolution.makespan, bestSolution.migrationCost)
		}
	}
	
	// Apply TOPSIS to select final solution from Pareto front
	paretoFront := fronts[0]
	finalSolution := opt.applyTOPSIS(population, paretoFront, ssn)
	
	klog.V(4).Infof("Vortex ST-NSGA-II: Optimization complete, final makespan=%.2f, migration=%.2f",
		finalSolution.makespan, finalSolution.migrationCost)
	
	return finalSolution
}

// initializePopulation creates initial population
func (opt *STNSGAIIOptimizer) initializePopulation(ssn *framework.Session) []Individual {
	population := make([]Individual, opt.populationSize)
	
	// Collect all pending tasks
	var tasks []*api.TaskInfo
	for _, job := range ssn.Jobs {
		for _, task := range job.Tasks {
			if task.Status == api.Pending {
				tasks = append(tasks, task)
			}
		}
	}
	
	// Get available nodes
	nodes := make([]*api.NodeInfo, 0, len(ssn.Nodes))
	for _, node := range ssn.Nodes {
		nodes = append(nodes, node)
	}
	
	if len(nodes) == 0 || len(tasks) == 0 {
		return population
	}
	
	// Generate diverse initial solutions
	for i := range population {
		population[i] = Individual{
			taskAssignment: make(map[string]string),
			timeSlot:       make(map[string]int),
			dataPlacement:  make(map[string]string),
		}
		
		// Random assignment for diversity
		for _, task := range tasks {
			taskID := string(task.UID)
			
			// Assign to random node
			nodeIdx := opt.random.Intn(len(nodes))
			population[i].taskAssignment[taskID] = nodes[nodeIdx].Name
			
			// Assign to random time slot
			population[i].timeSlot[taskID] = opt.random.Intn(len(opt.spatioTemporal.timeSlots))
		}
		
		// For first few individuals, use heuristics for better starting point
		if i < opt.populationSize/10 {
			opt.applyGreedyHeuristic(&population[i], tasks, nodes, ssn)
		}
	}
	
	return population
}

// applyGreedyHeuristic applies greedy initialization
func (opt *STNSGAIIOptimizer) applyGreedyHeuristic(
	ind *Individual,
	tasks []*api.TaskInfo,
	nodes []*api.NodeInfo,
	ssn *framework.Session,
) {
	// Sort tasks by priority
	sort.Slice(tasks, func(i, j int) bool {
		return tasks[i].Priority > tasks[j].Priority
	})
	
	// Assign each task to best node based on gravity
	for _, task := range tasks {
		taskID := string(task.UID)
		job := ssn.Jobs[task.Job]
		
		bestNode := nodes[0].Name
		bestScore := -math.MaxFloat64
		
		for _, node := range nodes {
			gravity := opt.gravityModel.CalculateGravityStrength(task, node, job, ssn)
			jetTime, _ := opt.jetPredictor.PredictJET(task, node, ssn)
			
			score := gravity*10.0 - jetTime/60.0
			
			if score > bestScore {
				bestScore = score
				bestNode = node.Name
			}
		}
		
		ind.taskAssignment[taskID] = bestNode
		ind.timeSlot[taskID] = 0 // Start ASAP
	}
}

// evaluateIndividual computes objective values
func (opt *STNSGAIIOptimizer) evaluateIndividual(ind *Individual, ssn *framework.Session) {
	ind.makespan = 0
	ind.migrationCost = 0
	
	// Track completion time for each task
	taskCompletions := make(map[string]float64)
	
	for taskID, nodeName := range ind.taskAssignment {
		// Find task
		var task *api.TaskInfo
		var job *api.JobInfo
		for _, j := range ssn.Jobs {
			for _, t := range j.Tasks {
				if string(t.UID) == taskID {
					task = t
					job = j
					break
				}
			}
			if task != nil {
				break
			}
		}
		
		if task == nil || job == nil {
			continue
		}
		
		node := ssn.Nodes[nodeName]
		if node == nil {
			continue
		}
		
		// Calculate execution time
		jetTime, _ := opt.jetPredictor.PredictJET(task, node, ssn)
		
		// Calculate migration cost
		migrationCost := opt.gravityModel.CalculateMigrationCost(task, job)
		
		// Total completion time = queue time + execution time + migration time
		timeSlot := ind.timeSlot[taskID]
		queueTime := float64(timeSlot) * opt.spatioTemporal.timeSlotDuration.Seconds()
		
		completionTime := queueTime + jetTime + migrationCost
		taskCompletions[taskID] = completionTime
		
		// Update makespan
		if completionTime > ind.makespan {
			ind.makespan = completionTime
		}
		
		// Accumulate migration cost
		ind.migrationCost += migrationCost
	}
}

// generateOffspring creates offspring through genetic operators
func (opt *STNSGAIIOptimizer) generateOffspring(population []Individual, ssn *framework.Session) []Individual {
	offspring := make([]Individual, opt.populationSize)
	
	for i := 0; i < opt.populationSize; i++ {
		// Tournament selection
		parent1 := opt.tournamentSelection(population)
		parent2 := opt.tournamentSelection(population)
		
		// Spatio-temporal crossover
		child := opt.spatioTemporalCrossover(parent1, parent2, ssn)
		
		// Adaptive mutation
		opt.adaptiveMutation(&child, ssn)
		
		// Evaluate child
		opt.evaluateIndividual(&child, ssn)
		
		offspring[i] = child
	}
	
	return offspring
}

// tournamentSelection performs tournament selection
func (opt *STNSGAIIOptimizer) tournamentSelection(population []Individual) *Individual {
	tournamentSize := 3
	best := &population[opt.random.Intn(len(population))]
	
	for i := 1; i < tournamentSize; i++ {
		competitor := &population[opt.random.Intn(len(population))]
		
		// Compare: lower rank is better, higher crowding distance is better if same rank
		if competitor.rank < best.rank ||
			(competitor.rank == best.rank && competitor.crowdingDist > best.crowdingDist) {
			best = competitor
		}
	}
	
	return best
}

// spatioTemporalCrossover performs crossover preserving spatio-temporal constraints
func (opt *STNSGAIIOptimizer) spatioTemporalCrossover(p1, p2 *Individual, ssn *framework.Session) Individual {
	child := Individual{
		taskAssignment: make(map[string]string),
		timeSlot:       make(map[string]int),
		dataPlacement:  make(map[string]string),
	}
	
	// Two-point crossover on task assignments
	for taskID := range p1.taskAssignment {
		if opt.random.Float64() < 0.5 {
			child.taskAssignment[taskID] = p1.taskAssignment[taskID]
			child.timeSlot[taskID] = p1.timeSlot[taskID]
		} else {
			child.taskAssignment[taskID] = p2.taskAssignment[taskID]
			child.timeSlot[taskID] = p2.timeSlot[taskID]
		}
	}
	
	// Repair to ensure valid spatio-temporal assignment
	opt.repairSolution(&child, ssn)
	
	return child
}

// adaptiveMutation applies mutation based on solution quality
func (opt *STNSGAIIOptimizer) adaptiveMutation(ind *Individual, ssn *framework.Session) {
	// Base mutation rate
	baseMutationRate := 0.1
	
	// Increase mutation for poor solutions
	mutationRate := baseMutationRate
	if ind.makespan > 1000 || ind.migrationCost > 500 {
		mutationRate = 0.3
	}
	
	// Mutate task assignments
	for taskID := range ind.taskAssignment {
		if opt.random.Float64() < mutationRate {
			// Select random new node
			availableNodes := opt.spatioTemporal.GetAvailableNodes(time.Now())
			if len(availableNodes) > 0 {
				newNode := availableNodes[opt.random.Intn(len(availableNodes))]
				ind.taskAssignment[taskID] = newNode
			}
		}
		
		// Mutate time slot
		if opt.random.Float64() < mutationRate {
			ind.timeSlot[taskID] = opt.random.Intn(len(opt.spatioTemporal.timeSlots))
		}
	}
	
	opt.repairSolution(ind, ssn)
}

// repairSolution ensures solution validity
func (opt *STNSGAIIOptimizer) repairSolution(ind *Individual, ssn *framework.Session) {
	// Ensure all assignments respect resource constraints and temporal availability
	for taskID, nodeName := range ind.taskAssignment {
		node := ssn.Nodes[nodeName]
		if node == nil {
			// Assign to first available node
			for n := range ssn.Nodes {
				ind.taskAssignment[taskID] = n
				break
			}
		}
	}
}

// fastNonDominatedSort performs Pareto ranking
func (opt *STNSGAIIOptimizer) fastNonDominatedSort(population []Individual) [][]int {
	n := len(population)
	fronts := make([][]int, 0)
	
	// Initialize dominance relationships
	for i := range population {
		population[i].dominatedCount = 0
		population[i].dominates = make([]int, 0)
	}
	
	// Find dominance relationships
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if opt.dominates(&population[i], &population[j]) {
				population[i].dominates = append(population[i].dominates, j)
				population[j].dominatedCount++
			} else if opt.dominates(&population[j], &population[i]) {
				population[j].dominates = append(population[j].dominates, i)
				population[i].dominatedCount++
			}
		}
	}
	
	// Build fronts
	currentFront := make([]int, 0)
	for i := range population {
		if population[i].dominatedCount == 0 {
			population[i].rank = 0
			currentFront = append(currentFront, i)
		}
	}
	
	rank := 0
	for len(currentFront) > 0 {
		fronts = append(fronts, currentFront)
		nextFront := make([]int, 0)
		
		for _, i := range currentFront {
			for _, j := range population[i].dominates {
				population[j].dominatedCount--
				if population[j].dominatedCount == 0 {
					population[j].rank = rank + 1
					nextFront = append(nextFront, j)
				}
			}
		}
		
		currentFront = nextFront
		rank++
	}
	
	return fronts
}

// dominates checks if solution a dominates solution b
func (opt *STNSGAIIOptimizer) dominates(a, b *Individual) bool {
	// a dominates b if a is no worse in all objectives and better in at least one
	betterInOne := false
	
	if a.makespan < b.makespan {
		betterInOne = true
	} else if a.makespan > b.makespan {
		return false
	}
	
	if a.migrationCost < b.migrationCost {
		betterInOne = true
	} else if a.migrationCost > b.migrationCost {
		return false
	}
	
	return betterInOne
}

// calculateCrowdingDistance computes crowding distance for diversity
func (opt *STNSGAIIOptimizer) calculateCrowdingDistance(population []Individual, front []int) {
	if len(front) == 0 {
		return
	}
	
	// Initialize distances
	for _, i := range front {
		population[i].crowdingDist = 0
	}
	
	// Calculate for each objective
	for obj := 0; obj < 2; obj++ {
		// Sort by objective
		sort.Slice(front, func(i, j int) bool {
			if obj == 0 {
				return population[front[i]].makespan < population[front[j]].makespan
			}
			return population[front[i]].migrationCost < population[front[j]].migrationCost
		})
		
		// Boundary solutions get infinite distance
		population[front[0]].crowdingDist = math.MaxFloat64
		population[front[len(front)-1]].crowdingDist = math.MaxFloat64
		
		// Calculate normalized distance for others
		var objMin, objMax float64
		if obj == 0 {
			objMin = population[front[0]].makespan
			objMax = population[front[len(front)-1]].makespan
		} else {
			objMin = population[front[0]].migrationCost
			objMax = population[front[len(front)-1]].migrationCost
		}
		
		objRange := objMax - objMin
		if objRange > 0 {
			for i := 1; i < len(front)-1; i++ {
				var dist float64
				if obj == 0 {
					dist = (population[front[i+1]].makespan - population[front[i-1]].makespan) / objRange
				} else {
					dist = (population[front[i+1]].migrationCost - population[front[i-1]].migrationCost) / objRange
				}
				population[front[i]].crowdingDist += dist
			}
		}
	}
}

// selectNextGeneration selects individuals for next generation
func (opt *STNSGAIIOptimizer) selectNextGeneration(population []Individual, fronts [][]int) []Individual {
	nextGen := make([]Individual, 0, opt.populationSize)
	
	for _, front := range fronts {
		if len(nextGen)+len(front) <= opt.populationSize {
			// Add entire front
			for _, i := range front {
				nextGen = append(nextGen, population[i])
			}
		} else {
			// Add best individuals from this front based on crowding distance
			remaining := opt.populationSize - len(nextGen)
			
			// Sort by crowding distance
			sort.Slice(front, func(i, j int) bool {
				return population[front[i]].crowdingDist > population[front[j]].crowdingDist
			})
			
			for i := 0; i < remaining; i++ {
				nextGen = append(nextGen, population[front[i]])
			}
			break
		}
	}
	
	return nextGen
}

// applyTOPSIS selects best solution from Pareto front using TOPSIS
func (opt *STNSGAIIOptimizer) applyTOPSIS(population []Individual, front []int, ssn *framework.Session) *Individual {
	if len(front) == 0 {
		return &population[0]
	}
	
	// Dynamic weight based on system state
	wMakespan := opt.alpha
	wMigration := 1.0 - opt.alpha
	
	// Normalize objectives
	var minMakespan, maxMakespan, minMigration, maxMigration float64
	minMakespan, maxMakespan = math.MaxFloat64, -math.MaxFloat64
	minMigration, maxMigration = math.MaxFloat64, -math.MaxFloat64
	
	for _, i := range front {
		if population[i].makespan < minMakespan {
			minMakespan = population[i].makespan
		}
		if population[i].makespan > maxMakespan {
			maxMakespan = population[i].makespan
		}
		if population[i].migrationCost < minMigration {
			minMigration = population[i].migrationCost
		}
		if population[i].migrationCost > maxMigration {
			maxMigration = population[i].migrationCost
		}
	}
	
	// Calculate TOPSIS scores
	bestScore := -math.MaxFloat64
	var bestSolution *Individual
	
	for _, i := range front {
		// Normalize
		normMakespan := 0.0
		normMigration := 0.0
		
		if maxMakespan > minMakespan {
			normMakespan = (population[i].makespan - minMakespan) / (maxMakespan - minMakespan)
		}
		if maxMigration > minMigration {
			normMigration = (population[i].migrationCost - minMigration) / (maxMigration - minMigration)
		}
		
		// Weighted normalized values (lower is better for both objectives)
		vMakespan := wMakespan * normMakespan
		vMigration := wMigration * normMigration
		
		// Distance to ideal (0, 0)
		dPlus := math.Sqrt(vMakespan*vMakespan + vMigration*vMigration)
		
		// Distance to anti-ideal (1, 1)
		dMinus := math.Sqrt((wMakespan-vMakespan)*(wMakespan-vMakespan) +
			(wMigration-vMigration)*(wMigration-vMigration))
		
		// TOPSIS score (closer to 1 is better)
		score := dMinus / (dPlus + dMinus + 1e-10)
		
		if score > bestScore {
			bestScore = score
			bestSolution = &population[i]
		}
	}
	
	return bestSolution
}
