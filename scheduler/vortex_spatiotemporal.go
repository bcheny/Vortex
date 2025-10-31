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

// SpatioTemporalModel represents the three-dimensional scheduling space
type SpatioTemporalModel struct {
	// Geographic topology
	latencyMatrix   map[string]map[string]float64  // node-to-node latency
	bandwidthMatrix map[string]map[string]float64  // node-to-node bandwidth
	nodeLocations   map[string]Location             // node coordinates
	
	// Temporal dimension
	timeSlots         []TimeSlot
	nodeAvailability  map[string][]bool               // node availability by time slot
	timeSlotDuration  time.Duration
	
	// Resource capability
	resourceTensor    map[string]map[v1.ResourceName][]float64  // node -> resource -> time slots
	heterogeneityIndex map[string]float64                       // node heterogeneity score
	
	// Current state
	currentAllocations map[string][]*api.TaskInfo  // node -> tasks
	mutex              sync.RWMutex
}

// Location represents geographic coordinates
type Location struct {
	Latitude  float64
	Longitude float64
	Zone      string  // e.g., "us-west", "eu-central"
}

// TimeSlot represents a discrete time interval
type TimeSlot struct {
	Start time.Time
	End   time.Time
	Index int
}

// NewSpatioTemporalModel creates a new spatio-temporal resource model
func NewSpatioTemporalModel(ssn *framework.Session, slotDuration time.Duration) *SpatioTemporalModel {
	stm := &SpatioTemporalModel{
		latencyMatrix:      make(map[string]map[string]float64),
		bandwidthMatrix:    make(map[string]map[string]float64),
		nodeLocations:      make(map[string]Location),
		timeSlots:          make([]TimeSlot, 0),
		nodeAvailability:   make(map[string][]bool),
		timeSlotDuration:   slotDuration,
		resourceTensor:     make(map[string]map[v1.ResourceName][]float64),
		heterogeneityIndex: make(map[string]float64),
		currentAllocations: make(map[string][]*api.TaskInfo),
	}
	
	stm.initializeGeographicTopology(ssn)
	stm.initializeTemporalDimension()
	stm.initializeResourceCapability(ssn)
	
	return stm
}

// initializeGeographicTopology builds the network topology model
func (stm *SpatioTemporalModel) initializeGeographicTopology(ssn *framework.Session) {
	// Extract location from node labels or annotations
	for nodeName, node := range ssn.Nodes {
		// Try to get location from labels
		loc := stm.extractLocation(node.Node)
		stm.nodeLocations[nodeName] = loc
		
		// Initialize matrices
		stm.latencyMatrix[nodeName] = make(map[string]float64)
		stm.bandwidthMatrix[nodeName] = make(map[string]float64)
	}
	
	// Calculate inter-node latency and bandwidth
	for node1 := range ssn.Nodes {
		for node2 := range ssn.Nodes {
			if node1 == node2 {
				stm.latencyMatrix[node1][node2] = 0
				stm.bandwidthMatrix[node1][node2] = math.MaxFloat64 // Infinite local bandwidth
			} else {
				// Calculate distance-based latency
				distance := stm.calculateDistance(
					stm.nodeLocations[node1],
					stm.nodeLocations[node2],
				)
				
				// Latency model: base latency + distance factor
				stm.latencyMatrix[node1][node2] = 5.0 + distance*0.1 // ms
				
				// Bandwidth model: inverse to distance (simplified)
				// Typical inter-DC bandwidth: 1-10 Gbps
				if distance < 100 {
					stm.bandwidthMatrix[node1][node2] = 10000.0 // 10 Gbps in MB/s
				} else if distance < 1000 {
					stm.bandwidthMatrix[node1][node2] = 5000.0 // 5 Gbps
				} else {
					stm.bandwidthMatrix[node1][node2] = 1000.0 // 1 Gbps
				}
			}
		}
	}
	
	klog.V(4).Infof("Vortex: Initialized geographic topology for %d nodes", len(ssn.Nodes))
}

// extractLocation extracts location information from node
func (stm *SpatioTemporalModel) extractLocation(node *v1.Node) Location {
	loc := Location{
		Zone: "unknown",
	}
	
	// Try to get zone from standard labels
	if zone, ok := node.Labels["topology.kubernetes.io/zone"]; ok {
		loc.Zone = zone
	} else if zone, ok := node.Labels["failure-domain.beta.kubernetes.io/zone"]; ok {
		loc.Zone = zone
	}
	
	// Map zones to approximate coordinates (simplified)
	zoneCoordinates := map[string]struct{ lat, lon float64 }{
		"us-west-1":     {37.77, -122.42},  // San Francisco
		"us-east-1":     {38.90, -77.04},   // Virginia
		"eu-central-1":  {50.11, 8.68},     // Frankfurt
		"ap-northeast-1": {35.68, 139.69},  // Tokyo
		"ap-south-1":    {19.07, 72.87},    // Mumbai
	}
	
	if coords, ok := zoneCoordinates[loc.Zone]; ok {
		loc.Latitude = coords.lat
		loc.Longitude = coords.lon
	}
	
	return loc
}

// calculateDistance computes geographic distance between two locations
func (stm *SpatioTemporalModel) calculateDistance(loc1, loc2 Location) float64 {
	// Haversine formula for great circle distance
	const earthRadius = 6371.0 // km
	
	lat1 := loc1.Latitude * math.Pi / 180.0
	lat2 := loc2.Latitude * math.Pi / 180.0
	deltaLat := (loc2.Latitude - loc1.Latitude) * math.Pi / 180.0
	deltaLon := (loc2.Longitude - loc1.Longitude) * math.Pi / 180.0
	
	a := math.Sin(deltaLat/2)*math.Sin(deltaLat/2) +
		math.Cos(lat1)*math.Cos(lat2)*
			math.Sin(deltaLon/2)*math.Sin(deltaLon/2)
	
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
	
	return earthRadius * c
}

// initializeTemporalDimension sets up time slots and availability windows
func (stm *SpatioTemporalModel) initializeTemporalDimension() {
	// Create time slots for the next 24 hours
	now := time.Now()
	startTime := now.Truncate(stm.timeSlotDuration)
	
	numSlots := int(24 * time.Hour / stm.timeSlotDuration)
	for i := 0; i < numSlots; i++ {
		slot := TimeSlot{
			Start: startTime.Add(time.Duration(i) * stm.timeSlotDuration),
			End:   startTime.Add(time.Duration(i+1) * stm.timeSlotDuration),
			Index: i,
		}
		stm.timeSlots = append(stm.timeSlots, slot)
	}
	
	klog.V(4).Infof("Vortex: Initialized %d time slots of %v duration", numSlots, stm.timeSlotDuration)
}

// initializeResourceCapability builds the resource capability tensor
func (stm *SpatioTemporalModel) initializeResourceCapability(ssn *framework.Session) {
	resourceTypes := []v1.ResourceName{
		v1.ResourceCPU,
		v1.ResourceMemory,
		v1.ResourceStorage,
		"nvidia.com/gpu",
	}
	
	for nodeName, node := range ssn.Nodes {
		stm.resourceTensor[nodeName] = make(map[v1.ResourceName][]float64)
		stm.nodeAvailability[nodeName] = make([]bool, len(stm.timeSlots))
		
		// Initialize availability (default: always available)
		// In practice, this would be configured based on node labels/annotations
		for i := range stm.timeSlots {
			stm.nodeAvailability[nodeName][i] = true
		}
		
		// Check for availability window annotations
		if window, ok := node.Node.Annotations["vortex.scheduler/availability-window"]; ok {
			// Parse window format: "HH:MM-HH:MM" (e.g., "00:00-06:00")
			stm.parseAvailabilityWindow(nodeName, window)
		}
		
		// Build resource capacity for each time slot
		for _, resType := range resourceTypes {
			capacities := make([]float64, len(stm.timeSlots))
			
			var maxCapacity float64
			switch resType {
			case v1.ResourceCPU:
				maxCapacity = float64(node.Allocatable.MilliCPU)
			case v1.ResourceMemory:
				maxCapacity = float64(node.Allocatable.Memory)
			case v1.ResourceStorage:
				maxCapacity = float64(node.Allocatable.Storage)
			default:
				if node.Allocatable.ScalarResources != nil {
					maxCapacity = node.Allocatable.ScalarResources[string(resType)]
				}
			}
			
			// Set capacity for each time slot (considering availability)
			for i := range stm.timeSlots {
				if stm.nodeAvailability[nodeName][i] {
					capacities[i] = maxCapacity
				} else {
					capacities[i] = 0
				}
			}
			
			stm.resourceTensor[nodeName][resType] = capacities
		}
		
		// Calculate heterogeneity index
		stm.heterogeneityIndex[nodeName] = stm.calculateHeterogeneityIndex(node)
	}
	
	klog.V(4).Infof("Vortex: Initialized resource capability tensor for %d nodes", len(ssn.Nodes))
}

// parseAvailabilityWindow parses availability window string
func (stm *SpatioTemporalModel) parseAvailabilityWindow(nodeName, window string) {
	// Simplified parsing - format: "00:00-06:00"
	// In practice, use proper time parsing
	for i := range stm.timeSlots {
		hour := stm.timeSlots[i].Start.Hour()
		// Example: node available from midnight to 6 AM
		if window == "00:00-06:00" {
			stm.nodeAvailability[nodeName][i] = (hour >= 0 && hour < 6)
		}
		// Add more patterns as needed
	}
}

// calculateHeterogeneityIndex computes node heterogeneity score
func (stm *SpatioTemporalModel) calculateHeterogeneityIndex(node *api.NodeInfo) float64 {
	// Measure relative capability compared to a reference
	// Higher index = more capable node
	
	cpuScore := float64(node.Allocatable.MilliCPU) / 8000.0    // 8 cores reference
	memScore := float64(node.Allocatable.Memory) / 16000.0     // 16 GB reference
	
	// Check for GPU
	gpuScore := 0.0
	if node.Allocatable.ScalarResources != nil {
		if gpu, ok := node.Allocatable.ScalarResources["nvidia.com/gpu"]; ok {
			gpuScore = gpu * 2.0 // GPUs significantly increase capability
		}
	}
	
	return (cpuScore + memScore + gpuScore) / 3.0
}

// GetLatency returns network latency between two nodes
func (stm *SpatioTemporalModel) GetLatency(node1, node2 string) float64 {
	stm.mutex.RLock()
	defer stm.mutex.RUnlock()
	
	if latencies, ok := stm.latencyMatrix[node1]; ok {
		if latency, ok := latencies[node2]; ok {
			return latency
		}
	}
	return 100.0 // Default high latency
}

// GetBandwidth returns network bandwidth between two nodes
func (stm *SpatioTemporalModel) GetBandwidth(node1, node2 string) float64 {
	stm.mutex.RLock()
	defer stm.mutex.RUnlock()
	
	if bandwidths, ok := stm.bandwidthMatrix[node1]; ok {
		if bandwidth, ok := bandwidths[node2]; ok {
			return bandwidth
		}
	}
	return 100.0 // Default: 100 MB/s
}

// GetAvailableNodes returns nodes available in the current time window
func (stm *SpatioTemporalModel) GetAvailableNodes(t time.Time) []string {
	stm.mutex.RLock()
	defer stm.mutex.RUnlock()
	
	// Find current time slot
	slotIndex := stm.getTimeSlotIndex(t)
	if slotIndex < 0 {
		return nil
	}
	
	var available []string
	for nodeName, availability := range stm.nodeAvailability {
		if slotIndex < len(availability) && availability[slotIndex] {
			available = append(available, nodeName)
		}
	}
	
	return available
}

// getTimeSlotIndex finds the time slot index for a given time
func (stm *SpatioTemporalModel) getTimeSlotIndex(t time.Time) int {
	for i, slot := range stm.timeSlots {
		if t.After(slot.Start) && t.Before(slot.End) {
			return i
		}
	}
	return -1
}

// UpdateAllocation updates resource allocation state
func (stm *SpatioTemporalModel) UpdateAllocation(task *api.TaskInfo, nodeName string, allocate bool) {
	stm.mutex.Lock()
	defer stm.mutex.Unlock()
	
	if allocate {
		stm.currentAllocations[nodeName] = append(stm.currentAllocations[nodeName], task)
	} else {
		// Remove task from allocations
		tasks := stm.currentAllocations[nodeName]
		for i, t := range tasks {
			if t.UID == task.UID {
				stm.currentAllocations[nodeName] = append(tasks[:i], tasks[i+1:]...)
				break
			}
		}
	}
}

// GetResourceUtilization returns current resource utilization for a node
func (stm *SpatioTemporalModel) GetResourceUtilization(nodeName string, resourceType v1.ResourceName) float64 {
	stm.mutex.RLock()
	defer stm.mutex.RUnlock()
	
	slotIndex := stm.getTimeSlotIndex(time.Now())
	if slotIndex < 0 {
		return 0
	}
	
	if resources, ok := stm.resourceTensor[nodeName]; ok {
		if capacities, ok := resources[resourceType]; ok {
			if slotIndex < len(capacities) {
				totalCapacity := capacities[slotIndex]
				if totalCapacity <= 0 {
					return 0
				}
				
				// Calculate used resources
				var used float64
				for _, task := range stm.currentAllocations[nodeName] {
					switch resourceType {
					case v1.ResourceCPU:
						used += float64(task.Resreq.MilliCPU)
					case v1.ResourceMemory:
						used += float64(task.Resreq.Memory)
					}
				}
				
				return used / totalCapacity
			}
		}
	}
	
	return 0
}
