// Package bev implements Bird's-Eye-View (BEV) coordinate transformations
// for Tesla FSD occupancy streaming with temporal fusion across frames.
package bev

import (
	"fmt"
	"math"
	"sync"
)

// BEVConfig defines the bird's-eye-view grid configuration
type BEVConfig struct {
	RangeMeters  float64 // Detection range from ego vehicle
	ResolutionM  float64 // Meters per grid cell
	GridSize     int     // Grid dimension (GridSize x GridSize)
	NumClasses   int     // Semantic class count
	TemporalFrames int   // Number of historical frames for fusion
}

// DefaultConfig returns a production-grade BEV configuration
func DefaultConfig() BEVConfig {
	return BEVConfig{
		RangeMeters:    50.0,
		ResolutionM:    0.25,
		GridSize:       400,
		NumClasses:     23,
		TemporalFrames: 4,
	}
}

// EgoPose represents the vehicle's pose in world coordinates
type EgoPose struct {
	X, Y, Z    float64 // Translation (meters)
	Roll, Pitch, Yaw float64 // Rotation (radians)
	Timestamp  uint64  // Nanoseconds
}

// OccupancyCell represents a single BEV grid cell
type OccupancyCell struct {
	ClassLogits    []float32
	Occupancy      float32
	Height         float32 // Max height in this cell
	Velocity       [2]float32 // Vx, Vy flow
	PredictedClass int
}

// BEVGrid holds the complete bird's-eye-view occupancy grid
type BEVGrid struct {
	mu     sync.RWMutex
	config BEVConfig
	cells  [][]OccupancyCell
	pose   EgoPose
	frames []BEVGrid // Temporal buffer (no mutex, inner grids)
	frameIdx int
}

// NewBEVGrid creates a new BEV grid with the given configuration
func NewBEVGrid(cfg BEVConfig) *BEVGrid {
	cells := make([][]OccupancyCell, cfg.GridSize)
	for i := range cells {
		row := make([]OccupancyCell, cfg.GridSize)
		for j := range row {
			row[j].ClassLogits = make([]float32, cfg.NumClasses)
		}
		cells[i] = row
	}
	return &BEVGrid{
		config: cfg,
		cells:  cells,
	}
}

// WorldToBEV converts world coordinates to BEV grid indices relative to ego pose
func (g *BEVGrid) WorldToBEV(worldX, worldY float64) (int, int, bool) {
	// Translate to ego-relative coordinates
	dx := worldX - g.pose.X
	dy := worldY - g.pose.Y

	// Rotate by negative yaw (align with ego heading)
	cosYaw := math.Cos(-g.pose.Yaw)
	sinYaw := math.Sin(-g.pose.Yaw)
	egoX := dx*cosYaw - dy*sinYaw
	egoY := dx*sinYaw + dy*cosYaw

	// Convert to grid indices
	gridX := int((egoX + g.config.RangeMeters) / g.config.ResolutionM)
	gridY := int((egoY + g.config.RangeMeters) / g.config.ResolutionM)

	inBounds := gridX >= 0 && gridX < g.config.GridSize && gridY >= 0 && gridY < g.config.GridSize
	return gridX, gridY, inBounds
}

// BEVToWorld converts BEV grid indices back to world coordinates
func (g *BEVGrid) BEVToWorld(gridX, gridY int) (float64, float64) {
	egoX := float64(gridX)*g.config.ResolutionM - g.config.RangeMeters
	egoY := float64(gridY)*g.config.ResolutionM - g.config.RangeMeters

	cosYaw := math.Cos(g.pose.Yaw)
	sinYaw := math.Sin(g.pose.Yaw)
	worldX := egoX*cosYaw - egoY*sinYaw + g.pose.X
	worldY := egoX*sinYaw + egoY*cosYaw + g.pose.Y

	return worldX, worldY
}

// UpdatePose sets the ego vehicle pose and triggers coordinate rebasing
func (g *BEVGrid) UpdatePose(pose EgoPose) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.pose = pose
}

// SetOccupancy writes occupancy data for a world-space point
func (g *BEVGrid) SetOccupancy(worldX, worldY float64, classID int, prob float32) error {
	gx, gy, ok := g.WorldToBEV(worldX, worldY)
	if !ok {
		return fmt.Errorf("point (%.2f, %.2f) outside BEV range", worldX, worldY)
	}
	g.mu.Lock()
	defer g.mu.Unlock()

	cell := &g.cells[gx][gy]
	if classID < len(cell.ClassLogits) {
		cell.ClassLogits[classID] += float32(math.Log(float64(prob) / (1.0 - float64(prob) + 1e-7)))
	}
	cell.Occupancy = prob
	cell.PredictedClass = classID
	return nil
}

// OccupiedCellCount returns the number of cells with occupancy > threshold
func (g *BEVGrid) OccupiedCellCount(threshold float32) int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	count := 0
	for _, row := range g.cells {
		for _, cell := range row {
			if cell.Occupancy > threshold {
				count++
			}
		}
	}
	return count
}

// Stats returns grid statistics
func (g *BEVGrid) Stats() map[string]interface{} {
	g.mu.RLock()
	defer g.mu.RUnlock()
	totalCells := g.config.GridSize * g.config.GridSize
	occupied := g.OccupiedCellCount(0.5)
	return map[string]interface{}{
		"grid_size":    g.config.GridSize,
		"resolution_m": g.config.ResolutionM,
		"range_m":      g.config.RangeMeters,
		"total_cells":  totalCells,
		"occupied":     occupied,
		"occupancy_%":  float64(occupied) / float64(totalCells) * 100,
	}
}
