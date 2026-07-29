/**
 * Tesla FSD Occupancy Network — CUDA Voxel Grid Raycasting Kernel
 * Implements real-time 3D occupancy grid construction from multi-camera
 * surround vision using parallel voxel raycasting with semantic labels.
 */

#include <cstdio>
#include <cmath>

#define GRID_X 200
#define GRID_Y 200
#define GRID_Z 16
#define VOXEL_SIZE_M 0.25f
#define NUM_CLASSES 23  // road, vehicle, pedestrian, bike, etc.
#define NUM_CAMERAS 8

struct CameraIntrinsics {
    float fx, fy, cx, cy;
    int width, height;
};

struct CameraExtrinsics {
    float rotation[9];   // 3x3 rotation matrix (row-major)
    float translation[3]; // 3D translation
};

struct OccupancyVoxel {
    float logit[NUM_CLASSES];  // semantic class logits
    float occupancy_prob;       // [0,1] occupancy probability
    unsigned char predicted_class;
};

/**
 * CUDA kernel: Project voxel centers to camera views and accumulate features
 * Each thread handles one voxel in the 3D grid.
 */
__global__ void voxel_raycast_kernel(
    OccupancyVoxel* __restrict__ grid,           // [GRID_X * GRID_Y * GRID_Z]
    const float* __restrict__ camera_features,    // [NUM_CAMERAS x H x W x C]
    const CameraIntrinsics* __restrict__ intrinsics,  // [NUM_CAMERAS]
    const CameraExtrinsics* __restrict__ extrinsics,  // [NUM_CAMERAS]
    int feature_channels,
    int feat_h, int feat_w
) {
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = GRID_X * GRID_Y * GRID_Z;
    if (voxel_idx >= total_voxels) return;

    // Compute 3D world position from voxel index
    int vz = voxel_idx / (GRID_X * GRID_Y);
    int vy = (voxel_idx % (GRID_X * GRID_Y)) / GRID_X;
    int vx = voxel_idx % GRID_X;

    float world_x = (vx - GRID_X / 2) * VOXEL_SIZE_M;
    float world_y = (vy - GRID_Y / 2) * VOXEL_SIZE_M;
    float world_z = vz * VOXEL_SIZE_M - 2.0f; // ground plane offset

    OccupancyVoxel& voxel = grid[voxel_idx];
    float total_weight = 0.0f;

    // Project to each camera and sample features
    for (int cam = 0; cam < NUM_CAMERAS; cam++) {
        const CameraExtrinsics& ext = extrinsics[cam];
        const CameraIntrinsics& intr = intrinsics[cam];

        // World to camera transform
        float cx = ext.rotation[0] * world_x + ext.rotation[1] * world_y +
                   ext.rotation[2] * world_z + ext.translation[0];
        float cy_cam = ext.rotation[3] * world_x + ext.rotation[4] * world_y +
                       ext.rotation[5] * world_z + ext.translation[1];
        float cz = ext.rotation[6] * world_x + ext.rotation[7] * world_y +
                   ext.rotation[8] * world_z + ext.translation[2];

        if (cz <= 0.1f) continue; // Behind camera

        // Project to pixel coordinates
        float u = intr.fx * cx / cz + intr.cx;
        float v = intr.fy * cy_cam / cz + intr.cy;

        // Feature map coordinates (downsampled)
        int fu = (int)(u * feat_w / intr.width);
        int fv = (int)(v * feat_h / intr.height);

        if (fu < 0 || fu >= feat_w || fv < 0 || fv >= feat_h) continue;

        // Depth-based weighting (closer = more weight)
        float depth = sqrtf(cx * cx + cy_cam * cy_cam + cz * cz);
        float weight = 1.0f / fmaxf(depth, 0.1f);

        // Accumulate semantic logits from camera features
        int feat_offset = cam * feat_h * feat_w * feature_channels + fv * feat_w * feature_channels + fu * feature_channels;
        for (int c = 0; c < NUM_CLASSES && c < feature_channels; c++) {
            voxel.logit[c] += camera_features[feat_offset + c] * weight;
        }
        total_weight += weight;
    }

    // Normalize and compute occupancy
    if (total_weight > 0.0f) {
        float max_logit = -1e9f;
        int best_class = 0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            voxel.logit[c] /= total_weight;
            if (voxel.logit[c] > max_logit) {
                max_logit = voxel.logit[c];
                best_class = c;
            }
        }
        voxel.predicted_class = (unsigned char)best_class;
        // Sigmoid for occupancy probability
        voxel.occupancy_prob = 1.0f / (1.0f + expf(-max_logit));
    }
}

/**
 * Host function: Launch occupancy grid construction
 */
extern "C" int build_occupancy_grid(int feat_channels, int feat_h, int feat_w) {
    int total_voxels = GRID_X * GRID_Y * GRID_Z;
    int block_size = 256;
    int num_blocks = (total_voxels + block_size - 1) / block_size;

    printf("[OccupancyGrid] Grid: %dx%dx%d = %d voxels, voxel_size=%.2fm\n",
           GRID_X, GRID_Y, GRID_Z, total_voxels, VOXEL_SIZE_M);
    printf("[OccupancyGrid] Cameras: %d, Features: %dx%dx%d\n",
           NUM_CAMERAS, feat_h, feat_w, feat_channels);
    printf("[OccupancyGrid] Launch: %d blocks x %d threads\n", num_blocks, block_size);

    return total_voxels;
}
