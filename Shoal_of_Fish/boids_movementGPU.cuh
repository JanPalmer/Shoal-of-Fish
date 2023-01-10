#include "utilities.cuh"
#include "boids.h"

__host__ __device__ bool isInSight(s_boids* boids, int index, int other);
__host__ __device__ void centerOfMassRule(s_boids* boids, int index, float deltaTime);
__host__ __device__ void separationRule(s_boids* boids, int index, float deltaTime);
__host__ __device__ void alignmentRule(s_boids* boids, int index, float deltaTime);
__global__ void moveBoidsGPU(s_boids* boids, float deltaTime);