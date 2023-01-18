#pragma once
#include <stdlib.h>
#include "cuda_runtime.h"
#include "Dependencies/Helpers/helper_cuda.h"

#define TRIANGLES_PER_BOID 3
#define VERTICES_PER_TRIANGLE 3
#define FLOATS_PER_VERTEX 3
#define BOID_LENGTH 0.05
#define BOID_WIDTH 0.01

struct s_vec3 {
	int count;
	float* x, * y, * z; // pointers to position arrays, one for each coordinate
};

struct s_simopts {
	float3 lowCubeCorner = { -1, -1, -1 };
	float3 highCubeCorner = { 1, 1, 1 };
	float boidSpeed = 0.1;
	float visionRadius = 0.1;
	float visionAngles = 6;
	float cohesionFactor = 0.1;
	float separationFactor = 0.026;
	float alignFactor = 0.05;
};

struct s_boids {
	int count;
	s_vec3 position;
	s_vec3 direction;
	float* velocity;
	float* triangleVertices; // drawn with GL_TRIANGLES directive
	s_simopts simulationOptions;
};

// Data initialization
inline void initVec3Array(s_vec3& vec3, int num) {
	vec3.count = num;
	vec3.x = new float[num];
	vec3.y = new float[num];
	vec3.z = new float[num];
}

inline void initBoids(s_boids& boids, int boidNum) {
	boids.count = boidNum;
	initVec3Array(boids.position, boidNum);
	initVec3Array(boids.direction, boidNum);
	boids.velocity = new float[boidNum];
	boids.triangleVertices = new float[boidNum * TRIANGLES_PER_BOID * VERTICES_PER_TRIANGLE * FLOATS_PER_VERTEX];
}

inline void initVec3ArrayGPU(s_vec3& vec3, int num) {
	checkCudaErrors(cudaMalloc((void**)&vec3.x, num * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&vec3.y, num * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&vec3.z, num * sizeof(float)));
}

inline void initBoidsGPU(s_boids& boids, s_boids& helper, int boidNum) {
	initVec3ArrayGPU(helper.position, boidNum);
	initVec3ArrayGPU(helper.direction, boidNum);
	checkCudaErrors(cudaMalloc((void**)&helper.velocity, boidNum * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&helper.triangleVertices,
		boidNum * TRIANGLES_PER_BOID * VERTICES_PER_TRIANGLE * FLOATS_PER_VERTEX * sizeof(float)));

	checkCudaErrors(cudaMemcpy(&boids.position.x, &helper.position.x, sizeof(float*), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&boids.position.y, &helper.position.y, sizeof(float*), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&boids.position.z, &helper.position.z, sizeof(float*), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&boids.direction.x, &helper.direction.x, sizeof(float*), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&boids.direction.y, &helper.direction.y, sizeof(float*), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&boids.direction.z, &helper.direction.z, sizeof(float*), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&boids.velocity, &helper.velocity, sizeof(float*), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&boids.triangleVertices, &helper.triangleVertices, sizeof(float*), cudaMemcpyHostToDevice));

}

inline void freeVec3Array(s_vec3& vec3) {
	delete vec3.x;
	delete vec3.y;
	delete vec3.z;
}

inline void freeBoids(s_boids& boids) {
	freeVec3Array(boids.position);
	freeVec3Array(boids.direction);

	delete boids.velocity;
	delete boids.triangleVertices;
}

inline void freeVec3ArrayGPU(s_vec3& vec3) {
	cudaFree(vec3.x);
	cudaFree(vec3.y);
	cudaFree(vec3.z);
}

inline void setBoid(s_boids& boids, int i, float3 position, float3 direction, float velocity) {
	if (i > boids.count) return;

	boids.position.x[i] = position.x;
	boids.position.y[i] = position.y;
	boids.position.z[i] = position.z;
	boids.direction.x[i] = direction.x;
	boids.direction.y[i] = direction.y;
	boids.direction.z[i] = direction.z;
	boids.velocity[i] = velocity;
}

// Random data generation
static float getRandomfloat(float start, float end) {
	return start + static_cast <float> (rand() / (static_cast <float> ((float)RAND_MAX / (end - start))));
}

static float3 getRandomfloat3(float start, float end) {
	float r1 = getRandomfloat(start, end);
	float r2 = getRandomfloat(start, end);
	float r3 = getRandomfloat(start, end);
	return make_float3(r1, r2, r3);
}

static void randomizeBoids(s_boids& boids, float velocityMin, float velocityMax, float positionMin, float positionMax) {
	srand(time(NULL));
	for (int i = 0; i < boids.count; i++) {
		float3 randomPosition = getRandomfloat3(positionMin, positionMax);
		float3 randomDirection = getRandomfloat3(-1, 1);
		float randomVelocity = getRandomfloat(velocityMin, velocityMax);

		setBoid(boids, i, randomPosition, randomDirection, randomVelocity);
	}
}