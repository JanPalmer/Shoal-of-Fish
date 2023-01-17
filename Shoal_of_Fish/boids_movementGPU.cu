#include "boids_movementGPU.cuh"

__host__ __device__ bool isInSight(s_boids* boids, int index, int other) {
	float3 pos = getFloat3FromVec3(boids->position, index);
	float3 otherpos = getFloat3FromVec3(boids->position, other);
	
	// Check if the other boid is close enough to the target
	float3 posvector = pos - otherpos;
	if (vectorLength(posvector) > boids->simulationOptions.visionRadius) {
		return false;
	}

	// Check if the other boid is in the vision cone
	float3 dir = getFloat3FromVec3(boids->direction, index);
	if (vectorDistance(dir, normalize(posvector)) > boids->simulationOptions.visionAngles) {
		return false;
	}

	return true;
}

__host__ __device__ void centerOfMassRule(s_boids* boids, int index, float deltaTime) {
	// Boids try to approach the center of mass of the flockmates they can see

	float3 positionSum = { 0, 0, 0 };
	int numberOfBoidsNearby = 0;

	for (int i = 0; i < boids->count; i++) {
		if (isInSight(boids, index, i)) {
			positionSum += getFloat3FromVec3(boids->position, i);
			numberOfBoidsNearby++;
		}

	}

	float3 centerOfMass = positionSum / numberOfBoidsNearby;
	float3 movementVector = centerOfMass - getFloat3FromVec3(boids->position, index);
	movementVector *= boids->simulationOptions.cohesionFactor;

	float3 currDirection = getFloat3FromVec3(boids->direction, index);
	currDirection.x += movementVector.x;
	currDirection.y += movementVector.y;
	currDirection.z += movementVector.z;
	currDirection = normalize(currDirection);
	boids->direction.x[index] = currDirection.x;
	boids->direction.y[index] = currDirection.y;
	boids->direction.z[index] = currDirection.z;
	return;
}

__host__ __device__ void separationRule(s_boids* boids, int index, float deltaTime) {
	// Boids try to stay away from each other
	// Also, if a boid goes out of bounds of the cube, a force pushing him back inside will be applied

	float3 repelSum = { 0, 0, 0 };
	int numberOfBoidsNearby = 0;
	float3 pos = getFloat3FromVec3(boids->position, index);

	for (int i = 0; i < boids->count; i++) {
		if (isInSight(boids, index, i)) {
			repelSum += pos - getFloat3FromVec3(boids->position, i);
			numberOfBoidsNearby++;
		}
	}

	float3 repelForce = repelSum / numberOfBoidsNearby;	
	float3 lowCorner = boids->simulationOptions.lowCubeCorner;
	float3 highCorner = boids->simulationOptions.highCubeCorner;

	float separationFactor = boids->simulationOptions.separationFactor;
	repelForce *= separationFactor * 4;
	float3 currDirection = getFloat3FromVec3(boids->direction, index);
	currDirection.x += repelForce.x;
	currDirection.y += repelForce.y;
	currDirection.z += repelForce.z;

	// if boid is out of bounds of the box, push him back inside
	if (pos.x < lowCorner.x || pos.x > highCorner.x) {
		currDirection.x += -(pos.x * separationFactor);
	}
	if (pos.y < lowCorner.y || pos.y > highCorner.y) {
		currDirection.y += -(pos.y * separationFactor);
	}
	if (pos.z < lowCorner.z || pos.z > highCorner.z) {
		currDirection.z += -(pos.z * separationFactor);
	}

	currDirection = normalize(currDirection);
	boids->direction.x[index] = currDirection.x;
	boids->direction.y[index] = currDirection.y;
	boids->direction.z[index] = currDirection.z;
	return;
}

__host__ __device__ void alignmentRule(s_boids* boids, int index, float deltaTime) {
	// Boids try to adjust their heading direction to an average of the flockmates they can see

	float3 alignSum = { 0, 0, 0 };
	int numberOfBoidsNearby = 0;
	float3 pos = getFloat3FromVec3(boids->position, index);

	for (int i = 0; i < boids->count; i++) {
		if (isInSight(boids, index, i)) {
			alignSum += getFloat3FromVec3(boids->direction, i);
			numberOfBoidsNearby++;
		}
	}

	float3 alignForce = alignSum / numberOfBoidsNearby;

	float alignmentFactor = boids->simulationOptions.alignFactor;
	alignForce *= alignmentFactor;
	float3 currDirection = getFloat3FromVec3(boids->direction, index);
	currDirection.x += alignForce.x;
	currDirection.y += alignForce.y;
	currDirection.z += alignForce.z;

	currDirection = normalize(currDirection);
	boids->direction.x[index] = currDirection.x;
	boids->direction.y[index] = currDirection.y;
	boids->direction.z[index] = currDirection.z;
	return;
}

__global__ void moveBoidsGPU(s_boids* boids, float deltaTime) {

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index >= boids->count) return;

	// RULES
	// fly towards the center of mass
	// keep distance away from other boids and cube borders
	// try to match velocity of other boids

	centerOfMassRule(boids, index, deltaTime);
	separationRule(boids, index, deltaTime);
	alignmentRule(boids, index, deltaTime);

	boids->position.x[index] += boids->direction.x[index] * boids->simulationOptions.boidSpeed;
	boids->position.y[index] += boids->direction.y[index] * boids->simulationOptions.boidSpeed;
	boids->position.z[index] += boids->direction.z[index] * boids->simulationOptions.boidSpeed;
}