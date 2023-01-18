#pragma once

#include "boids.h"
#include "utilities.cuh"
#include "boids_movementGPU.cuh"

void write_float3(float* dst, float3 v, int index) {
	dst[index] = v.x;
	dst[index + 1] = v.y;
	dst[index + 2] = v.z;
}

void print_float3(float3 f3) {
	printf("%3.1f, %3.1f, %3.1f\n", f3.x, f3.y, f3.z);
}

void generateBoidVertices(s_boids* boids) {
	const float3 UP = make_float3(0, 1, 0);

	for (int i = 0; i < boids->count; i++) {
		float3 pos = make_float3(
			boids->position.x[i],
			boids->position.y[i],
			boids->position.z[i]
		);
		float3 dir = make_float3(
			boids->direction.x[i], 
			boids->direction.y[i], 
			boids->direction.z[i]
		);

		// TOP vertex
		float3 top = make_float3(
			pos.x + dir.x * BOID_LENGTH,
			pos.y + dir.y * BOID_LENGTH,
			pos.z + dir.z * BOID_LENGTH
		);

		// Vector perpendicular to the direction vector
		float3 dir_normal;
		if (dir == UP) {
			dir_normal = { 0, 0, 1 };
		}
		else {
			dir_normal = normalize(cross(dir, UP));
		}

		// BOTTOM vertices
		float3 v1 = rotateVec3(dir, dir_normal, 1.1);
		float3 v2 = rotateVec3(v1, dir, 2 * PI / 3);
		float3 v3 = rotateVec3(v2, dir, 2 * PI / 3);

		float3 bottom1 = pos + v1 * BOID_WIDTH;
		float3 bottom2 = pos + v2 * BOID_WIDTH;
		float3 bottom3 = pos + v3 * BOID_WIDTH;	

		write_float3(boids->triangleVertices, top, 3 * (9 * i));
		write_float3(boids->triangleVertices, bottom1, 3 * (9 * i + 1));
		write_float3(boids->triangleVertices, bottom2, 3 * (9 * i + 2));
		write_float3(boids->triangleVertices, top, 3 * (9 * i + 3));
		write_float3(boids->triangleVertices, bottom2, 3 * (9 * i + 4));
		write_float3(boids->triangleVertices, bottom3, 3 * (9 * i + 5));
		write_float3(boids->triangleVertices, top, 3 * (9 * i + 6));
		write_float3(boids->triangleVertices, bottom3, 3 * (9 * i + 7));
		write_float3(boids->triangleVertices, bottom1, 3 * (9 * i + 8));
	}
}

inline void moveBoidsCPU(s_boids* boids) {
	for (int i = 0; i < boids->count; i++) {
		centerOfMassRule(boids, i);
		separationRule(boids, i);
		alignmentRule(boids, i);

		boids->position.x[i] += boids->direction.x[i] * boids->simulationOptions.boidSpeed;
		boids->position.y[i] += boids->direction.y[i] * boids->simulationOptions.boidSpeed;
		boids->position.z[i] += boids->direction.z[i] * boids->simulationOptions.boidSpeed;
	}
}