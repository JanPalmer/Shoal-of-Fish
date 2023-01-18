#ifndef UTILITIES_H
#define UTILITIES_H

#include "Dependencies/Helpers/helper_cuda.h"
#include "Dependencies/Helpers/helper_math.h"
#include "boids.h"

static const float PI = 3.1415926535897932385f;

__host__ __device__ static float3 rotateVec3(float3& vec3, float3& dir, double theta) {
	float cos_theta = (float)cos(theta);
	float sin_theta = (float)sin(theta);

	return (vec3 * cos_theta) + (cross(dir, vec3) * sin_theta) + (dir * dot(dir, vec3)) * (1 - cos_theta);
}

__host__ __device__ inline bool operator==(const float3& a, const float3& b) {
	if (a.x == b.x && a.y == b.y && a.z == b.z)
		return true;
	else
		return false;
}

__host__ __device__ static float vectorLength(const float3& v) {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ static float vectorDistance(const float3& v, const float3& u) {
	float3 tmp = v - u;
	return vectorLength(tmp);
}

__host__ __device__ static float3 getFloat3FromVec3(s_vec3& vec3, int index) {
	return make_float3(
		vec3.x[index],
		vec3.y[index],
		vec3.z[index]
	);
}

#endif