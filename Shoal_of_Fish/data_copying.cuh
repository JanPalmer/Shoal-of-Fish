#include "boids.h"
#include "Dependencies/Helpers/helper_cuda.h"
#include "Dependencies/Helpers/helper_math.h"

static void copyHostToDevice_vec3(s_vec3& dst, s_vec3& src) {
    checkCudaErrors(cudaMemcpy(dst.x, src.x, sizeof(float) * src.count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.y, src.y, sizeof(float) * src.count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dst.z, src.z, sizeof(float) * src.count, cudaMemcpyHostToDevice));
}

static void copyHostToDevice_boids(s_boids& dst, s_boids& helper, s_boids& src) {
    checkCudaErrors(cudaMemcpy(&dst, &src, sizeof(s_boids), cudaMemcpyHostToDevice));

    copyHostToDevice_vec3(helper.position, src.position);
    copyHostToDevice_vec3(helper.direction, src.direction);
    checkCudaErrors(cudaMemcpy(helper.velocity, src.velocity, sizeof(float) * src.count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(helper.triangleVertices, src.triangleVertices, sizeof(float) * src.count * TRIANGLES_PER_BOID * VERTICES_PER_TRIANGLE * FLOATS_PER_VERTEX, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(&dst.position.x, &helper.position.x, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst.position.y, &helper.position.y, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst.position.z, &helper.position.z, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst.direction.x, &helper.direction.x, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst.direction.y, &helper.direction.y, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst.direction.z, &helper.direction.z, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst.velocity, &helper.velocity, sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dst.triangleVertices, &helper.triangleVertices, sizeof(float*), cudaMemcpyHostToDevice));
}

static void copyDeviceToHost_vec3(s_vec3& dst, s_vec3& src, int count) {
    checkCudaErrors(cudaMemcpy(dst.x, src.x, sizeof(float) * count, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dst.y, src.y, sizeof(float) * count, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dst.z, src.z, sizeof(float) * count, cudaMemcpyDeviceToHost));
}

static void copyDeviceToHost_boids(s_boids& dst, s_boids& helper, s_boids& src) {
    //We only need to copy the position and direction of each boid, the 
    copyDeviceToHost_vec3(dst.position, helper.position, dst.count);
    copyDeviceToHost_vec3(dst.direction, helper.direction, dst.count);
}