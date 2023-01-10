/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /*
     Particle system example with collisions using uniform grid

     CUDA 2.1 SDK release 12/2008
     - removed atomic grid method, some optimization, added demo mode.

     CUDA 2.2 release 3/2009
     - replaced sort function with latest radix sort, now disables v-sync.
     - added support for automated testing and comparison to a reference value.
 */

 // OpenGL Graphics includes

#include "Dependencies/GL/glew.h"
#include "Dependencies/GL/freeglut.h"
#include "Dependencies/Helpers/helper_gl.h"
// CUDA runtime
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

// CUDA utilities and system includes
#include "Dependencies/Helpers/helper_cuda.h"  // includes cuda.h and cuda_runtime_api.h
#include "Dependencies/Helpers/helper_timer.h"

#include "boids.h"
#include "boids_movementCPU.h"
#include "boids_movementGPU.cuh"
#include "data_copying.cuh"

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD 0.30f

#define GRID_SIZE 64
#define NUM_BOIDS 38500

const int width = 640, height = 480;

// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = { 0, 0, -3 };
float camera_rot[] = { 0, 0, 0 };
float camera_trans_lag[] = { 0, 0, -3 };
float camera_rot_lag[] = { 0, 0, 0 };
const float inertia = 0.1f;

int mode = 0;
bool displayEnabled = true;
bool bPause = false;
bool wireframe = false;
bool useCPU = false;
//bool demoMode = false;
//int idleCounter = 0;
//int demoCounter = 0;
//const int idleDelay = 2000;

enum { M_VIEW = 0 };

const int cubeSize = 2;
int numBoids = 1;
uint3 gridSize;

// simulation parameters
float timestep = 0.5f;

// fps
static int frameCount = 0;
static int fpsCount = 0;
static int fpsLimit = 30;
StopWatchInterface* timer = NULL;
StopWatchInterface* renderTimer = NULL;

float modelView[16];

s_boids h_boids, helper, *d_boids;
float renderTime = 0;
float timeFor15frames = 0;
float copyToDeviceTime, computationTime, copyToHostTime;

void drawBoids(s_boids* boids) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glColor3f(0.9f, 0.6f, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glVertexPointer(3, GL_FLOAT, 0, boids->triangleVertices);
    glDrawArrays(GL_TRIANGLES, 0, boids->count * TRIANGLES_PER_BOID * VERTICES_PER_TRIANGLE);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glColor3f(0, 0, 0);
    glVertexPointer(3, GL_FLOAT, 0, boids->triangleVertices);
    glDrawArrays(GL_TRIANGLES, 0, boids->count * TRIANGLES_PER_BOID * VERTICES_PER_TRIANGLE);

    glDisableClientState(GL_VERTEX_ARRAY);
}

void cleanup() {
    sdkDeleteTimer(&timer);
    sdkDeleteTimer(&renderTimer);
    freeVec3ArrayGPU(helper.position);
    freeVec3ArrayGPU(helper.direction);
    cudaFree(helper.velocity);
    cudaFree(helper.triangleVertices);
    cudaFree(d_boids);
    freeBoids(h_boids);
    return;
}

// initialize OpenGL
void initGL(int* argc, char** argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA Particles");

    if (!isGLVersionSupported(2, 0) ||
        !areGLExtensionsSupported(
            "GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);

    glutReportErrors();
}

void computeFPS_GPU(float copyToDeviceTime, float computationTime, float copyToHostTime) {
    fpsCount++;

    timeFor15frames += copyToDeviceTime + computationTime + copyToHostTime;

    if (fpsCount == 15) {
        char fps[256];
        float ifps = 1.0f / timeFor15frames * 15.0f;
        sprintf(fps, "CUDA Shoal of Fish (%d boids): %1.5f + %1.8f + %1.5f, %3.1f fps",
            numBoids, copyToDeviceTime, computationTime, copyToHostTime, ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;
        timeFor15frames = 0;
    }
}

void display() {
    sdkStartTimer(&renderTimer);

    // update the simulation
    if (!bPause) {
        if (useCPU) {
            moveBoidsCPU(&h_boids, renderTime);
        }
        else {
            sdkResetTimer(&timer);
            sdkStartTimer(&timer);
			copyHostToDevice_boids(*d_boids, helper, h_boids);
            copyToDeviceTime = 0.001f * sdkGetTimerValue(&timer);

            sdkResetTimer(&timer);
            sdkStartTimer(&timer);

			int blocks = (numBoids / 512 + 1);
			int threads = 512;
			moveBoidsGPU<<<blocks, threads>>>(d_boids, renderTime);
            computationTime = 0.001f * sdkGetTimerValue(&timer);

            sdkResetTimer(&timer);
            sdkStartTimer(&timer);

			copyDeviceToHost_boids(h_boids, helper, *d_boids);
            copyToHostTime = 0.001f * sdkGetTimerValue(&timer);

            sdkStopTimer(&timer);
        }

        //float3 dir = make_float3(
        //    h_boids.position.x[0],
        //    h_boids.position.y[0],
        //    h_boids.position.z[0]
        //);
        //print_float3(dir);
        //float3 tri = make_float3(
        //    h_boids.triangleVertices[0],
        //    h_boids.triangleVertices[1],
        //    h_boids.triangleVertices[2]
        //);
        //print_float3(tri);

        //h_boids.position.x[0] += h_boids.direction.x[0] * 0.01;
        //h_boids.position.y[0] += h_boids.direction.y[0] * 0.01;
        //h_boids.position.z[0] += h_boids.direction.z[0] * 0.01;

        generateBoidVertices(&h_boids);
    }

    // render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawBoids(&h_boids);

    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    for (int c = 0; c < 3; ++c) {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    // cube
    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(cubeSize);

    sdkStopTimer(&renderTimer);
    renderTime = 0.001f * sdkGetTimerValue(&renderTimer);
    sdkResetTimer(&renderTimer);

    glutSwapBuffers();
    glutReportErrors();

    computeFPS_GPU(copyToDeviceTime, computationTime, copyToHostTime);
}

inline float frand() { return rand() / (float)RAND_MAX; }

void reshape(int w, int h) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float)w / (float)h, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

void mouse(int button, int state, int x, int y) {
    int mods;

    if (state == GLUT_DOWN) {
        buttonState |= 1 << button;
    }
    else if (state == GLUT_UP) {
        buttonState = 0;
    }

    mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT) {
        buttonState = 2;
    }
    else if (mods & GLUT_ACTIVE_CTRL) {
        buttonState = 3;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}

// transform vector by matrix
void xform(float* v, float* r, GLfloat* m) {
    r[0] = v[0] * m[0] + v[1] * m[4] + v[2] * m[8] + m[12];
    r[1] = v[0] * m[1] + v[1] * m[5] + v[2] * m[9] + m[13];
    r[2] = v[0] * m[2] + v[1] * m[6] + v[2] * m[10] + m[14];
}

// transform vector by transpose of matrix
void ixform(float* v, float* r, GLfloat* m) {
    r[0] = v[0] * m[0] + v[1] * m[1] + v[2] * m[2];
    r[1] = v[0] * m[4] + v[1] * m[5] + v[2] * m[6];
    r[2] = v[0] * m[8] + v[1] * m[9] + v[2] * m[10];
}

void ixformPoint(float* v, float* r, GLfloat* m) {
    float x[4];
    x[0] = v[0] - m[12];
    x[1] = v[1] - m[13];
    x[2] = v[2] - m[14];
    x[3] = 1.0f;
    ixform(x, r, m);
}

void motion(int x, int y) {
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    switch (mode) {
    case M_VIEW:
        if (buttonState == 3) {
            // left+middle = zoom
            camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
        }
        else if (buttonState & 2) {
            // middle = translate
            camera_trans[0] += dx / 100.0f;
            camera_trans[1] -= dy / 100.0f;
        }
        else if (buttonState & 1) {
            // left = rotate
            camera_rot[0] += dy / 5.0f;
            camera_rot[1] += dx / 5.0f;
        }

        break;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/) {
    switch (key) {
    case ' ':
        bPause = !bPause;
        break;
    case 'G':
        useCPU = false;
        break;
    case 'g':
        useCPU = true;
        break;
    case 'q':
#if defined(__APPLE__) || defined(MACOSX)
        exit(EXIT_SUCCESS);
#else
        glutDestroyWindow(glutGetWindow());
        return;
#endif
    }

    glutPostRedisplay();
}

void idle(void) {
    glutPostRedisplay();
}

void mainMenu(int i) { key((unsigned char)i, 0, 0); }

void initMenus() {
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Reset block [1]", '1');
    glutAddMenuEntry("Reset random [2]", '2');
    glutAddMenuEntry("Add sphere [3]", '3');
    glutAddMenuEntry("View mode [v]", 'v');
    glutAddMenuEntry("Move cursor mode [m]", 'm');
    glutAddMenuEntry("Toggle point rendering [p]", 'p');
    glutAddMenuEntry("Toggle animation [ ]", ' ');
    glutAddMenuEntry("Step animation [ret]", 13);
    glutAddMenuEntry("Toggle sliders [h]", 'h');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    printf(
        "NOTE: The CUDA Samples are not meant for performance measurements. "
        "Results may vary when GPU Boost is enabled.\n\n");

    numBoids = NUM_BOIDS;
    int gridDim = GRID_SIZE;

    gridSize.x = gridSize.y = gridSize.z = gridDim;
    printf("grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z,
        gridSize.x * gridSize.y * gridSize.z);
    printf("Boids: %d\n", numBoids);

    initGL(&argc, argv);
    initBoids(h_boids, numBoids);
    checkCudaErrors(cudaMalloc((void**)&d_boids, sizeof(s_boids)));
    initBoidsGPU(*d_boids, helper, numBoids);
    printf("h_boids count: %d\n", h_boids.count);
    setBoid(h_boids, 0, { 0, 0, 0 }, { 0, 1, 0 }, 0);
    randomizeBoids(h_boids, 0.01, 0.1, -1, 1);
    generateBoidVertices(&h_boids);
    sdkCreateTimer(&timer);
    sdkCreateTimer(&renderTimer);


    //for (int i = 0; i < 5; i++) {
    //    float3 pos = {
    //        h_boids.triangleVertices[3 * i],
    //        h_boids.triangleVertices[3 * i + 1],
    //        h_boids.triangleVertices[3 * i + 2]
    //    };
    //    printf("V %d: %3.1f, %3.1f, %3.1f\n", i, pos.x, pos.y, pos.z);
    //}

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(key);
	glutIdleFunc(idle);

	glutCloseFunc(cleanup);

	glutMainLoop();

    freeBoids(h_boids);
}
