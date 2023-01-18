// OpenGL Graphics includes
#include "Dependencies/GL/glew.h"
#include "Dependencies/GL/freeglut.h"
#include "Dependencies/Helpers/helper_gl.h"

// CUDA runtime
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

// CUDA utilities
#include "Dependencies/Helpers/helper_cuda.h"  // includes cuda.h and cuda_runtime_api.h
#include "Dependencies/Helpers/helper_timer.h"

// File includes
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

#define CUBE_SIZE 2
#define NUM_BOIDS 10000
#define NUM_BOIDS_CHANGE 1000
#define BOID_CHANGE_SPEED 0.001f
#define BOID_CHANGE_COHESION 0.005f
#define BOID_CHANGE_SEPARATION 0.005f
#define BOID_CHANGE_ALIGNMENT 0.005f

const int width = 640, height = 480;

// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = { 0, 0, -3 };
float camera_rot[] = { 0, 0, 0 };
float camera_trans_lag[] = { 0, 0, -3 };
float camera_rot_lag[] = { 0, 0, 0 };
const float inertia = 0.1f;

bool bPause = false;
bool wireframe = false;
bool useCPU = false;

int numBoids = NUM_BOIDS;

// fps
static int fpsCount = 0;
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

void InitializeBoids(int boidCount) {
    initBoids(h_boids, numBoids);
    checkCudaErrors(cudaMalloc((void**)&d_boids, sizeof(s_boids)));
    initBoidsGPU(*d_boids, helper, numBoids);
    printf("Boid count: %d\n", h_boids.count);
    randomizeBoids(h_boids, 0.01f, 0.1f, -1, 1);
    generateBoidVertices(&h_boids);
}

void ReinitializeBoids(int boidCount) {
    cleanup();
    InitializeBoids(boidCount);
    sdkCreateTimer(&timer);
    sdkCreateTimer(&renderTimer);
    sdkResetTimer(&timer);
    sdkResetTimer(&renderTimer);
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

void computeFPS_CPU(float computationTime, float loopTime) {
    fpsCount++;

    timeFor15frames += loopTime;

    if (fpsCount >= 15) {
        char fps[256];
        float ifps = 1.0f / timeFor15frames * 15.0f;
        sprintf(fps, "Fish (%d boids) - CPU: %1.5f, %3.1f fps",
            numBoids, computationTime, ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;
        timeFor15frames = 0;
    }
}

void computeFPS_GPU(float copyToDeviceTime, float computationTime, float copyToHostTime, float loopTime) {
    fpsCount++;

    timeFor15frames += loopTime;

    if (fpsCount >= 15) {
        char fps[256];
        float ifps = 1.0f / timeFor15frames * 15.0f;
        sprintf(fps, "Fish (%d boids) - GPU: %1.5f + %1.8f + %1.5f, %3.1f fps",
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
            sdkResetTimer(&timer);
            sdkStartTimer(&timer);

            moveBoidsCPU(&h_boids);
            computationTime = 0.001f * sdkGetTimerValue(&timer);
            sdkStopTimer(&timer);
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
			moveBoidsGPU<<<blocks, threads>>>(d_boids);
            computationTime = 0.001f * sdkGetTimerValue(&timer);

            sdkResetTimer(&timer);
            sdkStartTimer(&timer);

			copyDeviceToHost_boids(h_boids, helper, *d_boids);
            copyToHostTime = 0.001f * sdkGetTimerValue(&timer);

            sdkStopTimer(&timer);
        }

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
    glutWireCube(CUBE_SIZE);

    sdkStopTimer(&renderTimer);
    renderTime = 0.001f * sdkGetTimerValue(&renderTimer);
    sdkResetTimer(&renderTimer);

    glutSwapBuffers();
    glutReportErrors();

    if (useCPU) {
        computeFPS_CPU(computationTime, renderTime);
    }
    else {
        computeFPS_GPU(copyToDeviceTime, computationTime, copyToHostTime, renderTime);    
    }
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
        fpsCount = 0;
        break;
    case 'g':
        useCPU = true;
        fpsCount = 0;
        break;
    case '+':
        numBoids += NUM_BOIDS_CHANGE;
        ReinitializeBoids(numBoids);
        fpsCount = 0;
        break;
    case '-':
        numBoids -= NUM_BOIDS_CHANGE;
        ReinitializeBoids(numBoids);
        fpsCount = 0;
        break;
    case 'Q':
        h_boids.simulationOptions.boidSpeed += BOID_CHANGE_SPEED;
        break;
    case 'q':
        h_boids.simulationOptions.boidSpeed -= BOID_CHANGE_SPEED;
        if (h_boids.simulationOptions.boidSpeed < 0)
            h_boids.simulationOptions.boidSpeed = 0;
        break;
    case 'W':
        h_boids.simulationOptions.cohesionFactor += BOID_CHANGE_COHESION;
        break;
    case 'w':
        h_boids.simulationOptions.cohesionFactor -= BOID_CHANGE_COHESION;
        break;
    case 'E':
        h_boids.simulationOptions.separationFactor += BOID_CHANGE_SEPARATION;
        break;
    case 'e':
        h_boids.simulationOptions.separationFactor -= BOID_CHANGE_SEPARATION;
        break;
    case 'R':
        h_boids.simulationOptions.alignFactor += BOID_CHANGE_ALIGNMENT;
        break;
    case 'r':
        h_boids.simulationOptions.alignFactor -= BOID_CHANGE_ALIGNMENT;
        break;
    case 'z':
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

void printToolTip() {
    printf("Welcome to the Shoal of Fish simulation\n");
    printf("Simulation controls:\n");
    printf("Use mouse dragging to rotate the scene\n");
    printf("Hold CTRL + mouse dragging - zoom in, zoom out\n");
    printf("Hold Shift + mouse dragging - camera panning\n");
    printf("SPACE - pause/unpause simulation\n");
    printf("G/g - calculate boid movement using GPU/CPU\n");
    printf("+/- - increase/decrease boid number (1000 per step)\n");
    printf("Q/q - increase/decrease boid speed\n");
    printf("W/w - increase/decrease flock cohesion factor\n");
    printf("E/e - increase/decrease boid separation factor\n");
    printf("R/r - increase/decrease boid alignment factor\n");
    printf("z - exit application");
    printf("When using the GPU for calculations, the values displayed\n");
    printf("on the appbar mark:\n");
    printf("1. Time to copy data onto the GPU\n");
    printf("2. Computation time\n");
    printf("3. Time to copy data back from the GPU\n");
    printf("When using the CPU, only the computation time will be displayed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    printToolTip();

    initGL(&argc, argv);
    InitializeBoids(numBoids);
    sdkCreateTimer(&timer);
    sdkCreateTimer(&renderTimer);

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
