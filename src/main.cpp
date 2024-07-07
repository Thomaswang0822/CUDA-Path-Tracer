#include "main.h"
#include "preview.h"
#include "intersections.h"
#include <cstring>
#include <random>
#include <filesystem>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

//static float dtheta = 0, dphi = 0;
//static glm::vec3 cammove;
//
//float zoom, theta, phi;
//glm::vec3 cameraPosition;
//glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
RenderState* renderState;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
	startTimeString = currentTimeString();

	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}

	const char* sceneFile = argv[1];

	// Load scene file
	scene = new Scene(sceneFile);
	scene->buildDevData();

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	renderState = &scene->state;
	Camera& cam = scene->camera;
	width = cam.resolution.x;
	height = cam.resolution.y;

	// Initialize CUDA and GL components
	init();

	// GLFW main loop
	mainLoop();

	scene->clear();
	Resource::clear();

	return 0;
}

void saveImage() {
#if AVERAGE_SPP
	float spp = iteration;
#else
	float spp = 1.0f;
#endif // AVERAGE_SPP
	
	// output image file
	Image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 color = renderState->image[index] / spp;
			switch (Settings::toneMapping) {
			case ToneMapping::Filmic:
				color = Math::filmic(color);
				break;
			case ToneMapping::ACES:
				color = Math::ACES(color);
				break;
			case ToneMapping::None:
				break;
			}
			color = Math::correctGamma(color);
			img.setPixel(width - 1 - x, y, color);
		}
	}

	std::string filename = renderState->imageName;
	std::ostringstream ss;
	ss << filename << "." << startTimeString << "." << spp << "spp";
	if (Settings::tracer == Tracer::ReSTIR_DI) {
		ss << ReSTIRSettings::M_Light << "Ml" << ReSTIRSettings::M_BSDF << "Mb";
	}
	filename = ss.str();

	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
	if (State::camChanged) {
		iteration = 0;
		scene->camera.update();
		State::camChanged = false;
	}

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (iteration == 0) {
		pathTraceFree();
		pathTraceInit(scene);
	}

	if (iteration < renderState->iterations) {
		uchar4* pbo_dptr = NULL;
		iteration++;
		cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

		// execute the kernel
		int frame = 0;
		pathTrace(pbo_dptr, frame, iteration);

		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
	}
	else {
		saveImage();
		pathTraceFree();
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	Camera& cam = scene->camera;

	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			saveImage();
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_S:
			saveImage();
			break;
		case GLFW_KEY_T:
			Settings::toneMapping = (Settings::toneMapping + 1) % 3;
			break;
		case GLFW_KEY_LEFT_SHIFT:
			cam.position += glm::vec3(0.f, -.1f, 0.f);
			State::camChanged = true;
			break;
		case GLFW_KEY_SPACE:
			cam.position += glm::vec3(0.f, .1f, 0.f);
			State::camChanged = true;
			break;
		case GLFW_KEY_R:
			State::camChanged = true;
			break;
		}
	}
}

void mouseScrollCallback(GLFWwindow* window, double offsetX, double offsetY) {
	scene->camera.fov.y -= offsetY;
	scene->camera.fov.y = std::min(scene->camera.fov.y, 45.f);
	State::camChanged = true;
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (MouseOverImGuiWindow()) {
		return;
	}
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	Camera& cam = scene->camera;

	if (xpos == lastX || ypos == lastY) {
		return; // otherwise, clicking back into window causes re-start
	}

	if (leftMousePressed) {
		// compute new camera parameters
		cam.rotation.x -= (xpos - lastX) / width * 40.f;
		cam.rotation.y += (ypos - lastY) / height * 40.f;
		cam.rotation.y = glm::clamp(cam.rotation.y, -89.9f, 89.9f);
		State::camChanged = true;
	}
	else if (rightMousePressed) {
		float dy = (ypos - lastY) / height;
		cam.position.y += dy;
		State::camChanged = true;
	}
	else if (middleMousePressed) {
		renderState = &scene->state;
		glm::vec3 forward = cam.view;
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		glm::vec3 right = cam.right;
		right.y = 0.0f;
		right = glm::normalize(right);

		cam.position -= (float)(xpos - lastX) * right * 0.01f;
		cam.position += (float)(ypos - lastY) * forward * 0.01f;
		State::camChanged = true;
	}
	lastX = xpos;
	lastY = ypos;
}