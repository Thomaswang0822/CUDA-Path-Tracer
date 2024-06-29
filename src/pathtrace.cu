#include "pathtrace.h"
#include "materials.h"
#include "sampler.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>  // "let VS know" blockIdx etc.
#include <thrust/device_ptr.h>
#include <thrust/remove.h>


#define PixelIdxForTerminated -1
static Scene* hst_scene = nullptr;
static GuiDataContainer* guiData = nullptr;
static glm::vec3* dev_image = nullptr;
static Intersection* dev_intersections = nullptr;
// One for running kernels, the other for storage
static PathSegment* paths_alive = nullptr;
static PathSegment* paths_done = nullptr;
static thrust::device_ptr<PathSegment> thr_paths_alive;
static thrust::device_ptr<PathSegment> thr_paths_done;

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&paths_alive, pixelcount * sizeof(PathSegment));
	thr_paths_alive = thrust::device_ptr<PathSegment>(paths_alive);
	cudaMalloc(&paths_done, pixelcount * sizeof(PathSegment));
	thr_paths_done = thrust::device_ptr<PathSegment>(paths_done);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(Intersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(Intersection));

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(paths_alive);
	cudaFree(paths_done);
	cudaFree(dev_intersections);
	
	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(const Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		//segment.ray.origin = cam.position;
		segment.throughput = glm::vec3(1.0f);
		segment.radiance = glm::vec3(0.f);

		// antialiasing by jittering the ray
		/*segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);*/
		Sampler sampler(iter, index, 0);
		cam.generateCameraRay(segment.ray, x, y, sampler.sample2D());

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth,
	int num_paths,
	PathSegment* pathSegments,
	DevScene* scene,
	Intersection* intersections
) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths) {
		PathSegment& pathSegment = pathSegments[path_index];
#if BVH_DEBUG_VISUALIZATION
		scene->visualizedIntersect(pathSegment.ray, intersections[path_index]);
#else
		Intersection intersec;
		scene->intersect(pathSegment.ray, intersec);

		if (intersec.primId != NullPrimitive) {
			if (scene->devMaterials[intersec.materialId].type == Material::Type::Light) {
#if SCENE_LIGHT_SINGLE_SIDED
				if (glm::dot(intersec.normal, pathSegment.ray.direction) < 0.f) {
					intersec.primId = NullPrimitive;
				}
				else
#endif // SCENE_LIGHT_SINGLE_SIDED
					if (depth != 0) {
						// If not first ray, preserve previous sampling information for
						// MIS calculation
						intersec.prevPos = pathSegment.ray.origin;
						// intersec.prevBSDFPdf = segment.BSDFPdf;
					}
			}
			else {
				intersec.wo = -pathSegment.ray.direction;
			}
		}
		intersections[path_index] = intersec;
#endif // BVH_DEBUG_VISUALIZATION
	}  // end if (path_index < num_paths)
}

__global__ void shadeSegment(
	int iter,
	int depth,
	int numPaths,
	Intersection* intersections,
	PathSegment* segments,
	DevScene* scene
) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	// Should be unnecessary with stream compaction implemented
	if (idx >= numPaths) {
		return;
	}

	// Deal with miss
	Intersection& intersec = intersections[idx];
	if (intersec.primId == NullPrimitive) {
		segments[idx].remainingBounces = 0;
		return;
	}

#if BVH_DEBUG_VISUALIZATION
	float logDepth = 0.f;
	int size = scene->BVHSize;
	while (size) {
		logDepth += 1.f;
		size >>= 1;
	}
	segment.radiance = glm::vec3(float(intersec.primId) / logDepth * .1f);
	//segment.radiance = intersec.primId > 16 ? glm::vec3(1.f) : glm::vec3(0.f);
	segment.remainingBounces = 0;
	return;
#endif

	Sampler sampler(iter, idx, segments[idx].remainingBounces);
	const Material& material = scene->devMaterials[intersec.materialId];
	PathSegment& segment = segments[idx];
	glm::vec3 accRadiance(0.f);

	//bool deltaBSDF = material.type == Material::Type::Dielectric;

	/// If hit a light source
	if (material.type == Material::Type::Light) {
		glm::vec3 radiance = material.baseColor * material.emittance;
		if (depth == 0 || segment.isDeltaSample) {
			// If this is the first bounce or if we just had a specular bounce
			accRadiance += radiance * segment.throughput;
		}
		else {
			/// previous shading point (ray.origin) bounced off to a light
			/// so we do MIS
			float lightPdf = Math::pdfAreaToSolidAngle(Math::luminance(radiance) * scene->sumLightPowerInv,
				intersec.prevPos, intersec.position, intersec.normal);
			float BSDFPdf = segment.BSDFpdf;
			accRadiance += radiance * segment.throughput * Math::powerHeuristic(BSDFPdf, lightPdf);
		}
		// stop bouncing in both cases
		segment.remainingBounces = 0;
	}
	/// Do MIS
	else {
		bool deltaBSDF = (material.type == Material::Type::Dielectric);
		if (!deltaBSDF && glm::dot(intersec.normal, intersec.wo) < 0.f) {
			// other than Dielectric (glass), we don't allow refraction.
			intersec.normal = -intersec.normal;
		}

		// Light Sampling: accumulate radiance right away
		if (!deltaBSDF) {
			glm::vec3 radiance;
			glm::vec3 wi;
			float lightPdf = scene->sampleDirectLight(intersec.position, sampler.sample4D(), radiance, wi);

			if (lightPdf > 0.f) {
				float BSDFPdf = material.pdf(intersec.normal, intersec.wo, wi);
				accRadiance += segment.throughput *
					material.BSDF(intersec.normal, intersec.wo, wi) *
					radiance *
					Math::nonnegativeDot(intersec.normal, wi) /
					lightPdf * Math::powerHeuristic(lightPdf, BSDFPdf);
			}
		}

		// BSDF sampling: postpone to next bounce
		BSDFSample sample;
		material.sample(intersec.normal, intersec.wo, sampler.sample3D(), sample);

		if (sample.type == BSDFSampleType::Invalid) {
			// Terminate path if sampling fails
			segment.remainingBounces = 0;
		}
		else {
			bool deltaSample = (sample.type & BSDFSampleType::Specular);
			segment.throughput *= sample.bsdf / sample.pdf *
				(deltaSample ? 1.f : Math::absDot(intersec.normal, sample.dir));
			segment.ray = Ray::makeOffsetRay(intersec.position, sample.dir);
			segment.BSDFpdf = sample.pdf;
			segment.isDeltaSample = deltaSample;
			segment.remainingBounces--;
		}
	}
	segment.radiance += accRadiance;
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		const PathSegment& iterationPath = iterationPaths[index];
		//image[iterationPath.pixelIndex] += iterationPath.throughput;
		if (iterationPath.pixelIndex >= 0 && iterationPath.remainingBounces <= 0) {
			if (Debug::isNanInf(iterationPath.radiance)) {
				return;
			}
			image[iterationPath.pixelIndex] += glm::clamp(iterationPath.radiance,
				glm::vec3(0.f), glm::vec3(1e6f));
		}
	}
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 colorRawAvg = image[index] / float(iter);

		// Do ACES tone mapping and Gamma correction
		glm::vec3 colorOut = Math::correctGamma(Math::mapACES(colorRawAvg));
		glm::vec3 intColor = glm::clamp(glm::ivec3(colorOut * 255.f), glm::ivec3(0.f), glm::ivec3(255.f));

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = intColor.x;
		pbo[index].y = intColor.y;
		pbo[index].z = intColor.z;
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {

	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y
	);

	// 1D block for path tracing
	const int blockSize1d = 128;

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d>>> (cam, iter, traceDepth, paths_alive);
	checkCUDAError("generateRayFromCamera");
	cudaDeviceSynchronize();

	// increment each iteration
	int depth = 0;
	int num_paths = pixelcount;

	/// @note With stream compaction, thr_paths_done points to the start of terminated paths data.
	/// And we also need a running pointer that gives the next clean memory address,
	/// which enables more data being written in the next iteration.
	auto next_thr_paths_done = thr_paths_done;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(Intersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
			depth,
			num_paths,
			paths_alive,  // PathSegment*
			hst_scene->devScene,
			dev_intersections
		);
		checkCUDAError("computeIntersections");
		cudaDeviceSynchronize();
		//depth++;

		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// TODO:
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.

		//shadeFakeMaterial <<<numblocksPathSegmentTracing, blockSize1d>>> (iter, num_paths, dev_intersections, paths_alive, dev_materials);
		shadeSegment << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter, 
			depth,
			num_paths, 
			dev_intersections, 
			paths_alive, 
			hst_scene->devScene
		);
		checkCUDAError("shadeSegment");
		cudaDeviceSynchronize();

		/**
		 * @brief Compact paths that are terminated but carry contribution into a separate buffer.
		 * It copies to next_thr_paths_done and advance it to next clean memory address, but thr_paths_alive isn't shortened.
		 * 
		 * @see https://nvidia.github.io/cccl/thrust/api/function_group__stream__compaction_1gaeec02acfde68e411ca7d09063241f4d7.html#thrust-remove-copy-if.
		 */
		next_thr_paths_done = thrust::remove_copy_if(thr_paths_alive, thr_paths_alive + num_paths, next_thr_paths_done, CompactTerminatedPaths());
		// Remove paths that yield no contribution
		/**
		 * @brief Remove paths that yield no contribution.
		 * 
		 * @see https://nvidia.github.io/cccl/thrust/api/function_group__stream__compaction_1gaf01d45b30fecba794afae065d625f94f.html#thrust-remove-if
		 */
		auto thr_paths_alive_end = thrust::remove_if(thr_paths_alive, thr_paths_alive + num_paths, RemoveInvalidPaths());
		num_paths = static_cast<int>(thr_paths_alive_end - thr_paths_alive);
		//std::cout << "Remaining paths: " << num_paths << "\n";

		iterationComplete = bool(num_paths == 0);
		depth++;

		if (guiData != nullptr)
		{
			guiData->TracedDepth = depth;
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	int numEffectivePaths = static_cast<int>(next_thr_paths_done.get() - paths_done);
	finalGather <<<numBlocksPixels, blockSize1d >>> (numEffectivePaths, dev_image, paths_done);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO <<<blocksPerGrid2d, blockSize2d>>> (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	Cuda::memcpyDevToHost(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3));

	checkCUDAError("pathtrace");
}
