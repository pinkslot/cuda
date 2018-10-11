/*
*  CUDA based triangle mesh path tracer using BVH acceleration by Sam lapere, 2016
*  BVH implementation based on real-time CUDA ray tracer by Thanassis Tsiodras, 
*  http://users.softlab.ntua.gr/~ttsiod/cudarenderer-BVH.html 
*  Interactive camera with depth of field based on CUDA path tracer code 
*  by Peter Kutz and Yining Karl Li, https://github.com/peterkutz/GPUPathTracer
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
 
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <math_functions.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "device_launch_parameters.h"
#include "cutil_math.h"
#include "C:\Program Files\NVIDIA Corporation\Installer2\CUDASamples_7.5.{D3FD22D5-4D82-406C-ADC6-962E5889C52D}\common\inc\GL\glew.h"
#include "C:\Program Files\NVIDIA Corporation\Installer2\CUDASamples_7.5.{D3FD22D5-4D82-406C-ADC6-962E5889C52D}\common\inc\GL\freeglut.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_pathtracer.h"

#define TWO_PI 6.2831853071795864769252867665590057683943f
#define NUDGE_FACTOR     1e-3f  // epsilon
#define samps  1 // samples
#define BVH_STACK_SIZE 32
#define SCREEN_DIST (height*2)

int texturewidth = 0;
int textureheight = 0;
int total_number_of_triangles;

__device__ int depth = 0;


// Textures for vertices, triangles and BVH data
// (see CudaRender() below, as well as main() to see the data setup process)
texture<uint1, 1, cudaReadModeElementType> g_triIdxListTexture;
texture<float2, 1, cudaReadModeElementType> g_pCFBVHlimitsTexture;
texture<uint4, 1, cudaReadModeElementType> g_pCFBVHindexesOrTrilistsTexture;
texture<float4, 1, cudaReadModeElementType> g_trianglesTexture;

Vertex* cudaVertices;
float* cudaTriangleIntersectionData;
int* cudaTriIdxList = NULL;
float* cudaBVHlimits = NULL;
int* cudaBVHindexesOrTrilists = NULL;
Triangle* cudaTriangles = NULL;
Camera* cudaRendercam = NULL;


struct Ray {
	Vector3Df orig;	// ray origin
	Vector3Df dir;		// ray direction
	__device__ Ray(Vector3Df o_, Vector3Df d_) : orig(o_), dir(d_) {}
};

struct Media {
	float3 lambda;
	float mu, k, spec_frac, dif_refl_frac;
};

struct Source {
	float3 dir;
	float pow, wide;
};

struct Sphere {
	float rad;
	float3 pos;
	Media med;
	Source src;

	__device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit 
		// Ray/sphere intersection
		// Quadratic formula required to solve ax^2 + bx + c = 0 
		// Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 

		Vector3Df op = Vector3Df(pos) - r.orig;  //
		float t;
		float b = dot(op, r.dir);
		float disc = b*b - dot(op, op) + rad*rad; // discriminant
		if (disc<0)
			return -1.f;
		disc = sqrtf(disc);
		return (t = b - disc) > 0 ? t : ((t = b + disc) > 0 ? t : -1.f);
	}
};

#define	INIT_SPHERE(...) { ##__VA_ARGS__,  },

#define clearMedia { { 0.f, 0.f, 0.f }, 0.f, 1., 1., 1. }
#define superMedia { { .6f, .6f, .9f }, .001f, 1., 1.0, .0 }
#define superSrc { { 1, .001, 0.0 }, 5000., 10. }
#define glassMedia { { 0.f, 0.f, 0.f }, 0.f, 1.3, 1., 1. }
#define bubbleMedia { { 0.f, 0.f, 0.f }, 0.f, 1.1, 1., 1. }
#define mirrorMedia { { 0.f, 0.f, 0.f }, 0.f, 5, 1., 0. }
#define transpMedia { { 0.f, 0.f, 0.f }, 0.f, .001, 0., 0. }
#define diffMedia { { 0.f, 0.f, 0.f }, 0.f, 1., 0., 1.0 }
#define scatMedia { { .1f, .7f, .8f }, 2.3, 1., 1.0, .0 }
#define objectMedia { { .1f, .7f, .8f }, 2.3, 1.3, 0.5, 1. }
#define TIME 100.
#define TIME_WIDTH 6.

#define emptySrc { { 0, 0, 0 }, 0. }
#define RAD 440.f

__device__ Sphere spheres[] = {
	{ 
		RAD, { 20.f - RAD, 0.f, 0.0f },
		superMedia,
		superSrc
	}, 
	/*{
		.3f, { -.5f, 1.4f, 0.f },
		mirrorMedia,
		{ { .0, -1.0, 0.0 }, 70., .3 },
	},{
		.4f, { .0f, -0.5f, -1.f },
		objectMedia,
		emptySrc
	},
	{
		.4f, { -.5f, -0.5f, -.0f },
		mirrorMedia,
		emptySrc
	},
	{
		.4f, { .5f, -0.5f, -.0f },
		diffMedia,
		emptySrc
	},*/
	{
		1000000.f, { .0f, -1000001.f, .0f },
		diffMedia,
		emptySrc
	},
//	{ .7f, { 0, -1 + .7, 0 }, objectMedia, emptySrc },
	{ .5f, { -1.051436f, -1 + .5, .772216f }, diffMedia, emptySrc },
	{ .65f, { 1.679223f, -1 + .65, 1.592339f }, scatMedia, emptySrc },
	{ 0.4f, { -2.f, -1 + .4, -1.430057f }, scatMedia, emptySrc },
	{ 0.5f, { -0.102131f, -1 + .5, -1.430057f }, mirrorMedia, emptySrc },
	{ 0.4f, { 1.085430f, -1 + .4, -.848117f }, diffMedia, emptySrc },
	{ .5f, { 1.50875f, -1 + .5, -1.837097f }, mirrorMedia, emptySrc },
	

};

// Create OpenGL BGR value for assignment in OpenGL VBO buffer
__device__ int getColor(Vector3Df& p)  // converts Vector3Df colour to int
{
	return (((unsigned)p.z) << 16) | (((unsigned)p.y) << 8) | (((unsigned)p.x));
}

// Helper function, that checks whether a ray intersects a bounding box (BVH node)
__device__ bool RayIntersectsBox(const Vector3Df& originInWorldSpace, const Vector3Df& rayInWorldSpace, int boxIdx)
{
	// set Tnear = - infinity, Tfar = infinity
	//
	// For each pair of planes P associated with X, Y, and Z do:
	//     (example using X planes)
	//     if direction Xd = 0 then the ray is parallel to the X planes, so
	//         if origin Xo is not between the slabs ( Xo < Xl or Xo > Xh) then
	//             return false
	//     else, if the ray is not parallel to the plane then
	//     begin
	//         compute the intersection distance of the planes
	//         T1 = (Xl - Xo) / Xd
	//         T2 = (Xh - Xo) / Xd
	//         If T1 > T2 swap (T1, T2) /* since T1 intersection with near plane */
	//         If T1 > Tnear set Tnear =T1 /* want largest Tnear */
	//         If T2 < Tfar set Tfar="T2" /* want smallest Tfar */
	//         If Tnear > Tfar box is missed so
	//             return false
	//         If Tfar < 0 box is behind ray
	//             return false
	//     end
	// end of for loop

	float Tnear, Tfar;
	Tnear = -FLT_MAX;
	Tfar = FLT_MAX;

	float2 limits;

// box intersection routine
#define CHECK_NEAR_AND_FAR_INTERSECTION(c)							    \
    if (rayInWorldSpace.##c == 0.f) {						    \
	if (originInWorldSpace.##c < limits.x) return false;					    \
	if (originInWorldSpace.##c > limits.y) return false;					    \
	} else {											    \
	float T1 = (limits.x - originInWorldSpace.##c)/rayInWorldSpace.##c;			    \
	float T2 = (limits.y - originInWorldSpace.##c)/rayInWorldSpace.##c;			    \
	if (T1>T2) { float tmp=T1; T1=T2; T2=tmp; }						    \
	if (T1 > Tnear) Tnear = T1;								    \
	if (T2 < Tfar)  Tfar = T2;								    \
	if (Tnear > Tfar)	return false;									    \
	if (Tfar < 0.f)	return false;									    \
	}

	limits = tex1Dfetch(g_pCFBVHlimitsTexture, 3 * boxIdx); // box.bottom._x/top._x placed in limits.x/limits.y
	//limits = make_float2(cudaBVHlimits[6 * boxIdx + 0], cudaBVHlimits[6 * boxIdx + 1]);
	CHECK_NEAR_AND_FAR_INTERSECTION(x)
	limits = tex1Dfetch(g_pCFBVHlimitsTexture, 3 * boxIdx + 1); // box.bottom._y/top._y placed in limits.x/limits.y
	//limits = make_float2(cudaBVHlimits[6 * boxIdx + 2], cudaBVHlimits[6 * boxIdx + 3]);
	CHECK_NEAR_AND_FAR_INTERSECTION(y)
	limits = tex1Dfetch(g_pCFBVHlimitsTexture, 3 * boxIdx + 2); // box.bottom._z/top._z placed in limits.x/limits.y
	//limits = make_float2(cudaBVHlimits[6 * boxIdx + 4], cudaBVHlimits[6 * boxIdx + 5]);
	CHECK_NEAR_AND_FAR_INTERSECTION(z)

	// If Box survived all above tests, return true with intersection point Tnear and exit point Tfar.
	return true;
}


//////////////////////////////////////////
//	BVH intersection routine	//
//	using CUDA texture memory	//
//////////////////////////////////////////

// there are 3 forms of the BVH: a "pure" BVH, a cache-friendly BVH (taking up less memory space than the pure BVH)
// and a "textured" BVH which stores its data in CUDA texture memory (which is cached). The last one is gives the 
// best performance and is used here.

__device__ bool BVH_IntersectTriangles(
	int* cudaBVHindexesOrTrilists, const Vector3Df& origin, const Vector3Df& ray, unsigned avoidSelf,
	int& pBestTriIdx, Vector3Df& pointHitInWorldSpace, float& hitdist,
	float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList, Vector3Df& boxnormal)
{
	// in the loop below, maintain the closest triangle and the point where we hit it:
	pBestTriIdx = -1;
	float bestTriDist;

	// start from infinity
	bestTriDist = FLT_MAX;

	// create a stack for each ray
	// the stack is just a fixed size array of indices to BVH nodes
	int stack[BVH_STACK_SIZE];
	
	int stackIdx = 0;
	stack[stackIdx++] = 0; 
	Vector3Df hitpoint;

	// while the stack is not empty
	while (stackIdx) {
		
		// pop a BVH node (or AABB, Axis Aligned Bounding Box) from the stack
		int boxIdx = stack[stackIdx - 1];
		//uint* pCurrent = &cudaBVHindexesOrTrilists[boxIdx]; 
		
		// decrement the stackindex
		stackIdx--;

		// fetch the data (indices to childnodes or index in triangle list + trianglecount) associated with this node
		uint4 data = tex1Dfetch(g_pCFBVHindexesOrTrilistsTexture, boxIdx);

		// texture memory BVH form...
		// determine if BVH node is an inner node or a leaf node by checking the highest bit (bitwise AND operation)
		// inner node if highest bit is 1, leaf node if 0

		if (!(data.x & 0x80000000)) {   // INNER NODE

			// if ray intersects inner node, push indices of left and right child nodes on the stack
			if (RayIntersectsBox(origin, ray, boxIdx)) {
				stack[stackIdx++] = data.y; // right child node index
				stack[stackIdx++] = data.z; // left child node index
				// return if stack size is exceeded
				if (stackIdx>BVH_STACK_SIZE)
				{
					return false; 
				}
			}
		}
		else { // LEAF NODE
			for (unsigned i = data.w; i < data.w + (data.x & 0x7fffffff); i++) {
				// fetch the index of the current triangle
				int idx = tex1Dfetch(g_triIdxListTexture, i).x;
				// check if triangle is the same as the one intersected by previous ray	
				// to avoid self-reflections/refractions
				if (avoidSelf == idx)
					continue; 
				// fetch triangle center and normal from texture memory
				float4 center = tex1Dfetch(g_trianglesTexture, 5 * idx);
				float4 normal = tex1Dfetch(g_trianglesTexture, 5 * idx + 1);
				// use the pre-computed triangle intersection data: normal, d, e1/d1, e2/d2, e3/d3
				float k = dot(normal, ray);
				if (k == 0.0f)
					continue; // this triangle is parallel to the ray, ignore it.
				float s = (normal.w - dot(normal, origin)) / k;
				if (s <= 0.0f) // this triangle is "behind" the origin.
					continue;
				if (s <= NUDGE_FACTOR)  // epsilon
					continue;
				Vector3Df hit = ray * s;
				hit += origin;

				// ray triangle intersection
				// Is the intersection of the ray with the triangle's plane INSIDE the triangle?
				float4 ee1 = tex1Dfetch(g_trianglesTexture, 5 * idx + 2);
				float kt1 = dot(ee1, hit) - ee1.w; 
				if (kt1<0.0f) continue;
				float4 ee2 = tex1Dfetch(g_trianglesTexture, 5 * idx + 3);
				float kt2 = dot(ee2, hit) - ee2.w; 
				if (kt2<0.0f) continue;
				float4 ee3 = tex1Dfetch(g_trianglesTexture, 5 * idx + 4);
				float kt3 = dot(ee3, hit) - ee3.w; 
				if (kt3<0.0f) continue;
				// ray intersects triangle, "hit" is the world space coordinate of the intersection.
				{
					// is this intersection closer than all the others?
					float hitZ = distancesq(origin, hit);
					if (hitZ < bestTriDist) {
						// maintain the closest hit
						bestTriDist = hitZ;
						hitdist = sqrtf(bestTriDist);
						pBestTriIdx = idx;
						pointHitInWorldSpace = hit;
						// store barycentric coordinates (for texturing, not used for now)
					}
				}
			}
		}
	}
	
	return pBestTriIdx != -1;
}

__device__ void printv(Vector3Df &arr, char mark = ' ') {
	printf("%f, %f, %f %c%c%c\n", arr.x, arr.y, arr.z, mark, mark, mark);
}

__device__ float sqr(float x) {
	return x * x;
}

//////////////////////
// PATH TRACING
//////////////////////
struct PhasePoint {
	float time;
	Vector3Df orig;	// ray origin
	Vector3Df dir;		// ray direction
	int n;
	Vector3Df mask;
	Media *media;
	__device__ PhasePoint(float t, Vector3Df o_, Vector3Df d_, Media *media) :
		time(t), orig(o_), dir(d_), n(0), mask(1.0f, 1.0f, 1.0f), media(media) {}
	__device__ PhasePoint(float t, Vector3Df o_, Vector3Df d_, int n_, Vector3Df mask_, Media *media) :
		time(t), orig(o_), dir(d_), n(n_), mask(mask_), media(media) {}
	__device__ PhasePoint() {}
};

__device__ void rand_dir(
	curandState *randstate, 
	Vector3Df *new_dir, 
	Vector3Df *old_dir = nullptr, 
	bool only_pos = false,
	bool custom_indic = false,
	float indic = -1)
{
	if (!custom_indic) {
		indic = curand_uniform(randstate);
		if (!only_pos) {
			indic = indic * 2. - 1.;
		}
	}

	float phi = curand_uniform(randstate) * 2 * M_PI,
		sin_ind = sqrt(1 - indic * indic);

	*new_dir = Vector3Df(cos(phi) * sin_ind, sin(phi) * sin_ind, indic);

	if (!old_dir) {
		return;
	}

	float x1 = old_dir->x, x2 = old_dir->y, x3 = old_dir->z;
	if (abs(x3 - 1) > 1e-5) {
		Vector3Df rand_dir = *new_dir;
		float denom = sqrt(1 - sqr(x3));

		new_dir->x = dot(Vector3Df(x1 * x3 / denom,		-x2 / denom,	x1), rand_dir);
		new_dir->y = dot(Vector3Df(x2 * x3 / denom,		x1 / denom,		x2), rand_dir);
		new_dir->z = dot(Vector3Df(-denom,				0,				x3), rand_dir);
	}
	new_dir->normalize();
}

__device__ float normal_dist(float x, float a, float b) {
	return exp(-sqr(x - a) / 2. / sqr(b)) / b / sqrt(2. * M_PI);
}

__device__ float a2indic(float a, float neworig_len)
{
	float b = RAD - neworig_len,
		sina2 = a / 2. / RAD,
		sqra = a * a;
	return sqrt(1 - sqra * (1. - sina2 * sina2) / (sqra + b*b - 2. * a * b * sina2));
}

__device__ float angle_diff(Vector3Df &a, Vector3Df &b) {
	return acos(min(1., dot(a, b) / a.length() / b.length()));
}


__device__ void rand_dir_to_src(
	curandState *randstate,
	float wide,
	Vector3Df *newdir,
	float *weight,
	bool useOpt,
	float neworig_len,
	Vector3Df *old_orig,
	Vector3Df *src_orig,
	bool print = false
) {
	if (useOpt) {
		wide *= 1.5;
		Vector3Df
			src_normal = Vector3Df(-1., 0., 0.),
			to_src = *src_orig - *old_orig,
			new_orig;
		float norm_sample = curand_normal(randstate) * wide,
			uni_sample = curand_uniform(randstate) * 2. * M_PI;
		new_orig = *src_orig + Vector3Df(0., cos(uni_sample), sin(uni_sample)) * norm_sample;
		*newdir = new_orig - *old_orig;
		newdir->normalize();

		// weight calc
		float delta = .01,
			uni_sample_delta = uni_sample + delta;
		Vector3Df norm_delta_orig = *src_orig + Vector3Df(0., cos(uni_sample), sin(uni_sample)) * (norm_sample + delta),
			norm_delta_dir = norm_delta_orig - *old_orig,	
			uni_delta_vect = *src_orig + Vector3Df(0., cos(uni_sample_delta), sin(uni_sample_delta)) * norm_sample - new_orig;
		norm_delta_dir.normalize();

		Vector3Df 
			phi_dir = norm_delta_orig - new_orig,
			psi_dir = cross(norm_delta_dir, *newdir);
		phi_dir.normalize();
		psi_dir.normalize();
		Vector3Df 
			uni_phi_delta_vect = phi_dir * dot(phi_dir, uni_delta_vect),
			uni_psi_delta_vect = psi_dir * dot(psi_dir, uni_delta_vect),
			uni_phi_delta_dir = uni_phi_delta_vect + new_orig - *old_orig,
			uni_psi_delta_dir = uni_psi_delta_vect + new_orig - *old_orig;
		
		uni_phi_delta_dir.normalize();
		uni_psi_delta_dir.normalize();

		float norm_phi_delta = angle_diff(norm_delta_dir, *newdir),
			uni_phi_delta = angle_diff(uni_phi_delta_dir, *newdir),
			uni_psi_delta = angle_diff(uni_psi_delta_dir, *newdir),
			phi_prob = normal_dist(norm_sample, 0., wide) * delta / norm_phi_delta,
			psi_prob = (delta / 2. / M_PI - uni_phi_delta * phi_prob) / uni_psi_delta;

		if (norm_phi_delta < 1e-5 || uni_psi_delta < 1e-5 || phi_prob < .03 || phi_prob < .03) {
			*weight = 0.;
			return;
		}
		*weight = 1. / 2. / M_PI / phi_prob / psi_prob;

/*		if (print) {
			printv(new_orig, '-');
			//printf("%f\n", phi_prob, psi_prob, (delta / 2. / M_PI) / uni_psi_delta);
		}*/

		return;
	}
	else {
		*weight = 1.;
		rand_dir(randstate, newdir);
	}
/*	float
		a = abs(curand_normal(randstate))* wide,
		rand_indic = a2indic(a, neworig_len);

	b.normalize();
	*weight = 1. / 4. / RAD / normal_dist(a, 0, wide);
	rand_dir(randstate, newdir, &b, false, true, rand_indic);
*/}

__device__ Vector3Df path_trace(curandState *randstate, Vector3Df rayorig, Vector3Df raydir, int avoidSelf, float time,
	Triangle *pTriangles, int* cudaBVHindexesOrTrilists, float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList, 
	bool useOpt)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x == 251 && y == 251) {
		return Vector3Df(1., 0., 0.);
	}
	avoidSelf = -1; // doesn't work because of stack emulation, should be included to PhasePoint
	Vector3Df ret;
#define N 10
	Media _objectMedia = objectMedia, *mainMedia = &spheres[0].med;
	Source *mainSrc = &spheres[0].src;
	Vector3Df SRC_ORIG = spheres[0].pos + spheres[0].src.dir * spheres[0].rad;
	PhasePoint stack[(N + 1) * 3] = { PhasePoint(time, rayorig, raydir, mainMedia) };
	int top = 1;
	while (top){
		PhasePoint &cur = stack[--top];
		if (cur.n >= N || cur.time < 0) {
			continue;
		}
		raydir = cur.dir;
		rayorig = cur.orig + raydir * NUDGE_FACTOR;
		time = cur.time;

		Vector3Df mask = cur.mask, neworig, boxnormal, normal;
		float hit_dist = 1e10;
		Media *new_media = nullptr, *old_media = cur.media;
		
		int pBestTriIdx;
		if (!BVH_IntersectTriangles(
			cudaBVHindexesOrTrilists, rayorig, raydir, avoidSelf,
			pBestTriIdx, neworig, hit_dist, cudaBVHlimits,
			cudaTriangleIntersectionData, cudaTriIdxList, boxnormal)) {
			hit_dist = 1e10;
		}

		float d;
		int sphere_id = -1;
		for (int i = sizeof(spheres) / sizeof(Sphere); i--;) {
			if ((d = spheres[i].intersect(Ray(rayorig, raydir))) > 0 && d < hit_dist) {
				hit_dist = d; sphere_id = i;
			}
		}

		if (hit_dist >= 1e10) {
			 continue;
		}

		// intersect all spheres in the scene
		Source *src = nullptr;
		neworig = rayorig + raydir * hit_dist;
		float new_time = cur.time - hit_dist * old_media->k; 
		if (sphere_id >= 0) {
			Sphere &closest = spheres[sphere_id];
			normal = neworig - closest.pos;
			normal.normalize();
			new_media = &closest.med;
			src = &closest.src;
		}
		else {
			normal = pTriangles[pBestTriIdx]._normal;
			new_media = &_objectMedia;
		}

		bool into = true;
		if (dot(normal, raydir) > 0) {
			into = false;
			normal *= -1;
			new_media = mainMedia;
		}
		
		float optical_dist = exp(-old_media->mu * hit_dist);
		if (src && src->pow > 0) {
			Vector3Df
				hit_orig = neworig;


			// colorize src
			// mask *= Vector3Df(dift * 5., 0, (2. - dift) * 5.);



			float dim_dist = (SRC_ORIG - hit_orig).lengthsq(),
				time_dist = sqr(new_time - TIME);
			ret += 
					mask
					* optical_dist
					* src->pow
					* normal_dist(dim_dist, 0., src->wide)
					* exp(-time_dist / TIME_WIDTH)
					;
		}
		
		{
#define BRANCHS 0
			bool coin;
			if (BRANCHS || (coin = (old_media->mu > NUDGE_FACTOR && (curand_uniform(randstate) > optical_dist)))) {
				// scattering
				Vector3Df a = SRC_ORIG - rayorig, newdir;
				float
					lengthA = a.length(),
					weight = 1.,
					dist = log((optical_dist - 1) * curand_uniform(randstate) + 1) / (-old_media->mu);
				
				Vector3Df scat_neworig = rayorig + raydir * dist;
				rand_dir_to_src(randstate, mainSrc->wide, &newdir, &weight, useOpt, scat_neworig.length(), &scat_neworig, &SRC_ORIG);

				weight = min(100., weight);
				stack[top++] = PhasePoint(cur.time - dist * old_media->k, 
					scat_neworig, newdir, cur.n + 1,
					Vector3Df(old_media->lambda) * mask * weight
#if BRANCHS
					* (1.f - optical_dist)
#endif					
				, old_media);
			}
			if (BRANCHS || !coin) {
				float spec_frac = new_media->spec_frac, 
					dif_refl_frac = new_media->dif_refl_frac;

				// border
				Vector3Df new_mask = mask;
#if BRANCHS
				new_mask *= optical_dist;
#endif
#define MEDIA_K 1.f  // Index of Refraction air
#define BRANCHB 0
				float fcoin = curand_uniform(randstate);
				unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
				unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
				if (BRANCHB || fcoin < spec_frac) {
					float k = old_media->k / new_media->k;  // IOR ratio of refractive materials
					float ddn = dot(raydir, normal);
					float cos2t = 1.0f - sqr(k) * (1.f - sqr(ddn));	
					Vector3Df rdir = raydir + normal * 2.0f;
					rdir.normalize();

					Vector3Df spec_lambda(.1, .9, .1);
					if (cos2t < 0.0f) // total internal reflection
					{
						stack[top++] = PhasePoint(new_time, neworig, rdir, cur.n + 1, spec_lambda * new_mask, old_media);
					}
					else // cos2t > 0
					{
						Vector3Df tdir = raydir * k + normal * (ddn * k + sqrtf(cos2t));
						tdir.normalize();
						float R0 = sqr(old_media->k - new_media->k) / sqr(old_media->k + new_media->k);
						float c = 1 - (into? -dot(raydir, normal) : -dot(tdir, normal));
						float R = (R0 + (1.f - R0) * c * c * c * c * c);
#if BRANCHB
						new_mask *= spec_frac;
#endif
						if (BRANCHB || fcoin > R) {
							stack[top++] = PhasePoint(
								new_time,
								neworig, tdir, 
								cur.n + 1, 
								new_mask * (BRANCHB ? (1 - R) : 1.f),
								new_media
							);
						}	
						if (BRANCHB || fcoin < R) {
							stack[top++] = PhasePoint(
								new_time,
								neworig, rdir, 
								cur.n + 1, 
								spec_lambda * new_mask * (BRANCHB ? R : 1.f),
								old_media
							);
						}
					}
				}
				if (BRANCHB || fcoin > spec_frac) {
					Vector3Df difLambda(.7, .2, .2);
#if BRANCHB
					new_mask *= (1- spec_frac) / spec_frac;
#endif
					float weight;
					Vector3Df back_norm = normal * -1;

					Vector3Df rdir;
					rand_dir_to_src(randstate, mainSrc->wide, &rdir, &weight, useOpt, neworig.length(), &neworig, &SRC_ORIG,
					x == 250 && y == 250
					);
					weight = min(100., weight);
					weight *= abs(dot(rdir, normal)) * dot(rdir, normal) > 0 ? dif_refl_frac : (1. - dif_refl_frac);

					new_mask *= max(0., weight);

					stack[top++] = PhasePoint(
						new_time,
						neworig, rdir,
						cur.n + 1,
						difLambda * new_mask * (BRANCHB ? ((1 - spec_frac) * (1 - dif_refl_frac)) : 1.f) * weight,
						old_media
						);
				}
				/*
				if (BRANCHB || fcoin > spec_frac + (1 - spec_frac) * dif_refl_frac) {
					Vector3Df tdir;
					rand_dir_to_src(randstate, mainSrc->wide, SRC_ORIG - neworig, &tdir, &weight, &back_norm, useOpt);
					weight = min(10., weight);
					new_mask *= max(0., dot(tdir, back_norm));

					stack[top++] = PhasePoint(
						new_time,
						neworig, tdir,
						cur.n + 1,
						difLambda * new_mask * (BRANCHB ? ((1 - spec_frac) * dif_refl_frac) : 1.f) * weight,
						old_media
					);
				}

				*/
			}
		}
	}
	return ret;
}
union Colour  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

// the core path tracing kernel, 
// running in parallel for all pixels
__global__ void CoreLoopPathTracingKernel(Vector3Df* output, Vector3Df* accumbuffer, Triangle* pTriangles, Camera* cudaRendercam,
	int* cudaBVHindexesOrTrilists, float* cudaBVHlimits, float* cudaTriangleIntersectionData,
	int* cudaTriIdxList, unsigned int framenumber, unsigned int hashedframenumber)
{

	// assign a CUDA thread to every pixel by using the threadIndex
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	// create random number generator and initialise with hashed frame number, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedframenumber + threadId, 0, 0, &randState);

	Vector3Df finalcol; // final pixel colour  
	Vector3Df rendercampos = Vector3Df(cudaRendercam->position.x, cudaRendercam->position.y, cudaRendercam->position.z);


	int i = (height - y - 1)*width + x; // pixel index in buffer
	int pixelx = x; // pixel x-coordinate on screen
	int pixely = height - y - 1; // pixel y-coordintate on screen

	finalcol = Vector3Df(0.0f, 0.0f, 0.0f); // reset colour to zero for every pixel	
	for (int s = 0; s < samps; s++){

		// compute primary ray direction
		// use camera view of current frame (transformed on CPU side) to create local orthonormal basis
		Vector3Df rendercamview = Vector3Df(cudaRendercam->view.x, cudaRendercam->view.y, cudaRendercam->view.z); rendercamview.normalize(); // view is already supposed to be normalized, but normalize it explicitly just in case.
		Vector3Df rendercamup = Vector3Df(cudaRendercam->up.x, cudaRendercam->up.y, cudaRendercam->up.z); rendercamup.normalize();

		Vector3Df horizontalAxis = cross(rendercamview, rendercamup); horizontalAxis.normalize(); // Important to normalize!
		Vector3Df verticalAxis = cross(horizontalAxis, rendercamview); verticalAxis.normalize(); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

		Vector3Df middle = rendercampos + rendercamview;
		Vector3Df horizontal = horizontalAxis * tanf(cudaRendercam->fov.x * 0.5 * (M_PI / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.
		Vector3Df vertical = verticalAxis * tanf(-cudaRendercam->fov.y * 0.5 * (M_PI / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.

		// anti-aliasing
		// calculate center of current pixel and add random number in X and Y dimension
		// based on https://github.com/peterkutz/GPUPathTracer 
		float jitterValueX = curand_uniform(&randState) - 0.5;
		float jitterValueY = curand_uniform(&randState) - 0.5;
		float sx = (jitterValueX + pixelx) / (cudaRendercam->resolution.x - 1);
		float sy = (jitterValueY + pixely) / (cudaRendercam->resolution.y - 1);

		// compute pixel on screen
		Vector3Df pointOnPlaneOneUnitAwayFromEye = middle + ( horizontal * ((2 * sx) - 1)) + ( vertical * ((2 * sy) - 1));
		Vector3Df pointOnImagePlane = rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * cudaRendercam->focalDistance); // Important for depth of field!		

		// calculation of depth of field / camera aperture 
		// based on https://github.com/peterkutz/GPUPathTracer 
		
		Vector3Df aperturePoint;

		if (cudaRendercam->apertureRadius > 0.00001) { // the small number is an epsilon value.
		
			// generate random numbers for sampling a point on the aperture
			float random1 = curand_uniform(&randState);
			float random2 = curand_uniform(&randState);

			// randomly pick a point on the circular aperture
			float angle = TWO_PI * random1;
			float distance = cudaRendercam->apertureRadius * sqrtf(random2);
			float apertureX = cos(angle) * distance;
			float apertureY = sin(angle) * distance;

			aperturePoint = rendercampos + (horizontalAxis * apertureX) + (verticalAxis * apertureY);
		}
		else { // zero aperture
			aperturePoint = rendercampos;
		}

		// calculate ray direction of next ray in path
		Vector3Df apertureToImagePlane = pointOnImagePlane - aperturePoint; 
		apertureToImagePlane.normalize(); // ray direction, needs to be normalised
		Vector3Df rayInWorldSpace = apertureToImagePlane;
		// in theory, this should not be required
		rayInWorldSpace.normalize();

		// origin of next ray in path
		Vector3Df originInWorldSpace = aperturePoint;

		finalcol += path_trace(&randState, originInWorldSpace, rayInWorldSpace, -1, cudaRendercam->time, pTriangles,
			cudaBVHindexesOrTrilists, cudaBVHlimits, cudaTriangleIntersectionData, cudaTriIdxList, cudaRendercam->useOpt) * (1.0f/samps);
	}       

	// add pixel colour to accumulation buffer (accumulates all samples) 
	accumbuffer[i] += finalcol;
	// averaged colour: divide colour by the number of calculated frames so far
	Vector3Df tempcol = accumbuffer[i] / framenumber;

	Colour fcolour;
	Vector3Df colour = Vector3Df(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f));
	// convert from 96-bit to 24-bit colour + perform gamma correction
	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255), (unsigned char)(powf(colour.y, 1 / 2.2f) * 255), (unsigned char)(powf(colour.z, 1 / 2.2f) * 255), 1);
	// store pixel coordinates and pixelcolour in OpenGL readable outputbuffer
	output[i] = Vector3Df(x, y, fcolour.c);

}

bool g_bFirstTime = true;

// the gateway to CUDA, called from C++ (in void disp() in main.cpp)
void cudarender(Vector3Df* dptr, Vector3Df* accumulatebuffer, Triangle* cudaTriangles, int* cudaBVHindexesOrTrilists,
	float* cudaBVHlimits, float* cudaTriangleIntersectionData, int* cudaTriIdxList, 
	unsigned framenumber, unsigned hashedframes, Camera* cudaRendercam){

	if (g_bFirstTime) {
		// if this is the first time cudarender() is called,
		// bind the scene data to CUDA textures!
		g_bFirstTime = false;

		printf("g_triIndexListNo: %d\n", g_triIndexListNo);
		printf("g_pCFBVH_No: %d\n", g_pCFBVH_No);
		printf("g_verticesNo: %d\n", g_verticesNo);
		printf("g_trianglesNo: %d\n", g_trianglesNo);

		cudaChannelFormatDesc channel1desc = cudaCreateChannelDesc<uint1>();
		cudaBindTexture(NULL, &g_triIdxListTexture, cudaTriIdxList, &channel1desc, g_triIndexListNo * sizeof(uint1));

		cudaChannelFormatDesc channel2desc = cudaCreateChannelDesc<float2>();
		cudaBindTexture(NULL, &g_pCFBVHlimitsTexture, cudaBVHlimits, &channel2desc, g_pCFBVH_No * 6 * sizeof(float));

		cudaChannelFormatDesc channel3desc = cudaCreateChannelDesc<uint4>();
		cudaBindTexture(NULL, &g_pCFBVHindexesOrTrilistsTexture, cudaBVHindexesOrTrilists, &channel3desc,
			g_pCFBVH_No * sizeof(uint4));

		//cudaChannelFormatDesc channel4desc = cudaCreateChannelDesc<float4>();
		//cudaBindTexture(NULL, &g_verticesTexture, cudaPtrVertices, &channel4desc, g_verticesNo * 8 * sizeof(float));

		cudaChannelFormatDesc channel5desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &g_trianglesTexture, cudaTriangleIntersectionData, &channel5desc, g_trianglesNo * 20 * sizeof(float));
	}
	dim3 block(32, 32, 1);   // dim3 CUDA specific syntax, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 grid(width / block.x, height / block.y, 1);

	/*cudaEvent_t     start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);*/

	CoreLoopPathTracingKernel << <grid, block >> >(dptr, accumulatebuffer, cudaTriangles, cudaRendercam, cudaBVHindexesOrTrilists,
		cudaBVHlimits, cudaTriangleIntersectionData, cudaTriIdxList, framenumber, hashedframes);
	// get stop time, and display the timing results
	/*cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float   elapsedTime;
	cudaEventElapsedTime(&elapsedTime,
		start, stop);
	printf("Time to generate:  %3.1f ms\n", elapsedTime);*/
}
