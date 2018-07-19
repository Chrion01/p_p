/* ------------
 * This code is provided solely for the personal and private use of 
 * students taking the CSC367H1 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * 
 * Authors: Bogdan Simion, Felipe de Azevedo Piovezan
 * 
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2017 Bogdan Simion
 * -------------
*/

#include "kernels.h"

__global__ void kernel4(const int8_t *filter, int32_t dimension, 
        const int32_t *input, int32_t *output, int32_t width, int32_t height, int32_t *g_max, int32_t *g_min)
{
	int start = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ int32_t sdata_min[512];
	__shared__ int32_t sdata_max[512];
	unsigned int tid = threadIdx.x;
	sdata_min[tid] = 999999;
	sdata_max[tid] = -999999;


	int32_t min_v = 999999;
	int32_t max_v = -999999;
	// if (height % devProp.maxThreadsDim[0] > 0) {rows += 1;}
	for(int idx = start; idx < height * width; idx += gridDim.x * blockDim.x){
		int32_t sum = 0;
		// int initial_off = dimension / 2;
		int img_x = idx % width - dimension / 2;
		int img_y = idx / width - dimension / 2;
		
		for(int y = 0; y < dimension; y++){
			for(int x = 0; x < dimension; x ++){
				if(img_x >= 0 && img_x < width && img_y >= 0 && img_y < height){
					int fil_pos = dimension * y + x;
					int img_pos = width * img_y + img_x;
					sum += input[img_pos] * filter[fil_pos];
				}
				img_x++;
			}
			
			img_y++;
			img_x = idx % width - dimension / 2;
		}
		output[idx] = sum;
		if(sum > max_v) {max_v = sum;}
		if(sum < min_v) {min_v = sum;}	
		
	}
	sdata_min[tid] = min_v;
	sdata_max[tid] = max_v;
	__syncthreads();
	for (unsigned int s = blockDim.x/2; s > 32; s >>= 1) { 
	    if (tid < s) {
	    	if(sdata_max[tid] < sdata_max[tid + s]){
	    		sdata_max[tid] = sdata_max[tid + s];
	    	}
	    	if(sdata_min[tid + s] < sdata_min[tid]){
	    		sdata_min[tid] = sdata_min[tid + s];
	    	}
	    }
	    __syncthreads();
	}
	unsigned int blockSize = blockDim.x;
	if (tid < 32) {
		volatile int32_t* smem_max = sdata_max;
		volatile int32_t* smem_min = sdata_min;
		if (blockSize >= 64) {
			if(smem_max[tid] < smem_max[tid + 32]){
	    		smem_max[tid] = smem_max[tid + 32];
	    	}
	    	if(smem_min[tid + 32] < smem_min[tid]){
	    		smem_min[tid] = smem_min[tid + 32];
	    	}
    	}
		if (blockSize >= 32) {
			if(smem_max[tid] < smem_max[tid + 16]){
	    		smem_max[tid] = smem_max[tid + 16];
	    	}
	    	if(smem_min[tid + 16] < smem_min[tid]){
	    		smem_min[tid] = smem_min[tid + 16];
	    	}
    	}
		if (blockSize >= 16) {
			if(smem_max[tid] < smem_max[tid + 8]){
	    		smem_max[tid] = smem_max[tid + 8];
	    	}
	    	if(smem_min[tid + 8] < smem_min[tid]){
	    		smem_min[tid] = smem_min[tid + 8];
	    	}
    	}
		if (blockSize >=  8) {
			if(smem_max[tid] < smem_max[tid + 4]){
	    		smem_max[tid] = smem_max[tid + 4];
	    	}
	    	if(smem_min[tid + 4] < smem_min[tid]){
	    		smem_min[tid] = smem_min[tid + 4];
	    	}
    	}
		if (blockSize >=  4) {
			if(smem_max[tid] < smem_max[tid + 2]){
	    		smem_max[tid] = smem_max[tid + 2];
	    	}
	    	if(smem_min[tid + 2] < smem_min[tid]){
	    		smem_min[tid] = smem_min[tid + 2];
	    	}
    	}
		if (blockSize >=  2) {
			if(smem_max[tid] < smem_max[tid + 1]){
	    		smem_max[tid] = smem_max[tid + 1];
	    	}
	    	if(smem_min[tid + 1] < smem_min[tid]){
	    		smem_min[tid] = smem_min[tid + 1];
	    	}
    	}
	}
	if (tid == 0) { 
		g_max[blockIdx.x] = sdata_max[0]; 
		g_min[blockIdx.x] = sdata_min[0]; 

	}
}


__global__ void normalize4(int32_t *image, int32_t width, int32_t height,
        int32_t smallest, int32_t biggest)
{
	if(smallest != biggest){
		int start = threadIdx.x + blockIdx.x * blockDim.x;
		for(int idx = start; idx < height * width; idx += gridDim.x * blockDim.x){
			if (idx < width * height){
					image[idx] = ((image[idx] - smallest) * 255) / (biggest - smallest);
			}	
		
		}
	}
}
