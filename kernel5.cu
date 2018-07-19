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
#include <stdio.h>
extern __constant__ int8_t cst_filter[81];

/* This is your own kernel, you should decide which parameters to add
   here*/
/*
__global__ void kernel5(int32_t dimension, 
        const int32_t *input, int32_t *output, int32_t width, int32_t height, int32_t *g_max, int32_t *g_min)
{

	__shared__ int32_t sdata_min[1024];
	__shared__ int32_t sdata_max[1024];
	__shared__ int32_t bordered_input[36][36];

	int tid = threadIdx.x + threadIdx.y * 32;
	//printf("ThreadIDX %d: %d x %d\n",tid, threadIdx.x, threadIdx.y);
	int y = blockIdx.y * 32 + threadIdx.y;
	int x = blockIdx.x * 32 + threadIdx.x;
	int shared_x = threadIdx.x + 2;
	int shared_y = threadIdx.y + 2;
	int input_pos = y * width + x;

	bordered_input[shared_y][shared_x] = 0;
	if(x < width && y < height){
		bordered_input[shared_y][shared_x] = input[input_pos];
	}
	
	if(threadIdx.x == 0 || threadIdx.x == 1){
		bordered_input[shared_y][shared_x - 2] = 0;
		if(blockIdx.x > 0 && y < height){ bordered_input[shared_y][shared_x - 2] = input[input_pos - 2];}
	}

	if(threadIdx.y == 0 || threadIdx.y == 1){
		bordered_input[shared_y - 2][shared_x] = 0;
		if(blockIdx.y > 0 && x < width){ bordered_input[shared_y - 2][shared_x] = input[input_pos - 2 * width];}
	}

	if(threadIdx.x == 0 && threadIdx.y == 0){
		bordered_input[shared_y - 1][shared_x - 1] = 0;
		bordered_input[shared_y - 2][shared_x - 2] = 0;
		if(blockIdx.x > 0 && blockIdx.y > 0){
			bordered_input[shared_y - 1][shared_x - 1] = input[input_pos - 1 - 1 * width];
			bordered_input[shared_y - 1][shared_x - 2] = input[input_pos - 2 - 1 * width];
			bordered_input[shared_y - 2][shared_x - 1] = input[input_pos - 1 - 2 * width];
			bordered_input[shared_y - 2][shared_x - 2] = input[input_pos - 2 - 2 * width];
		}
	}
	if(threadIdx.x == 31 && threadIdx.y == 0){
		bordered_input[shared_y - 1][shared_x + 1] = 0;
		bordered_input[shared_y - 2][shared_x + 2] = 0;
		if(x + 2 < width && blockIdx.y > 0){
			bordered_input[shared_y - 1][shared_x + 2] = input[input_pos + 2 - 1 * width];
			bordered_input[shared_y - 2][shared_x + 2] = input[input_pos + 2 - 2 * width];
		}
		if(x + 1 < width && blockIdx.y > 0){
			bordered_input[shared_y - 1][shared_x + 1] = input[input_pos + 1 - 1 * width];
			bordered_input[shared_y - 2][shared_x + 1] = input[input_pos + 1 - 2 * width];
		}
	}
	
	if(threadIdx.x == 0 && threadIdx.y == 31){
		bordered_input[shared_y + 1][shared_x - 1] = 0;
		bordered_input[shared_y + 2][shared_x - 2] = 0;
		if(blockIdx.x > 0 && y + 2 < height){
			bordered_input[shared_y + 2][shared_x - 1] = input[input_pos - 1 + 2 * width];
			bordered_input[shared_y + 2][shared_x - 2] = input[input_pos - 2 + 2 * width];
		}
		if(blockIdx.x > 0 && y + 1 < height){
			bordered_input[shared_y + 1][shared_x - 1] = input[input_pos - 1 + 1 * width];
			bordered_input[shared_y + 1][shared_x - 2] = input[input_pos - 2 + 1 * width];
		}
	}
	if(threadIdx.x == 31 && threadIdx.y == 31){
		bordered_input[shared_y + 1][shared_x + 1] = 0;
		bordered_input[shared_y + 2][shared_x + 2] = 0;
		if(x + 2 < width && y + 2 < height){
			bordered_input[shared_y + 1][shared_x + 2] = input[input_pos + 2 + 1 * width];
			bordered_input[shared_y + 2][shared_x + 1] = input[input_pos + 1 + 2 * width];
			bordered_input[shared_y + 2][shared_x + 2] = input[input_pos + 2 + 2 * width];
		}
		if(x + 1 < width && y + 1 < height ){
			bordered_input[shared_y + 1][shared_x + 1] = input[input_pos + 1 + 1 * width];
		}
	}

	
	if(threadIdx.x == 30 || threadIdx.x == 31){
		bordered_input[shared_y][shared_x + 2] = 0;
		if(x + 2 < width && y < height){
			bordered_input[shared_y][shared_x + 2] = input[input_pos + 2];
		}
	}
	if(threadIdx.y == 30 || threadIdx.y == 31){
		bordered_input[shared_y + 2][shared_x] = 0;
		if(y + 2 < height && x < width){bordered_input[shared_y + 2 ][shared_x] = input[input_pos + 2 * width];}
	}

	sdata_min[tid] = 999999;
	sdata_max[tid] = -999999;
	__syncthreads();
	*/
	/*
	if(tid == 0 && blockIdx.y == 1 && blockIdx.x == 1){
		printf("Block %d %d\n",blockIdx.x, blockIdx.y);
	    for(int p_y = 0; p_y < 36; p_y++){
		    for(int p_x = 0; p_x < 36; p_x++){
		        printf("%d ", bordered_input[p_y][p_x]);
		    }
		    printf("\n");
		}
    }
    */
	// if (height % devProp.maxThreadsDim[0] > 0) {rows += 1;}
	/*
	if(x < width && y < height){
		int32_t sum = 0;
		// int initial_off = dimension / 2;
		int img_x = shared_x - dimension / 2;
		int img_y = shared_y - dimension / 2;
		
		for(int f_y = 0; f_y < dimension; f_y++){
			for(int f_x = 0; f_x < dimension; f_x ++){

				int fil_pos = dimension * f_y + f_x;
				sum += bordered_input[img_y][img_x] * cst_filter[fil_pos];
				
				img_x++;
			}
			
			img_y++;
			img_x = shared_x - dimension / 2;
		}
		output[input_pos] = sum;
		sdata_min[tid] = sum;
		sdata_max[tid] = sum;
		
	}
    __syncthreads();
	for (unsigned int s = 512; s > 0; s >>= 1) { 
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
	if (tid == 0) { 
		g_max[blockIdx.x + gridDim.x * blockIdx.y] = sdata_max[0]; 
		g_min[blockIdx.x + gridDim.x * blockIdx.y] = sdata_min[0]; 

	}
}

__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
        int32_t smallest, int32_t biggest)
{
	int y = blockIdx.y * 32 + threadIdx.y;
	int x = blockIdx.x * 32 + threadIdx.x;
	int idx = y * width + x;
	if(smallest != biggest && x < width && y < height){
    	image[idx] = ((image[idx] - smallest) * 255) / (biggest - smallest);
	}
}
*/
__global__ void kernel5(int32_t dimension, 
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
					sum += input[img_pos] * cst_filter[fil_pos];
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


__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
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
