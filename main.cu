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

#include <stdio.h>
#include <string>
#include <unistd.h>

#include "pgm.h"
#include "kernels.h"
#include "clock.h"
#include "filters.h"

/* Use this function to print the time of each of your kernels.
 * The parameter names are intuitive, but don't hesitate to ask
 * for clarifications.
 * DO NOT modify this function.*/
int deb = 0;
int threads_p_block = 512;
__constant__ int8_t cst_filter[81];
const int dimension = 9;

void print_run(float time_cpu, int kernel, float time_gpu_computation,
        float time_gpu_transfer_in, float time_gpu_transfer_out)
{
    printf("%12.6f ", time_cpu);
    printf("%5d ", kernel);
    printf("%12.6f ", time_gpu_computation);
    printf("%14.6f ", time_gpu_transfer_in);
    printf("%15.6f ", time_gpu_transfer_out);
    printf("%13.2f ", time_cpu/time_gpu_computation);
    printf("%7.2f\n", time_cpu/
            (time_gpu_computation  + time_gpu_transfer_in + time_gpu_transfer_out));
}

void print_matrix(int32_t *input, int32_t width, int32_t height){
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            printf("%d ", input[y * width + x]);
        }
        printf("\n");
    }
}
void check_error(){

	cudaError_t error = cudaGetLastError();
	if(error != 0){
		printf("%s\n",cudaGetErrorString(error));
		
	}
}
__global__ void reduce(int32_t *g_min, int32_t *g_max, int total) {
    __shared__ int32_t sdata_min[512];
    __shared__ int32_t sdata_max[512];
    unsigned int blockSize = blockDim.x;
	unsigned int tid = threadIdx.x;
	sdata_min[tid] = 9999;
	sdata_max[tid] = -9999;
	// Global thread id
	unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
	if(i < total){
		sdata_min[tid] = g_min[i];
		sdata_max[tid] = g_max[i];
	}
	unsigned int i2 = i + blockDim.x;
	if(i2 < total  && g_max[i2] > sdata_max[tid]){
		sdata_max[tid] = g_max[i2];
	}
	if(i2 < total  && g_min[i2] < sdata_min[tid]){
		sdata_min[tid] = g_min[i2];
	}

	// do reduction in shared memory
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

	// write result for this block back to global memory

    if (tid == 0) { 
        g_max[blockIdx.x] = sdata_max[0]; 
        g_min[blockIdx.x] = sdata_min[0]; 

    }
}

void run_kernel(pgm_image source_img, int kernel, pgm_image gpu_output_img, size_t SIZE, size_t SIZE_F, int8_t *filter, int dimension, std::string gpu_file, float time_cpu){
    float transfer_in = 0.0, computation_time = 0.0, transfer_out = 0.0;   

    int32_t *g_max, *g_min;
    int32_t *d_input, *d_output;
    int8_t *d_filter;
    cudaMalloc((void **)&d_input, SIZE);
    cudaMalloc((void **)&d_output, SIZE);
    cudaMalloc((void **)&d_filter, SIZE_F);

    Clock c;
    if(kernel < 5){
		c.start();

		cudaMemcpy(d_input, source_img.matrix, SIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(d_filter, filter, SIZE_F, cudaMemcpyHostToDevice);

		transfer_in += c.stop();

		if(kernel == 1){
		    c.start();

		    int blocks = (source_img.width * source_img.height) / threads_p_block;
		    if ((source_img.width * source_img.height) % threads_p_block > 0){blocks += 1;}
		    
			cudaMalloc((void **)&g_max, blocks * sizeof(int32_t));
		    cudaMalloc((void **)&g_min, blocks * sizeof(int32_t));

		    kernel1<<<blocks, threads_p_block>>>(d_filter, dimension, d_input,
		                    d_output, source_img.width, source_img.height, g_max, g_min);

		    int reduce_blocks = blocks / threads_p_block;
		    if (blocks % threads_p_block > 0) {reduce_blocks++;}
		    reduce<<<reduce_blocks, threads_p_block>>>(g_min, g_max, blocks);
		    
		    int32_t *mins = (int32_t *)malloc(sizeof(int32_t) * reduce_blocks);
		    int32_t *maxs = (int32_t *)malloc(sizeof(int32_t) * reduce_blocks);
		    
    		computation_time += c.stop();
    		
    		c.start();
    		
		    cudaMemcpy(mins, g_min, sizeof(int32_t) * reduce_blocks, cudaMemcpyDeviceToHost);
		    cudaMemcpy(maxs, g_max, sizeof(int32_t) * reduce_blocks, cudaMemcpyDeviceToHost);
		    
		    transfer_out += c.stop();
		    
		    c.start();
		    
			int32_t largest = maxs[0];
		    int32_t smallest = mins[0];
		    for(int i = 1; i < reduce_blocks; i++){
		    	if(maxs[i] > largest){
		    		largest = maxs[i];
		    	}
		    	if(mins[i] < smallest){
		    		smallest = mins[i];
				}
		    }
		    normalize1<<<blocks, threads_p_block>>>(d_output, source_img.width, source_img.height, smallest, largest);
			computation_time += c.stop();
			free(mins);
    		free(maxs);
		}
		else if(kernel == 2){
		    c.start();

		    int blocks = source_img.width * source_img.height / threads_p_block;
		    if (source_img.width * source_img.height % threads_p_block > 0){blocks += 1;}

		    cudaMalloc((void **)&g_max, blocks * sizeof(int32_t));
		    cudaMalloc((void **)&g_min, blocks * sizeof(int32_t));

		    kernel2<<<blocks, threads_p_block>>>(d_filter, dimension, d_input,
		                    d_output, source_img.width, source_img.height, g_max, g_min);
		    
		    int reduce_blocks = blocks / threads_p_block;
		    if (blocks % threads_p_block > 0) {reduce_blocks++;}
		    reduce<<<reduce_blocks, threads_p_block>>>(g_min, g_max, blocks);
		    
		    int32_t *mins = (int32_t *)malloc(sizeof(int32_t) * reduce_blocks);
		    int32_t *maxs = (int32_t *)malloc(sizeof(int32_t) * reduce_blocks);
		     
    		computation_time += c.stop();
    		
    		c.start();
    		
		    cudaMemcpy(mins, g_min, sizeof(int32_t) * reduce_blocks, cudaMemcpyDeviceToHost);
		    cudaMemcpy(maxs, g_max, sizeof(int32_t) * reduce_blocks, cudaMemcpyDeviceToHost);
		    
		    transfer_out += c.stop();
		    
		    c.start();
		    
			int32_t largest = maxs[0];
		    int32_t smallest = mins[0];
		    for(int i = 1; i < reduce_blocks; i++){
		    	if(maxs[i] > largest){
		    		largest = maxs[i];
		    	}
		    	if(mins[i] < smallest){
		    		smallest = mins[i];
				}
		    }
		    normalize2<<<blocks, threads_p_block>>>(d_output, source_img.width, source_img.height, smallest, largest);
			computation_time += c.stop();
			free(mins);
    		free(maxs);
		}
		else if(kernel == 3){
		    c.start();

		    int blocks = source_img.height / threads_p_block;
		    //if(blocks > source_img.height){ blocks = source_img.height; }
		    if (source_img.height % threads_p_block > 0){blocks += 1;}

		    cudaMalloc((void **)&g_max, blocks * sizeof(int32_t));
		    cudaMalloc((void **)&g_min, blocks * sizeof(int32_t));

		    kernel3<<<blocks, threads_p_block>>>(d_filter, dimension, d_input,
		                    d_output, source_img.width, source_img.height, g_max, g_min);
		  
		  	int reduce_blocks = blocks / threads_p_block;
		    if (blocks % threads_p_block > 0) {reduce_blocks++;}
		    reduce<<<reduce_blocks, threads_p_block>>>(g_min, g_max, blocks);
	   
		    int32_t *mins = (int32_t *)malloc(sizeof(int32_t) * reduce_blocks);
		    int32_t *maxs = (int32_t *)malloc(sizeof(int32_t) * reduce_blocks);
		     
    		computation_time += c.stop();
    		
    		c.start();
    		
		    cudaMemcpy(mins, g_min, sizeof(int32_t) * reduce_blocks, cudaMemcpyDeviceToHost);
		    cudaMemcpy(maxs, g_max, sizeof(int32_t) * reduce_blocks, cudaMemcpyDeviceToHost);
		    
		    transfer_out += c.stop();
		    
		    c.start();
		    
			int32_t largest = maxs[0];
		    int32_t smallest = mins[0];
		    for(int i = 1; i < reduce_blocks; i++){
		    	if(maxs[i] > largest){
		    		largest = maxs[i];
		    	}
		    	if(mins[i] < smallest){
		    		smallest = mins[i];
				}
		    }
		    normalize3<<<blocks, threads_p_block>>>(d_output, source_img.width, source_img.height, smallest, largest);
			computation_time += c.stop();
			free(mins);
    		free(maxs);

		}
		else if(kernel == 4){
		    c.start();
		    int blocks = source_img.width / threads_p_block;
		    if (source_img.width % threads_p_block > 0){blocks += 1;}
		    blocks *= (source_img.height + 9) / 10;

		    cudaMalloc((void **)&g_max, blocks * sizeof(int32_t));
		    cudaMalloc((void **)&g_min, blocks * sizeof(int32_t));

		    kernel4<<<blocks, threads_p_block>>>(d_filter, dimension, d_input,
		                    d_output, source_img.width, source_img.height, g_max, g_min);
		                    
		    int reduce_blocks = blocks / threads_p_block;
		    if (blocks % threads_p_block > 0) {reduce_blocks++;}
		    reduce<<<reduce_blocks, threads_p_block>>>(g_min, g_max, blocks);
	   
		    int32_t *mins = (int32_t *)malloc(sizeof(int32_t) * reduce_blocks);
		    int32_t *maxs = (int32_t *)malloc(sizeof(int32_t) * reduce_blocks);
		     
    		computation_time += c.stop();
    		
    		c.start();
    		
		    cudaMemcpy(mins, g_min, sizeof(int32_t) * reduce_blocks, cudaMemcpyDeviceToHost);
		    cudaMemcpy(maxs, g_max, sizeof(int32_t) * reduce_blocks, cudaMemcpyDeviceToHost);
		    
		    transfer_out += c.stop();
		    
		    c.start();
		    
			int32_t largest = maxs[0];
		    int32_t smallest = mins[0];
		    for(int i = 1; i < reduce_blocks; i++){
		    	if(maxs[i] > largest){
		    		largest = maxs[i];
		    	}
		    	if(mins[i] < smallest){
		    		smallest = mins[i];
				}
		    }
		    
		    normalize4<<<blocks, threads_p_block>>>(d_output, source_img.width, source_img.height, smallest, largest);
			computation_time += c.stop();
			free(mins);
    		free(maxs);
		}
    }
    else if(kernel == 5){

		c.start();
		
		cudaMemcpy(d_input, source_img.matrix, SIZE, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(cst_filter, filter, SIZE_F);
		
		transfer_in = c.stop();
		/*
		check_error();
        c.start();
   		int blocks_x = source_img.width / 32;
        if (source_img.width % 32 > 0){blocks_x += 1;}
		int blocks_y = source_img.height / 32;
        if (source_img.height % 32 > 0){blocks_y += 1;}
        int blocks = blocks_x * blocks_y;
        cudaMalloc((void **)&g_max, blocks * sizeof(int32_t));
        cudaMalloc((void **)&g_min, blocks * sizeof(int32_t));
		
		dim3 dimBlock(blocks_x, blocks_y);
		dim3 dimThread(32, 32);

        kernel5<<<dimBlock, dimThread>>>(dimension, d_input,
                        d_output, source_img.width, source_img.height, g_max, g_min);
                        
        int reduce_blocks = blocks / threads_p_block;
        if (blocks % threads_p_block > 0) {reduce_blocks++;}
        reduce<<<reduce_blocks, threads_p_block>>>(g_min, g_max, blocks);
   
        int32_t *mins = (int32_t *)malloc(sizeof(int32_t) * reduce_blocks);
        int32_t *maxs = (int32_t *)malloc(sizeof(int32_t) * reduce_blocks);
         
		computation_time += c.stop();
		
		c.start();
		
	    cudaMemcpy(mins, g_min, sizeof(int32_t) * reduce_blocks, cudaMemcpyDeviceToHost);
	    cudaMemcpy(maxs, g_max, sizeof(int32_t) * reduce_blocks, cudaMemcpyDeviceToHost);
	    
	    transfer_out += c.stop();
	    
	    c.start();
		    
		int32_t largest = maxs[0];
        int32_t smallest = mins[0];
        for(int i = 1; i < reduce_blocks; i++){
        	if(maxs[i] > largest){
        		largest = maxs[i];
        	}
        	if(mins[i] < smallest){
        		smallest = mins[i];
    		}
        }
        
        normalize5<<<dimBlock, dimThread>>>(d_output, source_img.width, source_img.height, smallest, largest);
		computation_time += c.stop();
		free(mins);
    	free(maxs);
    	*/
    	c.start();
		    int blocks = source_img.width / threads_p_block;
		    if (source_img.width % threads_p_block > 0){blocks += 1;}
		    blocks *= (source_img.height + 9) / 10;

		    cudaMalloc((void **)&g_max, blocks * sizeof(int32_t));
		    cudaMalloc((void **)&g_min, blocks * sizeof(int32_t));

		    kernel5<<<blocks, threads_p_block>>>(dimension, d_input,
		                    d_output, source_img.width, source_img.height, g_max, g_min);
		                    
		    int reduce_blocks = blocks / threads_p_block;
		    if (blocks % threads_p_block > 0) {reduce_blocks++;}
		    reduce<<<reduce_blocks, threads_p_block>>>(g_min, g_max, blocks);
	   
		    int32_t *mins = (int32_t *)malloc(sizeof(int32_t) * reduce_blocks);
		    int32_t *maxs = (int32_t *)malloc(sizeof(int32_t) * reduce_blocks);
		     
    		computation_time += c.stop();
    		
    		c.start();
    		
		    cudaMemcpy(mins, g_min, sizeof(int32_t) * reduce_blocks, cudaMemcpyDeviceToHost);
		    cudaMemcpy(maxs, g_max, sizeof(int32_t) * reduce_blocks, cudaMemcpyDeviceToHost);
		    
		    transfer_out += c.stop();
		    
		    c.start();
		    
			int32_t largest = maxs[0];
		    int32_t smallest = mins[0];
		    for(int i = 1; i < reduce_blocks; i++){
		    	if(maxs[i] > largest){
		    		largest = maxs[i];
		    	}
		    	if(mins[i] < smallest){
		    		smallest = mins[i];
				}
		    }
		    
		    normalize5<<<blocks, threads_p_block>>>(d_output, source_img.width, source_img.height, smallest, largest);
			computation_time += c.stop();
			free(mins);
    		free(maxs);
    }

    
	c.start();

    cudaMemcpy(gpu_output_img.matrix, d_output, SIZE, cudaMemcpyDeviceToHost);

    transfer_out += c.stop();

    if(deb == 1){
        /*int32_t *matrix = (int32_t*) malloc(SIZE);
        cudaMemcpy(matrix, d_input, SIZE, cudaMemcpyDeviceToHost);
		printf("input\n");
        print_matrix(matrix, source_img.width, source_img.height);
        free(matrix);*/
		printf("result matrix \n");
		print_matrix(gpu_output_img.matrix, gpu_output_img.width, gpu_output_img.height);
    }
    print_run(time_cpu, kernel, computation_time, transfer_in, transfer_out);
    save_pgm_to_file(gpu_file.c_str(), &gpu_output_img);
    
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFree(g_max);
    cudaFree(g_min);
    
}


int main(int argc, char **argv)
{
    int c;
    std::string input_filename, cpu_output_filename, base_gpu_output_filename;
    if (argc < 3)
    {
        printf("Wrong usage. Expected -i <input_file> -o <output_file>\n");
        return 0;
    }

    while ((c = getopt (argc, argv, "i:o:d")) != -1)
    {
        switch (c)
        {
            case 'i':
                input_filename = std::string(optarg);
                break;
            case 'o':
                cpu_output_filename = std::string(optarg);
                base_gpu_output_filename = std::string(optarg);
                break;
            case 'd':
            	deb = 1;
            	break;
            default:
                return 0;
        }
    }

    pgm_image source_img;
    init_pgm_image(&source_img);

    if (load_pgm_from_file(input_filename.c_str(), &source_img) != NO_ERR)
    {
       printf("Error loading source image.\n");
       return 0;
    }

    /* Do not modify this printf */
    printf("CPU_time(ms) Kernel GPU_time(ms) TransferIn(ms) TransferOut(ms) "
            "Speedup_noTrf Speedup\n");

    // FILTER

    /*int8_t my_filter2[] =
    {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
    };*/
    /*
	int8_t my_filter[] =
    {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 24, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
    };
    */
    int8_t my_filter[] =
    {
        0, 1, 1, 2, 2, 2, 1, 1, 0,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        2, 5, 0, -24, -40, -24, 0, 5, 2,
        2, 5, 3, -12, -24, -12, 3, 5, 2,
        1, 4, 5, 3, 0, 3, 5, 4, 1,
        1, 2, 4, 5, 5, 5, 4, 2, 1,
        0, 1, 1, 2, 2, 2, 1, 1, 0,
    };
	filter lp3_f = {dimension, my_filter};

    pgm_image cpu_output_img;
    copy_pgm_image_size(&source_img, &cpu_output_img); 
    if(deb == 1){
    	printf("input\n");
        print_matrix(source_img.matrix, source_img.width, source_img.height);
    }
    Clock cpu_c;
    cpu_c.start();
    apply_filter2d_threaded(&lp3_f,
        source_img.matrix, cpu_output_img.matrix,
        source_img.width, source_img.height,
        4, SHARDED_ROWS, 0);
	/*apply_filter2d(&lp3_f,
        source_img.matrix, cpu_output_img.matrix,
        source_img.width, source_img.height);*/

    float time_cpu = cpu_c.stop();
    if(deb == 1){
		printf("cpu res\n");
        print_matrix(cpu_output_img.matrix, cpu_output_img.width, cpu_output_img.height);
    }
    /* TODO: run your CPU implementation here and get its time. Don't include
     * file IO in your measurement.*/
    save_pgm_to_file(cpu_output_filename.c_str(), &cpu_output_img);


    /* TODO:
     * run each of your gpu implementations here,
     * get their time,
     * and save the output image to a file.
     * Don't forget to add the number of the kernel
     * as a prefix to the output filename:
     * Print the execution times by calling print_run().
     */

    /* For example: */


    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    // int threads_p_block = devProp.maxThreadsPerBlock;


    std::string gpu_file1 = "1"+base_gpu_output_filename;
    pgm_image gpu_output_img1;
    copy_pgm_image_size(&source_img, &gpu_output_img1);
    size_t SIZE = source_img.width * source_img.height * sizeof(int32_t);
    size_t SIZE_F = dimension * dimension * sizeof(int8_t);
    
    run_kernel(source_img, 1, gpu_output_img1, SIZE, SIZE_F, my_filter, dimension, gpu_file1, time_cpu);
    /* Repeat that for all 5 kernels. Don't hesitate to ask if you don't
     * understand the idea. */

    std::string gpu_file2 = "2"+base_gpu_output_filename;
    pgm_image gpu_output_img2;
    copy_pgm_image_size(&source_img, &gpu_output_img2);

    run_kernel(source_img, 2, gpu_output_img2, SIZE, SIZE_F, my_filter, dimension, gpu_file2, time_cpu);

    
    // NUM3
    std::string gpu_file3 = "3"+base_gpu_output_filename;
    pgm_image gpu_output_img3;
    copy_pgm_image_size(&source_img, &gpu_output_img3);

    run_kernel(source_img, 3, gpu_output_img3, SIZE, SIZE_F, my_filter, dimension, gpu_file3, time_cpu);

    
    std::string gpu_file4 = "4"+base_gpu_output_filename;
    pgm_image gpu_output_img4;
    copy_pgm_image_size(&source_img, &gpu_output_img4);

    run_kernel(source_img, 4, gpu_output_img4, SIZE, SIZE_F, my_filter, dimension, gpu_file4, time_cpu);

	std::string gpu_file5 = "5"+base_gpu_output_filename;
    pgm_image gpu_output_img5;
    copy_pgm_image_size(&source_img, &gpu_output_img5);

    run_kernel(source_img, 5, gpu_output_img5, SIZE, SIZE_F, my_filter, dimension, gpu_file5, time_cpu);
    //std::string gpu_file5 = "5"+base_gpu_output_filename;

}
