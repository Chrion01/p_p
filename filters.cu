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

#include "filters.h"
#include <pthread.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>

typedef struct common_data_t {
	const filter *filter;
	const int32_t *original;
	int32_t *target;
	int32_t *output_image;
	int32_t width;
	int32_t height;
	int32_t threads;
	int32_t work_chunk;
	pthread_barrier_t *barrier;
} common_data;

typedef struct work_t {
	common_data *common;
	int32_t id;
} thread_data;

typedef struct queue_t {
	int end;
	int current;
	int stage;
	int chunk_p_row;
	pthread_mutex_t q_lock;
} queue;

int global_min = 999999;
int global_max = -999999;
pthread_mutex_t mutex;
queue global_queue;

/************** FILTER CONSTANTS*****************/
/* laplacian */
int8_t lp3_m[] =
    {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0,
    };
filter lp3_f = {3, lp3_m};

int8_t lp5_m[] =
    {
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, 24, -1, -1,
        -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,
    };
filter lp5_f = {5, lp5_m};

/* Laplacian of gaussian */
int8_t log_m[] =
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
filter log_f = {9, log_m};

/* Identity */
int8_t identity_m[] = {1};
filter identity_f = {1, identity_m};

filter *builtin_filters[NUM_FILTERS] = {&lp3_f, &lp5_f, &log_f, &identity_f};

/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest, 
        int32_t largest)
{
    if (smallest == largest)
    {
        return;
    }
    
    target[pixel_idx] = ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}
/*************** COMMON WORK ***********************/
/* Process a single pixel and returns the value of processed pixel
 * TODO: you don't have to implement/use this function, but this is a hint
 * on how to reuse your code.
 * */
int32_t apply2d(const filter *f, const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int row, int column)
{
	int f_size = f->dimension;
	int8_t *filter = f->matrix;
	int img_x = column - f_size / 2;
	int img_y = row - f_size / 2;
	int sum = 0;
	for(int y = 0; y < f_size; y++){
		for(int x = 0; x < f_size; x ++){
			if(img_x >= 0 && img_x < width && img_y >= 0 && img_y < height){
				int fil_pos = f_size * y + x;
				int img_pos = width * img_y + img_x;
				sum += original[img_pos] * filter[fil_pos];

			}
			img_x++;
		}
		
		img_y++;
		img_x = column - f_size / 2;
	}

    return sum;
}

/*********SEQUENTIAL IMPLEMENTATIONS ***************/
/* TODO: your sequential implementation goes here.
 */
void apply_filter2d(const filter *f, 
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height)
{
	int max = -999999999;
	int min = 999999999;
	for(int i = 0; i < width * height; i++){
		target[i] = apply2d(f, original, target, width, height, i / width, i % width);
		if(target[i] < min){ min = target[i]; }
		if(target[i] > max){ max = target[i]; }
	}
	for(int i = 0; i < width * height; i++){
		normalize_pixel(target, i, min, max);
	}
}

/****************** ROW/COLUMN SHARDING ************/
/* TODO: you don't have to implement this. It is just a suggestion for the
 * organization of the code.
 */

/* Recall that, once the filter is applied, all threads need to wait for
 * each other to finish before computing the smallest/largets elements
 * in the resulting matrix. To accomplish that, we declare a barrier variable:
 *      pthread_barrier_t barrier;
 * And then initialize it specifying the number of threads that need to call
 * wait() on it:
 *      pthread_barrier_init(&barrier, NULL, num_threads);
 * Once a thread has finished applying the filter, it waits for the other
 * threads by calling:
 *      pthread_barrier_wait(&barrier);
 * This function only returns after *num_threads* threads have called it.
 */
void* sharding_work_row(void *work)
{
    /* Your algorithm is essentially:
     *  1- Apply the filter on the image
     *  2- Wait for all threads to do the same
     *  3- Calculate global smallest/largest elements on the resulting image
     *  4- Scale back the pixels of the image. For the non work queue
     *      implementations, each thread should scale the same pixels
     *      that it worked on step 1.
     */
	thread_data *thread_datas = (thread_data *)work;
	common_data *data = thread_datas->common;
	int max = -99999;
	int min = 99999;
	int rows = (data->height + data->threads - 1)/data->threads;
	int end = (thread_datas->id + 1) * (data->width * rows);
	if (end > data->width  * data->height){
		end = data->width  * data->height;
	}
	for(int i = thread_datas->id * (rows * data->width); i < end; i++){
		data->target[i] = apply2d(data->filter, data->original, data->target, data->width, data->height, i / data->width, i % data->width);
		if(data->target[i] < min){ min = data->target[i]; }
		if(data->target[i] > max){ max = data->target[i]; }
	}
	pthread_mutex_lock(&mutex);
	if(min < global_min){ global_min = min; }
	if(max > global_max){ global_max = max; }
	pthread_mutex_unlock(&mutex);
	pthread_barrier_wait(data->barrier);
	for(int i = thread_datas->id * (rows * data->width); i < end; i++){
		normalize_pixel(data->target, i, global_min, global_max);
	}
	
    return NULL;
}
void* sharding_work_colc(void *work)
{
    /* Your algorithm is essentially:
     *  1- Apply the filter on the image
     *  2- Wait for all threads to do the same
     *  3- Calculate global smallest/largest elements on the resulting image
     *  4- Scale back the pixels of the image. For the non work queue
     *      implementations, each thread should scale the same pixels
     *      that it worked on step 1.
     */
	thread_data *thread_datas = (thread_data *)work;
	common_data *data = thread_datas->common;
	int max = -99999;
	int min = 99999;
	int columns = (data->height + data->threads - 1)/data->threads;
	int end = (thread_datas->id + 1) * columns;
	if (end > data->width){
		end = data->width;
	}
	int pos = 0;
	for(int col = columns * thread_datas->id; col < end; col++){
		for(int i = 0; i < data->height; i++){
			pos = data->width * i + col;
			data->target[pos] = apply2d(data->filter, data->original, data->target, data->width, data->height, i, col);
			if(data->target[pos] < min){ min = data->target[pos]; }
			if(data->target[pos] > max){ max = data->target[pos]; }
		}
	}
	pthread_mutex_lock(&mutex);
	if(min < global_min){ global_min = min; }
	if(max > global_max){ global_max = max; }
	pthread_mutex_unlock(&mutex);
	pthread_barrier_wait(data->barrier);

	for(int col = columns * thread_datas->id; col < end; col++){
		for(int i = 0; i < data->height; i++){
			pos = data->width * i + col;
			normalize_pixel(data->target, pos, global_min, global_max);
		}
	}
	
    return NULL;
}
void* sharding_work_colr(void *work)
{
    /* Your algorithm is essentially:
     *  1- Apply the filter on the image
     *  2- Wait for all threads to do the same
     *  3- Calculate global smallest/largest elements on the resulting image
     *  4- Scale back the pixels of the image. For the non work queue
     *      implementations, each thread should scale the same pixels
     *      that it worked on step 1.
     */
	thread_data *thread_datas = (thread_data *)work;
	common_data *data = thread_datas->common;
	int max = -99999;
	int min = 99999;
	int columns = (data->height + data->threads - 1)/data->threads;
	int end = (thread_datas->id + 1) * columns;
	if (end > data->width){
		end = data->width;
	}
	int pos = 0;
	int start = columns * thread_datas->id;
	for(int i = 0; i < data->height; i++){
		for(int col = start; col < end; col++){
			pos = data->width * i + col;
			data->target[pos] = apply2d(data->filter, data->original, data->target, data->width, data->height, i, col);
			if(data->target[pos] < min){ min = data->target[pos]; }
			if(data->target[pos] > max){ max = data->target[pos]; }
		}
	}
	pthread_mutex_lock(&mutex);
	if(min < global_min){ global_min = min; }
	if(max > global_max){ global_max = max; }
	pthread_mutex_unlock(&mutex);
	pthread_barrier_wait(data->barrier);

	for(int i = 0; i < data->height; i++){
		for(int col = start; col < end; col++){
			pos = data->width * i + col;
			normalize_pixel(data->target, pos, global_min, global_max);
		}
	}
	
	
    return NULL;
}
/***************** WORK QUEUE *******************/
/* TODO: you don't have to implement this. It is just a suggestion for the
 * organization of the code.
 */
void* queue_work(void *work)
{	
	int work_pos = 0;
	int stage = 0;
	thread_data *thread_datas = (thread_data*)work;
	common_data *data = thread_datas->common;
	while(1){
		pthread_mutex_lock(&global_queue.q_lock);
		if(global_queue.current == global_queue.end && global_queue.stage == 2){
			pthread_mutex_unlock(&global_queue.q_lock);
			return NULL;
		}
		
		if(global_queue.current == global_queue.end){ 
			global_queue.current = 0; 
			global_queue.stage++;
		}
		work_pos = global_queue.current;
		stage = global_queue.stage;
		global_queue.current++;
		pthread_mutex_unlock(&global_queue.q_lock);
		
		// int end_hor = (work_pos + 1) * data->work_chunk;
		// int end_vert = (work_pos + 1) * data->work_chunk;
		// if (end_hor > data->width){
		// 	end_hor = data->width;
		// }
		// if (end_vert > data->height){
		// 	end_vert = data->height;
		// }
		int pos = 0;
		int y0 = work_pos / global_queue.chunk_p_row * data->work_chunk;
		int x0 = work_pos % global_queue.chunk_p_row * data->work_chunk;

		if(stage == 0){
			
			int max = -99999;
			int min = 99999;
			
			for(int y = 0; y < data->work_chunk; y++){
				if(y + y0 < data->height){
					for(int x = 0; x < data->work_chunk; x++){
						if(x + x0 < data->height){
							pos = data->width * (y0 + y) + (x0 + x);
							data->target[pos] = apply2d(data->filter, data->original, data->target, data->width, data->height, y0 + y, x0 + x);
							if(data->target[pos] < min){ min = data->target[pos]; }
							if(data->target[pos] > max){ max = data->target[pos]; }
						}
					}
				}
			}
			pthread_mutex_lock(&mutex);
			if(min < global_min){ global_min = min; }
			if(max > global_max){ global_max = max; }
			pthread_mutex_unlock(&mutex);
		}
		else if(stage == 1){
			pthread_barrier_wait(data->barrier);
			pthread_mutex_lock(&global_queue.q_lock);
			if(global_queue.stage != 2){
				global_queue.stage = 2;
				global_queue.current = 0; 
			}
			pthread_mutex_unlock(&global_queue.q_lock);

		}
		else if (stage == 2){
			for(int y = 0; y < data->work_chunk; y++){
				if(y + y0 < data->height){
					for(int x = 0; x < data->work_chunk; x++){
						if(x + x0 < data->height){
							pos = data->width * (y0 + y) + (x0 + x);
							normalize_pixel(data->target, pos, global_min, global_max);
						}
					}
				}
			}
		}
	}
}

/***************** MULTITHREADED ENTRY POINT ******/
/* TODO: this is where you should implement the multithreaded version
 * of the code. Use this function to identify which method is being used
 * and then call some other function that implements it.
 */
void apply_filter2d_threaded(const filter *f,
        const int32_t *original, int32_t *target,
        int32_t width, int32_t height,
        int32_t num_threads, parallel_method method, int32_t work_chunk)
{	
	pthread_barrier_t bar;
	pthread_barrier_init(&bar, NULL, num_threads);
	common_data *data = (common_data*)malloc(sizeof(common_data));
	data->filter = f;
	data->original = original;
	data->target = target;
	data->width = width;
	data->height = height;
	data->threads = num_threads;
	data->work_chunk = work_chunk;
	data->barrier = &bar;
	
	pthread_mutex_init(&mutex, NULL);

	thread_data *thread_info = (thread_data *)malloc(sizeof(thread_data) * num_threads);
	int thread_ids[num_threads];
	pthread_t threads[num_threads];


	if(method == SHARDED_ROWS){
		for(int i = 0; i < num_threads; i++){
			thread_ids[i] = i;
			thread_info[i].id = thread_ids[i];
			thread_info[i].common = data;
			pthread_create(&threads[i], NULL, sharding_work_row, (void *)&thread_info[i]);

		}
		for(int i = 0; i < num_threads; i++){
			pthread_join(threads[i], NULL);
		}

	}
	else if(method == SHARDED_COLUMNS_COLUMN_MAJOR){
		for(int i = 0; i < num_threads; i++){
			thread_ids[i] = i;
			thread_info[i].id = thread_ids[i];
			thread_info[i].common = data;
			pthread_create(&threads[i], NULL, sharding_work_colc, (void *)&thread_info[i]);

		}
		for(int i = 0; i < num_threads; i++){
			pthread_join(threads[i], NULL);
		}
	}

	else if(method == SHARDED_COLUMNS_ROW_MAJOR) {
		for(int i = 0; i < num_threads; i++){
			thread_ids[i] = i;
			thread_info[i].id = thread_ids[i];
			thread_info[i].common = data;
			pthread_create(&threads[i], NULL, sharding_work_colr, (void *)&thread_info[i]);

		}
		for(int i = 0; i < num_threads; i++){
			pthread_join(threads[i], NULL);
		}
	}

	else if(method == WORK_QUEUE){
		int chunks = ((width + work_chunk - 1)/work_chunk) * ((height + work_chunk - 1)/work_chunk);
		pthread_mutex_init(&(global_queue.q_lock), NULL);
		global_queue.current = 0;
		global_queue.end = chunks;
		global_queue.stage = 0;
		global_queue.chunk_p_row = (width + work_chunk - 1)/work_chunk;

		for(int i = 0; i < num_threads; i++){
			thread_ids[i] = i;
			thread_info[i].id = thread_ids[i];
			thread_info[i].common = data;
			pthread_create(&threads[i], NULL, queue_work, (void *)&thread_info[i]);

		}
		for(int i = 0; i < num_threads; i++){
			pthread_join(threads[i], NULL);
		}
	}

    /* You probably want to define a struct to be passed as work for the
     * threads.
     * Some values are used by all threads, while others (like thread id)
     * are exclusive to a given thread. For instance:
     *   typedef struct common_work_t
     *   {
     *       const filter *f;
     *       const int32_t *original_image;
     *       int32_t *output_image;
     *       int32_t width;
     *       int32_t height;
     *       int32_t max_threads;
     *       pthread_barrier_t barrier;
     *   } common_work;
     *   typedef struct work_t
     *   {
     *       common_work *common;
     *       int32_t id;
     *   } work;
     *
     * An uglier (but simpler) solution is to define the shared variables
     * as global variables.
     */
}
