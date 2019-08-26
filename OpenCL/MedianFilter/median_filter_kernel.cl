//OPENCL KERNEL FOR MEDIAN FILTER CALCULATION
#define WINDOW_SIZE 5

__kernel
void median_filter(__global float* image, __global float* out,
		uint width, float threshold, __local float *local_patch, __local float *window)
{
//Variables
int i = 0;
int j = 0;
float tmp;
float median;

int xlid = get_local_id(0);
int ylid = get_local_id(1);
int xgid = get_global_id(0);
int ygid = get_global_id(1);

//FILL LOCAL PATCH
//Just first thread per group
if((xlid == 0) && (ylid == 0)){
	for(; i< (WINDOW_SIZE + (WINDOW_SIZE/2)); i++)
		for(;j < (WINDOW_SIZE + (WINDOW_SIZE/2)); j++)
			local_patch[i*(WINDOW_SIZE + (WINDOW_SIZE/2))+j] = 0;
}

//All the threads
local_patch[(xlid+1) * (WINDOW_SIZE + (WINDOW_SIZE/2)) + (ylid+1)] = image[xgid * width + ygid];

//WINDOW EXTRACTION IN PRIVATE MEMORY

for(i = xlid; i < WINDOW_SIZE; i++)
	for(j = ylid; j < WINDOW_SIZE; j++)
		window[i*WINDOW_SIZE+j] = local_patch[i*WINDOW_SIZE+j];		



//Bubble sort implementation
for (i=1; i<(WINDOW_SIZE * WINDOW_SIZE); i++)
	for (j=0 ; j<(WINDOW_SIZE * WINDOW_SIZE)- i; j++)
		if (window[j] > window[j+1]){
			tmp = window[j];
		window[j] = window[j+1];
		window[j+1] = tmp;
}

//Median calculation
median = window[(WINDOW_SIZE*WINDOW_SIZE - 1)>>1];

if (fabsf(median-local_patch[(xlid+1) * (WINDOW_SIZE + (WINDOW_SIZE/2)) + (ylid+1)]/median) <=threshold)
		out[xgid*width+ygid] = local_patch[(xlid+1) * (WINDOW_SIZE + (WINDOW_SIZE/2)) + (ylid+1)];
else
		out[xgid*width+ygid] = median;


}

