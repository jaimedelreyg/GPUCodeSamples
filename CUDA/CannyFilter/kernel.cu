#include <cuda.h>
#include <math.h>

#include "kernel.h"

#define TILEW 4
#define KERNEL_RADIUS 2

__global__ void calculateEdges(float *d_G, float *d_phi, float *d_out, float w, float h){

	__shared__ float s_mem_G[TILEW + 2][TILEW + 2];
	float s_mem_pedge[TILEW][TILEW];
	int d_x = blockIdx.x * blockDim.x + threadIdx.x;
	int d_y = blockIdx.y * blockDim.y + threadIdx.y;
	float level = 5.0;
	float lowthres, hithres;

	//Check if the thread is in the image range
	if((d_x < w) && (d_y < h)){

		//Load the shared memory
		for(int y = threadIdx.y; y <= (threadIdx.y + 2); y++){
			for(int x = threadIdx.x; x <= (threadIdx.x + 2); x++){

				if((x/1 == 0) || (y/1 == 0) || ((blockIdx.x * blockDim.x + x) > w) || ((blockIdx.y * blockDim.y + y) > h)) s_mem_G[y][x] = 0;
				else s_mem_G[y][x] = d_G[(int)((blockIdx.y * blockDim.y + y) * w + (blockIdx.x * blockDim.x + x))];

			}

		}

		//Calculate initial edges
		s_mem_pedge[threadIdx.y][threadIdx.x] = 0;
		if(d_phi[(int)(d_y * w + d_x)] == 0){
			if(s_mem_G[threadIdx.y+1][threadIdx.x+1]>s_mem_G[threadIdx.y+1][threadIdx.x+2] && s_mem_G[threadIdx.y+1][threadIdx.x+1]>s_mem_G[threadIdx.y+1][threadIdx.x]) //edge is in N-S
				s_mem_pedge[threadIdx.y][threadIdx.x] = 1;

		} else if(d_phi[(int)(d_y * w + d_x)] == 45) {
			if(s_mem_G[threadIdx.y+1][threadIdx.x+1]>s_mem_G[threadIdx.y+2][threadIdx.x+2] && s_mem_G[threadIdx.y+1][threadIdx.x+1]>s_mem_G[threadIdx.y][threadIdx.x]) // edge is in NW-SE
				s_mem_pedge[threadIdx.y][threadIdx.x] = 1;

		} else if(d_phi[(int)(d_y * w + d_x)] == 90) {
			if(s_mem_G[threadIdx.y+1][threadIdx.x+1]>s_mem_G[threadIdx.y+2][threadIdx.x+1] && s_mem_G[threadIdx.y+1][threadIdx.x+1]>s_mem_G[threadIdx.y][threadIdx.x+1]) //edge is in E-W
				s_mem_pedge[threadIdx.y][threadIdx.x] = 1;

		} else if(d_phi[(int)(d_y * w + d_x)] == 135) {
			if(s_mem_G[threadIdx.y+1][threadIdx.x+1]>s_mem_G[threadIdx.y+2][threadIdx.x+1] && s_mem_G[threadIdx.y+1][threadIdx.x+1]>s_mem_G[threadIdx.y+1][threadIdx.x+2]) // edge is in NE-SW
				s_mem_pedge[threadIdx.y][threadIdx.x] = 1;
		}

		// Hysteresis Thresholding
		lowthres = level/2;
		hithres  = 2*(level);

		if(s_mem_G[threadIdx.y+1][threadIdx.x+1]>hithres && s_mem_pedge[threadIdx.y][threadIdx.x])
			d_out[(int)(d_y * w + d_x)] = 255;
		else if(s_mem_pedge[threadIdx.y][threadIdx.x] && s_mem_G[threadIdx.y+1][threadIdx.x+1]>=lowthres && s_mem_G[threadIdx.y+1][threadIdx.x+1]<hithres){

			for(int y = threadIdx.y; y <= (threadIdx.y + 2); y++){
				for(int x = threadIdx.x; x <= (threadIdx.x + 2); x++){
					if (s_mem_G[y][x]>hithres)
						d_out[(int)(d_y * w + d_x)] = 255;
				}
			}
		}

	}
}

__global__ void calculateGandPhi(float *d_G_x, float *d_G_y, float *d_G, float *d_phi, float w,float h){

	__shared__ float s_mem_g_x[TILEW*TILEW];
	__shared__ float s_mem_g_y[TILEW*TILEW];

	int d_x = blockIdx.x * blockDim.x + threadIdx.x;
	int d_y = blockIdx.y * blockDim.y + threadIdx.y;
	float PI = 3.141593;

	//Check if the thread is in the image range
	if((d_x < w) && (d_y < h)){
		//Initialize shared memory
		s_mem_g_x[threadIdx.y * TILEW + threadIdx.x] = d_G_x[(int)(d_y * w + d_x)];
		s_mem_g_y[threadIdx.y * TILEW + threadIdx.x] = d_G_y[(int)(d_y * w + d_x)];


		d_G[(int)(d_y * w + d_x)] = sqrtf(powf(s_mem_g_x[threadIdx.y * TILEW + threadIdx.x],2)+ powf(s_mem_g_y[threadIdx.y * TILEW + threadIdx.x],2));
		d_phi[(int)(d_y * w + d_x)] = atan2f(fabs(s_mem_g_x[threadIdx.y * TILEW + threadIdx.x]),fabs(s_mem_g_x[threadIdx.y * TILEW + threadIdx.x]));

		if(fabs(d_phi[(int)(d_y * w + d_x)])<=PI/8 )
			d_phi[(int)(d_y * w + d_x)] = 0;
		else if (fabs(d_phi[(int)(d_y * w + d_x)])<= 3*(PI/8))
			d_phi[(int)(d_y * w + d_x)] = 45;
		else if (fabs(d_phi[(int)(d_y * w + d_x)]) <= 5*(PI/8))
			d_phi[(int)(d_y * w + d_x)] = 90;
		else if (fabs(d_phi[(int)(d_y * w + d_x)]) <= 7*(PI/8))
			d_phi[(int)(d_y * w + d_x)] = 135;
		else d_phi[(int)(d_y * w + d_x)] = 0;
	}

}

__global__ void convolutionGPU(float *kernel, float *d_out,
			float *d_in, float w, float h){

	__shared__ float s_mem[TILEW + KERNEL_RADIUS * 2][TILEW + KERNEL_RADIUS * 2];
	int d_x = blockIdx.x * blockDim.x + threadIdx.x;
	int d_y = blockIdx.y * blockDim.y + threadIdx.y;
	float sum = 0;	

	//Check if the thread is in the image range
	if((d_x < w) && (d_y < h)){

		//Load the shared memory
		for(int y = threadIdx.y; y <= (threadIdx.y + (KERNEL_RADIUS * 2)); y++){
			for(int x = threadIdx.x; x <= (threadIdx.x + (KERNEL_RADIUS * 2)); x++){

				if((x/KERNEL_RADIUS == 0) || (y/KERNEL_RADIUS == 0) || ((blockIdx.x * blockDim.x + x) > w) || ((blockIdx.y * blockDim.y + y) > h)) s_mem[y][x] = 0;
				else s_mem[y][x] = d_in[(int)((blockIdx.y * blockDim.y + y) * w + (blockIdx.x * blockDim.x + x))];

			}

		}

		__syncthreads();

		int ky = 0;
		int kx = 0;

		for(int y = threadIdx.y; y <= (threadIdx.y + (KERNEL_RADIUS * 2)); y++){
			for(int x = threadIdx.x; x <= (threadIdx.x + (KERNEL_RADIUS * 2)); x++){
				sum += s_mem[y][x] * kernel[ky * (KERNEL_RADIUS * 2 + 1) + kx];
				kx++;
			}
		ky++;
		kx = 0;
		}

		d_out[(int)(d_y * w + d_x)] = sum/159.0;

	}

}

void cannyGPU(float *im, float *image_out,
	float level,
	int height, int width)
{
	float gaussian_kernel[25] = {2,4,5,4,2,4,9,12,9,4,5,12,15,12,5,4,9,12,9,4,2,4,5,4,2};
	float g_x_kernel[25] = {1,2,0,-2,-1,4,8,0,-8,-4,6,12,0,-12,-6,4,8,0,-8,-4,1,2,0,-2,-1};
	float g_y_kernel[25] = {-1,-4,-6,-4,-1,-2,-8,-12,-8,-2,0,0,0,0,0,2,8,12,8,2,1,4,6,4,1};

	/* CUDA vesion */
	float *d_im_in, *d_im_out, *d_gaussian_kernel, *d_G_x, *d_G_y, *d_g_y_kernel,
	*d_g_x_kernel, *d_G, *d_phi, *d_edges_final;

	cudaMalloc((void**)&d_im_in, height*width*sizeof(float));
	cudaMemcpy(d_im_in, im, height*width*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_gaussian_kernel, 5*5*sizeof(float));
	cudaMemcpy(d_gaussian_kernel, gaussian_kernel, 5*5*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_im_out, height*width*sizeof(float));
	cudaMalloc((void**)&d_G_x, height*width*sizeof(float));
	cudaMalloc((void**)&d_G_y, height*width*sizeof(float));
	cudaMalloc((void**)&d_g_y_kernel, 5*5*sizeof(float));
	cudaMemcpy(d_g_y_kernel, g_y_kernel, 5*5*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_g_x_kernel, 5*5*sizeof(float));
	cudaMemcpy(d_g_x_kernel, g_x_kernel, 5*5*sizeof(float), cudaMemcpyHostToDevice);


	dim3 dimBlock(TILEW,TILEW);
	int blocks_x = ceil(width/TILEW);
	int blocks_y = ceil(height/TILEW);
	dim3 dimGrid(blocks_x,blocks_y);

	//Gaussian filter and other filters
	convolutionGPU<<<dimGrid,dimBlock>>>(d_gaussian_kernel,d_im_out,d_im_in,width,height);
	cudaMemcpy(image_out, d_im_out,height*width*sizeof(float), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
//	convolutionGPU<<<dimGrid,dimBlock>>>(d_g_x_kernel,d_G_x,d_im_out,width,height);
//	//cudaMemcpy(image_out, d_im_out,height*width*sizeof(float), cudaMemcpyDeviceToHost);
//	cudaThreadSynchronize();
//	convolutionGPU<<<dimGrid,dimBlock>>>(d_g_y_kernel,d_G_y,d_im_out,width,height);
//	//cudaMemcpy(image_out, d_im_out,height*width*sizeof(float), cudaMemcpyDeviceToHost);
//	cudaThreadSynchronize();
//
//	//Calculate G with Gx and Gy
//	cudaMalloc((void**)&d_G, height*width*sizeof(float));
//	cudaMalloc((void**)&d_phi, height*width*sizeof(float));
//
//	calculateGandPhi<<<dimGrid,dimBlock>>>(d_G_x,d_G_y,d_G,d_phi,width,height);
//	//cudaMemcpy(image_out, d_G,height*width*sizeof(float), cudaMemcpyDeviceToHost);
//	cudaThreadSynchronize();
//
//	cudaMalloc((void**)&d_edges_final, height*width*sizeof(float));

//	calculateEdges<<<dimGrid,dimBlock>>>(d_G, d_phi, d_edges_final, width, height);
//	cudaMemcpy(image_out, d_edges_final,height*width*sizeof(float), cudaMemcpyDeviceToHost);
//	cudaThreadSynchronize();


	cudaFree(d_im_in);
	cudaFree(d_im_out);
	cudaFree(d_gaussian_kernel);
	cudaFree(d_G_y);
	cudaFree(d_G_x);
	cudaFree(d_g_y_kernel);
	cudaFree(d_g_x_kernel);
	cudaFree(d_phi);
	cudaFree(d_G);
	cudaFree(d_edges_final);


}
