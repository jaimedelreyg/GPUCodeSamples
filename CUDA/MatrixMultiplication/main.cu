#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>


double wtime(void)
{
        static struct timeval   tv0;
        double time_;

        gettimeofday(&tv0,(struct timezone*)0);
        time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
        return( time_/1000000);
}

void matrixMulCPU(float *A, float *B, int hA, int wA, int wB, float *C)
{
	int i,j,k;

	for (i=0; i<hA; i++)
		for (j=0; j<wB; j++){
			C[i*wB+j] = 0.0;
			for (k=0; k<wA; k++)
				C[i*wB+j] += A[i*wA+k]*B[k*wB+j];
		}
}


__global__ void matrixMultGPU(float* a, float* b, float* c, int nColsA, 
				 int nRowsC, int nColsC){

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < nRowsC && col < nColsC){
	
		for(int i = 0; i < nColsA; ++i){
			//In b[], nColsA = nRowsB
			c[row * nColsC + col] += a[row * nColsA + i] * b[i * nColsA + col]; 
		}	
	
	}
	

}

void init_matrix(float *M, int hM, int wM, float k)
{
	int i,j;

	for (i=0; i<hM; i++)
		for (j=0; j<wM; j++)
			if (i==j)
				M[i*wM+j] = k*1.0f;
			else
				M[i*wM+j] = -1.0f/(float)(wM);
}

void print_matrix(float *M, int hM, int wM)
{
	int i,j;

	for (i=0; i<hM; i++){
//		printf("Line %i: ", i);
		for (j=0; j<wM; j++)
			printf("%4.1f ", M[i*wM+j]);
		printf("\n");
	}
}

int diff(float *A, float *B, int hA, int wA, int wB, float *C_gpu)
{
	float *C_cpu;
	int size_C = wB * hA;
	C_cpu = (float*)malloc(size_C*sizeof(float));

	int i,j,k;

	for (i=0; i<hA; i++)
		for (j=0; j<wB; j++){
			C_cpu[i*wB+j] = 0.0;
			for (k=0; k<wA; k++){
				C_cpu[i*wB+j] += A[i*wA+k]*B[k*wB+j];
			}
		}
	
	//printf("\n\nMATRIX C_cpu\n");print_matrix(C_cpu, hA, wB);

	for (i=0; i<hA; i++)
		for (j=0; j<wB; j++)
			if (fabsf(C_cpu[i*wB+j]-C_gpu[i*wB+j])>1e-5)
			{
				printf("[%i,%i]: %f!=%f\n", i, j, C_cpu[i*wB+j], C_gpu[i*wB+j]);
				return(0);
			}


	return(1);

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	// Matrix variables
	float *A, *B, *C;
	float *a_GPU, *b_GPU, *c_GPU, *gpu_c_host;
	int hA, wA, hB, wB;
	double t0, t1; 
	int i;
	
	setbuf(stdout, NULL);

	if (argc!=4){
		printf("./exec hA hB/WA wB\n");
		exit(-1);
	}

	hA = atoi(argv[1]);
	hB = wA = atoi(argv[2]);
	wB = atoi(argv[3]);

	//init matrix sizes
	int size_A = wA * hA;
	int size_B = wB * hB;
	int size_C = wB * hA;
	
	// CPU Init A and B, malloc C
	A = (float*)malloc(size_A*sizeof(float));
	init_matrix(A, hA, wA, 1.0);

	B = (float*)malloc(size_B*sizeof(float));
	init_matrix(B, hB, wB, 2.0);

	C = (float*)malloc(size_B*sizeof(float));
	for (i = 0; i < (hA*wB); i++) {
		C[i] = 0.0;
	}

	//GPU malloc A,B and C

	cudaMalloc((void **)&a_GPU, size_A * sizeof(float));
	cudaMalloc((void **)&b_GPU, size_B * sizeof(float));
	cudaMalloc((void **)&c_GPU, size_C * sizeof(float));

	//CPU -> GPU
	cudaMemcpy(a_GPU,A,size_A * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(b_GPU,B,size_B * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(c_GPU,C,size_C * sizeof(float),cudaMemcpyHostToDevice);

	
	//GPU Grid and Block ini
	
	//Define BlockSize	
	dim3 dimBlock(16,16);
	
	//Define grid size
	//What happend if C's matrix size it's a prime number?
	int grid_x_size = ceil((float)wB / 16);
	int grid_y_size = ceil((float)hA / 16);
	printf("grid_x_size = %i, grid_y_size = %i\n",grid_x_size, grid_y_size);
	dim3 dimGrid(grid_x_size, grid_y_size);

	
	//Start GPU timer
	t0 = wtime();

	//Invoque kernel
	matrixMultGPU<<<dimGrid,dimBlock>>>(a_GPU, b_GPU, c_GPU, wA, hA, wB);
	
	//Wait for the kernel
	cudaThreadSynchronize();
	
	//Stop timer and print value	
	t1 = wtime(); printf("Time GPU=%f\n", t1-t0);
	
	//GPU result to host
	gpu_c_host  = (float *)malloc(size_C * sizeof(float));
	cudaMemcpy(gpu_c_host,c_GPU,size_C * sizeof(float),cudaMemcpyDeviceToHost);
	

	if (!diff(A, B, hA, wA, wB, gpu_c_host)){
		printf("ERROR=GPU.vs.CPU matrix mult differs\n");
	}
	else{
		printf("GPU.vs.CPU matrix is equal! :D\n");
	}

	// print Matrix
	//printf("\n\nMATRIX A\n");print_matrix(A, hA, wA);
	//printf("\n\nMATRIX B\n");print_matrix(B, hB, wB);
	//printf("\n\nGPU C\n");print_matrix(gpu_c_host, hA, wB);

	/* Free CPU */
	free(A);
	free(B);
	free(gpu_c_host);

	/* Free GPU */
	cudaFree(a_GPU);
	cudaFree(b_GPU);
	cudaFree(c_GPU);
	
	return (1);
}

