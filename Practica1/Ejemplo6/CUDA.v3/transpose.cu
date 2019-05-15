#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

/* Time */
#include <sys/time.h>
#include <sys/resource.h>

static struct timeval tv0;
double getMicroSeconds()
{
	double t;
	gettimeofday(&tv0, (struct timezone*)0);
	t = ((tv0.tv_usec) + (tv0.tv_sec)*1000000);

	return (t);
}

void init_seed()
{
	int seedi=1;
	FILE *fd;

	/* Generated random values between 0.00 - 1.00 */
	fd = fopen("/dev/urandom", "r");
	fread( &seedi, sizeof(int), 1, fd);
	fclose (fd);
	srand( seedi );
}

void init2Drand(float **buffer, int n)
{
	int i, j;

	for (i=0; i<n; i++)
		for(j=0; j<n; j++)
			buffer[i][j] = 500.0*(float(rand())/RAND_MAX)-500.0; /* [-500 500]*/
}

float *getmemory1D( int nx )
{
	int i;
	float *buffer;

	if( (buffer=(float *)malloc(nx*sizeof(float *)))== NULL )
	{
		fprintf( stderr, "ERROR in memory allocation\n" );
		return( NULL );
	}

	for( i=0; i<nx; i++ )
		buffer[i] = 0.0;

	return( buffer );
}


float **getmemory2D(int nx, int ny)
{
	int i,j;
	float **buffer;

	if( (buffer=(float **)malloc(nx*sizeof(float *)))== NULL )
	{
		fprintf( stderr, "ERROR in memory allocation\n" );
		return( NULL );
	}

	if( (buffer[0]=(float *)malloc(nx*ny*sizeof(float)))==NULL )
	{
		fprintf( stderr, "ERROR in memory allocation\n" );
		free( buffer );
		return( NULL );
	}

	for( i=1; i<nx; i++ )
	{
		buffer[i] = buffer[i-1] + ny;
	}

	for( i=0; i<nx; i++ )
		for( j=0; j<ny; j++ )
		{
			buffer[i][j] = 0.0;
		}

	return( buffer );
}



/********************************************************************************/
/********************************************************************************/

/*
 * Traspose 2D version
 */
void transpose2D(float **in, float **out, int n)
{
	int i, j;

	for(j=0; j < n; j++) 
		for(i=0; i < n; i++) 
			out[j][i] = in[i][j]; 
}

/*
 * Traspose 1D version
 */
void transpose1D(float *in, float *out, int n)
{
	int i, j;

	for(j=0; j < n; j++) 
		for(i=0; i < n; i++) 
			out[j*n+i] = in[i*n+j]; 
}

/*
 * Traspose CUDA version
 */

#define TILE_DIM 16


__global__ void transpose_device_v3(float *in, float *out, int rows, int cols) 
{
	int i,j;
	__shared__ float tile[TILE_DIM][TILE_DIM];

	i = blockIdx.x * blockDim.x + threadIdx.x; 
	j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(i < rows && j<cols){
	
		tile[i][j] = in[j * cols + i]; 
		__syncthreads();

		i = threadIdx.x;
		j = threadIdx.y;	

		out [ i * cols + j ] = tile [i][j]; 

	} 
}

int check(float *GPU, float **CPU, int n)
{
	int i,j;

	for (i=0; i<n; i++)
		for(j = 0; j < n; j++)
			if(GPU[i * n + j]!=CPU[i][j])
				return(1);

	return(0);
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

int main(int argc, char **argv)
{
	int n;
	float **array2D, **array2D_trans;
	float *array2D_trans_GPU;
	double t0;
	float size_block = 16;

	if (argc==2)
		n = atoi(argv[1]);
	else {
		n = 4096;
		printf("./exec n (by default n=%i)\n", n);
	}
	
	/* Initizalization */
	init_seed();
	array2D       = getmemory2D(n,n);
	array2D_trans = getmemory2D(n,n);
	init2Drand(array2D, n);

	/* Transpose 2D version */
	t0 = getMicroSeconds();
	transpose2D(array2D, array2D_trans, n);
	printf("Transpose version 2D: %f MB/s\n", n*n*sizeof(float)/((getMicroSeconds()-t0)/1000000)/1024/1024);


	/* CUDA version */
	float *darray2D, *darray2D_trans;
	cudaMalloc((void**)&darray2D, n*n*sizeof(float));
	cudaMemcpy(darray2D, array2D, n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&darray2D_trans, n*n*sizeof(float));

	dim3 dimBlock(size_block,size_block);
	int blocks = ceil(n/size_block);
	dim3 dimGrid(blocks);

	t0 = getMicroSeconds();
	transpose_device_v3<<<dimGrid,dimBlock>>>(darray2D, darray2D_trans, n, n);	
	array2D_trans_GPU = (float *)malloc(n*n * sizeof(float));
	cudaMemcpy(array2D_trans_GPU, darray2D_trans, n*n*sizeof(float), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	
	printf("Transpose kernel version: %f MB/s\n", n*n*sizeof(float)/((getMicroSeconds()-t0)/1000000)/1024/1024);

	printf("Matriz GPU:\n");	
	print_matrix(array2D_trans_GPU,n,n);

	printf("Matriz CPU\n");
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			printf("%4.1f ",array2D_trans[i][j]);
		}
	printf("\n");
	}

	if (check(array2D_trans_GPU, array2D_trans, n*n))
		printf("Transpose CPU-GPU differs!!\n");


	return(1);
}
