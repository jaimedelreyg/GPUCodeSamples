#include <stdio.h>
#include <stdlib.h>
#include "my_ocl.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.h"

/* From common.c */
extern char *err_code (cl_int err_in);

#define N_STEPS 50

double calc_piOCL(int n)
{
	
	double* parcial_sums;
	cl_mem parcial_sums_d;
	double pi;

	// OpenCL host variables
	cl_device_id device_id;
	cl_int err;
	cl_platform_id platform_id;
	cl_uint num_platforms_returned;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	size_t global[1];
	
	// variables used to read kernel source file
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  // char string to hold kernel source

	// initialize parcial sums array

	parcial_sums = malloc(sizeof(double) * n);	
	
	// read the kernel
	fp = fopen("pi_kernel.cl","r");
	fseek(fp,0L, SEEK_END);
	filelen = ftell(fp);
	rewind(fp);

	kernel_src = malloc(sizeof(char)*(filelen+1));
	readlen = fread(kernel_src,1,filelen,fp);
	if(readlen!= filelen)
	{
		printf("error reading file\n");
		exit(1);
	}
	
	// ensure the string is NULL terminated
	kernel_src[filelen]='\0';

	// Set up platform and GPU device

	cl_uint numPlatforms;

	// Find number of platforms
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Error: Failed to find a platform!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	// Get all platforms
	cl_platform_id Platform[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, Platform, NULL);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Error: Failed to get the platform!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}

	// Secure a GPU
	int i;
	for (i = 0; i < numPlatforms; i++)
	{
		err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
		if (err == CL_SUCCESS)
		{
			break;
		}
	}

	if (device_id == NULL)
	{
		printf("Error: Failed to create a device group!\n%s\n",err_code(err));
		return EXIT_FAILURE;
	}


	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
		return EXIT_FAILURE;
	}

	// Create a command queue
	cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
		return EXIT_FAILURE;
	}

	// create command queue 
	command_queue = clCreateCommandQueue(context,device_id, 0, &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create command queue. Error Code=%d\n",err);
		exit(1);
	}
	 
	// create program object from source. 
	// kernel_src contains source read from file earlier
	program = clCreateProgramWithSource(context, 1 ,(const char **)
                                          &kernel_src, NULL, &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create program object. Error Code=%d\n",err);
		exit(1);
	}       
	
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
        	printf("Build failed. Error Code=%d\n", err);

		size_t len;
		char buffer[2048];
		// get the build log
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                  sizeof(buffer), buffer, &len);
		printf("--- Build Log -- \n %s\n",buffer);
		exit(1);
	}

	kernel = clCreateKernel(program, "pi", &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create kernel object. Error Code=%d\n",err);
		exit(1);
	}

        // Create the output arrays in device memory
        parcial_sums_d  = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(double)* n, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        } 


	// set the kernel arguments
	if ( clSetKernelArg(kernel, 0, sizeof(cl_mem), &parcial_sums_d) ||
             clSetKernelArg(kernel, 1, sizeof(cl_int), &n))
	{
		printf("Unable to set kernel arguments. Error Code=%d\n",err);
		exit(1);
	}

	// set the global work dimension size
	global[0]= n;
	
        // Enqueue the kernel object with 
	// Dimension size = 1, 
	// global worksize = global, 
	// local worksize = NULL - let OpenCL runtime determine
	// No event wait list
	err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
                                   global, NULL, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{	
		printf("Unable to enqueue kernel command. Error Code=%d\n",err);
		exit(1);
	}

	// wait for the command to finish
	clFinish(command_queue);

        //read output vectors into compute device memory 
        err = clEnqueueReadBuffer(commands, parcial_sums_d, CL_TRUE, 0, sizeof(double) * n, parcial_sums, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
           printf("Error: Failed to read d_o to source array!\n%s\n", err_code(err));
            exit(1);
        }
   	
	for (int z = 0; i < n; i++) pi += parcial_sums[i];
	pi /= n; 

	// clean up
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(kernel_src);
	free(parcial_sums);
	return pi;

}
