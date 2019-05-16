#include <stdio.h>
#include <stdlib.h>
#include "my_ocl.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.h"

/* From common.c */
extern char *err_code (cl_int err_in);

//From aes.c
unsigned char keys[] = {0x01, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
			0x02, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
			0x03, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03,
			0x04, 0x17, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04,
			0x05, 0x16, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05,
			0x06, 0x15, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06,
			0x07, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07,
			0x08, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08,
			0x09, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09,
			0x10, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10};

int lookup_table[] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};


#define N_KEYS 10

int aes_encriptionOCL(double *im, double *im_out, int height, int width) 
{
	cl_mem img_d;
	cl_mem keys_d;
	cl_mem lookup_table_d;
	
	// OpenCL host variables
	cl_device_id device_id;
	cl_int err;
	cl_platform_id platform_id;
	cl_uint num_platforms_returned;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	size_t global[2];
	size_t local[1];
	
	// variables used to read kernel source file
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  // char string to hold kernel source

	
	// read the kernel
	fp = fopen("aes_kernel.cl","r");
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

	kernel = clCreateKernel(program, "aes_encription", &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create kernel object. Error Code=%d\n",err);
		exit(1);
	}


        // Create the output arrays in device memory
        img_d = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(double)*height*width, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }
        
	keys_d = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(unsigned char)*16*N_KEYS, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        } 

        lookup_table_d = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(int)*16, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        } 

	err = clEnqueueWriteBuffer(commands, img_d, CL_TRUE, 0, sizeof(double) * width * height, im, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
           printf("Error: Failed to write im to source array!\n%s\n", err_code(err));
            exit(1);
        }


	err = clEnqueueWriteBuffer(commands, keys_d, CL_TRUE, 0, sizeof(unsigned char) * 16 * N_KEYS, keys, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
           printf("Error: Failed to write keys to source array!\n%s\n", err_code(err));
            exit(1);
        }

	err = clEnqueueWriteBuffer(commands, lookup_table_d, CL_TRUE, 0, sizeof(int) * 16, lookup_table, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
           printf("Error: Failed to write lookup table to source array!\n%s\n", err_code(err));
            exit(1);
        }
	
	// set the kernel arguments
	if ( clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_d) ||
             clSetKernelArg(kernel, 1, sizeof(cl_mem), &keys_d) ||
             clSetKernelArg(kernel, 2, sizeof(cl_mem), &lookup_table_d) ||
             clSetKernelArg(kernel, 3, sizeof(cl_int), &width))
	{
		printf("Unable to set kernel arguments. Error Code=%d\n",err);
		exit(1);
	}

	// set the global work dimension size
	global[0]= width/4;
	global[1] = height/4;

        // Enqueue the kernel object with 
	// Dimension size = 1, 
	// global worksize = global, 
	// local worksize = NULL - let OpenCL runtime determine
	// No event wait list
	err = clEnqueueNDRangeKernel(command_queue, kernel, 2 , NULL, 
                                   global, NULL, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{	
		printf("Unable to enqueue kernel command. Error Code=%d\n",err);
		exit(1);
	}

	// wait for the command to finish
	clFinish(command_queue);

        //read output vectors into compute device memory 
        err = clEnqueueReadBuffer(commands, img_d, CL_TRUE, 0, sizeof(double) * width * height, im_out , 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
           printf("Error: Failed to read d_o to source array!\n%s\n", err_code(err));
            exit(1);
        }

   	
	// clean up
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(kernel_src);
	return 0;

}

int aes_decriptionOCL(double *im, double *im_out, int height, int width) 
{
	cl_mem img_d;
	cl_mem keys_d;
	cl_mem lookup_table_d;
	
	// OpenCL host variables
	cl_device_id device_id;
	cl_int err;
	cl_platform_id platform_id;
	cl_uint num_platforms_returned;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	size_t global[2];
	
	// variables used to read kernel source file
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  // char string to hold kernel source

	
	// read the kernel
	fp = fopen("aes_kernel.cl","r");
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

	kernel = clCreateKernel(program, "aes_decription", &err);
	if (err != CL_SUCCESS)
	{	
		printf("Unable to create kernel object. Error Code=%d\n",err);
		exit(1);
	}


        // Create the output arrays in device memory
        img_d = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(double)*height*width, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        }
        
	keys_d = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(unsigned char)*16*N_KEYS, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        } 

        lookup_table_d = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(int)*16, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to allocate device memory!\n");
            exit(1);
        } 

	err = clEnqueueWriteBuffer(commands, img_d, CL_TRUE, 0, sizeof(double) * width * height, im, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
           printf("Error: Failed to write im to source array!\n%s\n", err_code(err));
            exit(1);
        }


	err = clEnqueueWriteBuffer(commands, keys_d, CL_TRUE, 0, sizeof(unsigned char) * 16 * N_KEYS, keys, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
           printf("Error: Failed to write keys to source array!\n%s\n", err_code(err));
            exit(1);
        }

	err = clEnqueueWriteBuffer(commands, lookup_table_d, CL_TRUE, 0, sizeof(int) * 16, lookup_table, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
           printf("Error: Failed to write lookup table to source array!\n%s\n", err_code(err));
            exit(1);
        }
	
	// set the kernel arguments
	if ( clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_d) ||
             clSetKernelArg(kernel, 1, sizeof(cl_mem), &keys_d) ||
             clSetKernelArg(kernel, 2, sizeof(cl_mem), &lookup_table_d) ||
             clSetKernelArg(kernel, 3, sizeof(cl_int), &width))
	{
		printf("Unable to set kernel arguments. Error Code=%d\n",err);
		exit(1);
	}

	// set the global work dimension size
	global[0]= width/4;
	global[1] = height/4;

        // Enqueue the kernel object with 
	// Dimension size = 1, 
	// global worksize = global, 
	// local worksize = NULL - let OpenCL runtime determine
	// No event wait list
	err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
                                   global, NULL, 0, NULL, NULL);

	if (err != CL_SUCCESS)
	{	
		printf("Unable to enqueue kernel command. Error Code=%d\n",err);
		exit(1);
	}

	// wait for the command to finish
	clFinish(command_queue);

        //read output vectors into compute device memory 
        err = clEnqueueReadBuffer(commands, img_d, CL_TRUE, 0, sizeof(double) * width * height, im_out , 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
           printf("Error: Failed to read d_o to source array!\n%s\n", err_code(err));
            exit(1);
        }
   	
	// clean up
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(kernel_src);
	return 0;

}
