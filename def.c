/*****************************************************
 * CG Solver (HPC Software Lab)
 *
 * Parallel Programming Models for Applications in the 
 * Area of High-Performance Computation
 *====================================================
 * IT Center (ITC)
 * RWTH Aachen University, Germany
 * Author: Tim Cramer (cramer@itc.rwth-aachen.de)
 * Date: 2010 - 2015
 *****************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _OPENACC
# include <openacc.h>
#endif

#ifdef CUDA
# include <cuda.h>
#endif

#if defined(_WIN32) || defined(_WIN64)
# include <windows.h>
#else
# include <sys/time.h>
#endif

#include "def.h"

/* Initialize the config with the default values.
 * During the runtime you can change these default
 * by setting CG_MAX_ITER or CG_TOLERANCE, e.g.,
 * $ CG_TOLERANCE=1e-12 CG_MAX_ITER=5000 ./cg_ser a.mtx */
struct config config = {
	.maxIter = 1000,
	.tolerance = 0.0000001
};

/* This init function overwrites the default values,
 * if the corresponding environment variable is set. 
 * Furthermore, a GPU warmup is done.*/
void init(){
	const char *tmp;

	if ((tmp = getenv("CG_MAX_ITER")) != NULL)
		config.maxIter = atoi(tmp);

	if ((tmp = getenv("CG_TOLERANCE")) != NULL)
		config.tolerance = strtod(tmp, NULL);
	
	gpuWarmup();
}

/* Use is time function to get the real time */
double getWTime() {
#if defined(_WIN32) || defined(_WIN64)
# define Li2Double(x) ((double)((x).HighPart) * 4.294967296E9 + \
                      (double)((x).LowPart))
	LARGE_INTEGER time, freq;
	double dtime, dfreq, res;

	if (QueryPerformanceCounter(&time) == 0) {
		DWORD err = GetLastError();
		LPVOID buf;
		FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
		    FORMAT_MESSAGE_FROM_SYSTEM, NULL, err,
		    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buf,
		    0, NULL);
		printf("QueryPerformanceCounter() failed with error %d: %s\n",
		    err, buf);
		exit(1);
	}

	if (QueryPerformanceFrequency(&freq) == 0)
	{
		DWORD err = GetLastError();
		LPVOID buf;
		FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
		    FORMAT_MESSAGE_FROM_SYSTEM, NULL, err,
		    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buf,
		    0, NULL);
		printf("QueryPerformanceFrequency() failed with error %d: %s\n",
		    err, buf);
		exit(1);
	}

	dtime = Li2Double(time);
	dfreq = Li2Double(freq);
	res = dtime / dfreq;

	return res;
#else
	struct timeval tv;
	gettimeofday(&tv, (struct timezone*)0);
	return ((double)tv.tv_sec + (double)tv.tv_usec / 1000000.0 );
#endif
}

/* Warmup function for the GPU. You do not need to modify this. */
void gpuWarmup(){
#ifdef _OPENACC
  acc_init(acc_device_nvidia);
#endif

#ifdef CUDA
	floatType* a;
	cudaMalloc((void**)&a, 1000);
  cudaFree(a);
#endif
}



