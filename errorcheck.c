/*****************************************************
 * CG Solver (HPC Software Lab)
 *
 * Parallel Programming Models for Applications in the 
 * Area of High-Performance Computation
 *====================================================
 * IT Center (ITC)
 * RWTH Aachen University, Germany
 * Author: Tim Cramer (cramer@itc.rwth-aachen.de)
 * 	   Fabian Schneider (f.schneider@itc.rwth-aachen.de)
 * Date: 2010 - 2015
 *****************************************************/


#include "errorcheck.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Check if the normalized residual (which is a ration of the final and the inital residual)
 * is smaller than the specified CG tolerance. 0 is returned in case of errors, 1 otherwise. */
int check_error(const floatType bnrm2, const floatType residual, const floatType cg_tol){
	return ((residual / bnrm2) <= cg_tol) ? 1 : 0;
}

/* Calculate the current residual for error checking. You must not change this function, 
 * it is not used to during the algorithm. There is no need to parallelize it. */
floatType get_residual(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* const b, const floatType* const x){
	int i,j, k;
	floatType* y;
	floatType residual;

	/* Allocate residual vector */
	y = (floatType*)malloc(n * sizeof(floatType));

	/* y = A * x */
	for (i = 0; i < n; i++) {
		y[i] = 0;
		for (j = 0; j < length[i]; j++) {
			k = j * n + i;
			y[i] += data[k] * x[indices[k]];
		}
	}

	/* y = | b - y | */
	for(i = 0; i < n; i++){
		y[i] = fabs(b[i]-y[i]);
	}	

	/* residual = || y ||_2 */
	residual = 0;
	for(i=0; i < n; i++){
		residual += y[i] * y[i];
	}
	residual = sqrt(residual);

	/* Clean up */
	free(y);
	
	return residual;
}

