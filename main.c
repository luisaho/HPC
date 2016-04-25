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



#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENACC
# include <openacc.h>
#endif 

#ifdef CUDA
# include <cuda/cuda.h>
#endif

#include "def.h"
#include "help.h"
#include "solver.h"
#include "errorcheck.h"
#include "output.h"
#include "io.h"


/* Init the right hand side (rhs), so that the solution is one for 
 * every entry in the x vector. Please note that due to rounding
 * errors this solution will not be reached for the implemented cg
 * method. For the error checking it is enought to check the residual. */
void initLGS(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, floatType* b, floatType* x){
	int i,j;
	memset(b, 0, n * sizeof(floatType));
	for(i = 0; i < n; i++){
		x[i] = 0;
		for (j = 0; j < length[i]; j++) {
			b[i] += data[j * n + i];
		}
	}
}

int main(int argc, char *argv[]){
	struct SolverConfig sc;
	floatType *b, *x;
	floatType residual, bnrm2;
	int correct;
	double ioTime, solveTime, totalTime;

	/* The folloing variables are used to 
	 * represent the matrix which is saved in a 
	 * special format, called ELLPACK-R (or ELL).
	 * It is basicly designed for sparse matrices
	 * which have a lot of zeros to save memory.
	 * The ELLPACK-R format is especially well
	 * suited for GPUs as it allows simple
	 * coalesced memory access.
	 * The dimension of the matrix is stored in n,
	 * the number of non zeros in nnz, and the
	 * maximum number of non-zeros in one row
	 * in maxNNZ.
	 * The ELLPACK-R format stores maxNNZ*n values.
	 * Thus it is well suited for matrices which
	 * have approximately the same number of non
	 * zeros per row.
	 * In the worst case one row is fully occupied. 
	 * Then the ELLPACK-R format will store every
	 * value (n*n) even if all other elements are
	 * zero.
	 * The ELLPACK-R format uses three arrays data,
	 * indices, and length to store data:
	 * * data is a double precision array with a
	 *   length of maxNNZ*n, which stores the
	 *   nonzero elements (and some zeros for
	 *   padding) of the matrix shifted to the
	 *   right, in a COLUMN MAJOR order.
	 * * indices is an integer array with the
	 *   same length as data. It stores the
	 *   column numbers of the elements stored
	 *   in the data array.
	 * * length is an integer array of length
	 *   n, which stores the number of non zeros
	 *   per row.
	 * To construct the ELLPACK-R format, do
	 * the following steps:
	 *  1. Shift all non zeros to the left.
	 *  2. Count the non zeros in each row
	 *     and compute thereby maxNNZ.
	 *  3. Go through the shifted matrix
	 *     column by column and save every
	 *     element of the first maxNNZ
	 *     columns.
	 *
	 * Example:
	 *    ( 11             )
	 *    ( 21  22         )
	 * A =(     32  33     )
	 *    ( 41      43  44 )
	 *
	 * Shifted matrix:
	 *      ( 11  0   0      )
	 *      ( 21  22  0      )
	 *  A' =( 32  33  0      )
	 *      ( 41  43  44     )
	 *
	 *
	 * The matrix A would be stored as followed in 
	 * the ELLPACK-R format:
	 *
	 * A.n = 4
	 * A.maxNNZ = 3
	 *
	 * A.data    |11|21|32|41| 0|22|33|43| 0| 0| 0|44|
	 * A.indices | 0| 0| 1| 0| 1| 1| 2| 2| 2| 2| 3| 3|
	 *
	 * A.length  | 1| 2| 2| 3| */
	int n;            
	int nnz;           
	int maxNNZ;
	floatType* data = NULL;
	int* indices = NULL;
	int* length = NULL;
	


	/* Check the input parameter and print "usage" in case of 
	 * too many / too less parameters */
	if (argc < 2 || argc >3) {
		help(argv[0]);
		return 1;
	} else if (!strcmp(argv[1], "-h")) {
		help(argv[0]);
		return 0;
	}

	/* Check environment variables CG_TOLERANCE and CG_MAX_ITER */
	init();

	/* Start time measurement for the total time */
	totalTime = getWTime();

	/* Parse the matrix, stop the time for IO. Please note that
	 * you do not have to optimize the IO time, because only the 
	 * solving time will be valued.  */
	ioTime = getWTime();
	parseMM(argv[1], &n, &nnz, &maxNNZ, &data, &indices, &length);
	ioTime = getWTime() - ioTime;

	/* Allocate memory for the LGS */
	b = (floatType*)malloc(n * sizeof(floatType));
	x = (floatType*)malloc(n * sizeof(floatType));

	/* Init the LGS */
	initLGS(n, nnz, maxNNZ, data, indices, length, b, x);

	/* Calculate the initial residuum for error checking */
	bnrm2 = get_residual(n, nnz, maxNNZ, data, indices, length, b, x);
	
	/* Set the solver configuration */
	sc.maxIter = config.maxIter;
	sc.tolerance = config.tolerance;


	/* Solving the system of linear equations including the time measurement.
	 * You should try to optimize this time, this will be valued for the
	 * competition. */
	solveTime = getWTime();
	cg(n, nnz, maxNNZ, data, indices, length, b, x, &sc);
	solveTime = getWTime()-solveTime;

	/* Print solution vector x or the first 10 values of the result. 
	 * Should be 1 in case of convergence. */
	if (n > 10){
		printf("First 10 values of the solution vector x = ");
		printVector(x, 10);
	} else {
		printf("Solution vector x = ");
		printVector(x, n);
	}
	
	/* Check error */
	residual = get_residual(n, nnz, maxNNZ, data, indices, length, b, x);
	correct = check_error(bnrm2, residual, sc.tolerance);

	FILE *fp;
	int i;

	if ((fp = fopen("x.out", "w")) == NULL) {
		fprintf(stderr, "Faield to write output file %s!\n", "x.out");
		exit(1);
	}

	for (i = 0; i < n; i++) {
		fprintf(fp, "%e\n", x[i]);
	}

	fclose(fp);



	/* Clean up */
	free(b);
	free(x);
	destroyMatrix(data, indices, length);

	totalTime = getWTime() - totalTime;

	/* Print out some information */
	output(
	    argv,
	    "NNZ", 'i', nnz,
	    "N", 'i', n,
	    "Max. iterations", 'i', sc.maxIter,
	    "Tolerance", 'e', sc.tolerance,
	    "Residual", 'e', sc.residual,
	    "Iterations", 'i', sc.iter,
	    "MatVec time", 'f', sc.timeMatvec,
	    /* TODO: Implement the calculation for the FLOPS of the here. */ 
	    /* Hint: Refer to solve.c and think about how many opertions */
	    /*       are done in the innmost loop and how often this is done.*/
	    "Hotspot GFLOP/s", 'f', ((2.0 * ((double)nnz) * ((double)(sc.iter+1))) / (sc.timeMatvec * 1000000000.0)),
	    "IO time", 'f', ioTime,
	    "Solve time", 'f', solveTime,
	    "Total time", 'f', totalTime,
			"RESULT CHECK", 's', correct == 0 ? "ERROR" : "OK", 
	    (const char*)NULL
	);

	return 0;
}
