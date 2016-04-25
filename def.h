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



#ifndef __DEF_H__
#define __DEF_H__

/* Define some ugly macros to wrap printf(). You can use
 * them to print out some debug stuff which do not want to
 * see in the productive runs. Deactivate by removing
 * "-DDEBUG" in the Makefile. */
#ifdef DEBUG
# define DBGMSG(msg, ...) printf(msg, ##__VA_ARGS__);
# define DBGSCA(msg, a) printf(msg); printf("%e\n", a);
# define DBGVEC(msg, x, n) printf(msg); printVector(x,n);
# define DBGMAT(msg, n, nnz, maxNNZ, data, indices, length) printf(msg); printMatrix(n, nnz, maxNNZ, data, indices, length);
# define DBGCRSMAT(mmsg, n, nnz, ptr, index, value) printf(msg); printCRSMatrix(n, nnz, ptr, index, value);
#else
# define DBGMSG(msg, ...)
# define DBGSCA(msg, a) 
# define DBGVEC(msg, x, n)
# define DBGMAT(msg, n, nnz, maxNNZ, data, indices, value) 
# define DBGCRSMAT(msg, n, nnz, ptr, index, value) 
#endif

/* Define floatType as double */
typedef double floatType;

/* This structure is to used to configure 
 * the parameters for the CG algorithm */
extern struct config {
	int maxIter;
	floatType tolerance;
} config;


/* This structure is to used to configure 
 * the parameters for the CG algorithm */
struct SolverConfig {
	floatType tolerance;
	int iter;
	int maxIter;
	floatType residual;
	floatType timeMatvec;
};

extern void init(void);
extern double getWTime(void);
void gpuWarmup();

#endif
