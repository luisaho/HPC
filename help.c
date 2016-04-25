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

#include "help.h"

/* Print out the usage of the application */
void help(const char *argv0) {
	printf("Usage: %s matrix\n"
	    "\n"
	    "matrix has to be a matrix market (mtx) file.\n"
	    "\n"
	    "Environment variables:\n"
	    "\tCG_MAX_ITER\tMaximum number of iterations.\n"
	    "\tCG_TOLERANCE\tAllowed tolerance after which to stop.\n"
	    "The defaults are:\n"
	    "\tCG_MAX_ITER\t1000\n"
	    "\tCG_TOLERANCE\t0.0000001\n"
	    "\n", argv0);
}
