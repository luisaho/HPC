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


#ifndef __CHECK_ERROR_H__
#define __CHECK_ERROR_H__

#include "def.h"
int check_error(const floatType bnrm2, const floatType residual, const floatType cg_tol);
floatType get_residual(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* const b, const floatType* const x);
#endif
