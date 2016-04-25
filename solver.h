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


#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "def.h"

#ifdef __cplusplus
	extern "C" {
#endif
	void vectorDot(const floatType* a, const floatType* b, const int n, floatType* ab);
	void axpy(const floatType a, const floatType* x, const int n, floatType* y);
	void xpay(const floatType* x, const floatType a, const int n, floatType* y);
	void matvec(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* x, floatType* y);
	void nrm2(const floatType* x, const int n, floatType* nrm);
	void cg(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* b, floatType* x, struct SolverConfig* sc);
#ifdef __cplusplus
	}
#endif


#endif
