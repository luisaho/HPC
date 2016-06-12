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
#include <math.h>

#ifdef _OPENACC
# include <openacc.h>
#endif

#ifdef CUDA
# include <cuda.h>
#endif

#include "solver.h"
#include "output.h"


/* ab <- a' * b */
void vectorDot(const floatType* a, const floatType* b, const int n, floatType* ab) {
	int i;
	floatType temp = 0;
#pragma acc parallel num_gangs(64) vector_length(192) present(a[0:n],b[0:n])
#pragma acc loop reduction(+:temp) private(i) 
	for (i=0; i<n; i++){
		temp += a[i]*b[i];
	}
	*ab = temp;
}

/* y <- ax + y */
void axpy(const floatType a, const floatType* x, const int n, floatType* y){
	int i;

#pragma acc parallel num_gangs(64) vector_length(192) present(x[0:n], y[0:n])
#pragma acc loop gang vector  
	for (i = 0; i < n; i++) {
		y[i]=a*x[i]+y[i];
	}
}

/* y <- x + ay */
void xpay(const floatType* x, const floatType a, const int n, floatType* y){
	int i;

#pragma acc parallel num_gangs(64) vector_length(192) present(x[0:n],y[0:n])
#pragma acc loop gang vector 
	for (i = 0; i < n; i++) {
		y[i]=x[i]+a*y[i];
	}
}


/* y <- A*x
 * Remember that A is stored in the ELLPACK-R format (data, indices, length, n, nnz, maxNNZ). */
void matvec(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* x, floatType* y){

#pragma acc parallel present(data[0:n*maxNNZ], indices[0:n*maxNNZ], length[0:n], x[0:n], y[0:n])
#pragma acc loop gang vector
	for (int i = 0; i < n; i++) {
		floatType temp = 0;	
	#pragma acc loop
		for (int j = 0; j < length[i]; j++) {
			int k = j * n + i;
			temp += data[k] * x[indices[k]];
		}

		y[i] = temp;
	}
}



/* nrm <- ||x||_2 */
void nrm2(const floatType* x, const int n, floatType* nrm) {
	int i;	
	floatType temp;
	temp = 0;
#pragma acc parallel num_gangs(64) vector_length(192) present(x[0:n])
#pragma acc loop reduction(+:temp)
	for (i=0; i<n; i++){
		temp += x[i]*x[i];
	}
	*nrm = 1/sqrt(temp);
}


void vectorSquare(const floatType* x, const int n, floatType* sq) {
	int i;
	floatType temp = 0;

#pragma acc parallel num_gangs(64) vector_length(192) present(x[0:n])
#pragma acc loop reduction(+:temp)
	for (i=0; i<n; i++){
		temp += x[i]*x[i];
	}

	*sq = temp;
}



void diagMult(const floatType* diag, const floatType* x, const int n, floatType* out) {

#pragma acc parallel num_gangs(64) vector_length(192) present(x[0:n],diag[0:n], out[0:n])
#pragma acc loop gang vector
	for (int i=0; i<n; i++){
		out[i] = x[i]*diag[i];
	}
}

void getDiag(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, floatType* diag) {

#pragma acc parallel num_gangs(64) vector_length(192) present(data[0:n*maxNNZ], indices[0:n*maxNNZ], length[0:n], diag[0:n])
#pragma acc loop gang vector
	for (int i=0; i<n; i++) {
		for (int j = 0; j < length[i]; j++) {
			int idx = j*n + i;
			int realcol = indices[idx];
			if (i == realcol) {
				diag[i] = 1/data[idx];
			}
		}
	}
}


/***************************************
 *         Conjugate Gradient          *
 *   This function will do the CG      *
 *  algorithm without preconditioning. *
 *    For optimiziation you must not   *
 *        change the algorithm.        *
 ***************************************
 r(0)    = b - Ax(0)
 p(0)    = r(0)
 rho(0)    =  <r(0),r(0)>                
 ***************************************
 for k=0,1,2,...,n-1
   q(k)      = A * p(k)                 
   dot_pq    = <p(k),q(k)>             
   alpha     = rho(k) / dot_pq
   x(k+1)    = x(k) + alpha*p(k)      
   r(k+1)    = r(k) - alpha*q(k)     
   check convergence ||r(k+1)||_2 < eps  
	 rho(k+1)  = <r(k+1), r(k+1)>         
   beta      = rho(k+1) / rho(k)
   p(k+1)    = r(k+1) + beta*p(k)      
***************************************/
void cg(const int n, const int nnz, const int maxNNZ, const floatType* restrict const data, const int* restrict const indices, const int* restrict const length, const floatType* restrict const b, floatType* restrict const x, struct SolverConfig* sc){

	floatType *r, *p, *q, *z, *diag;
	floatType alpha, beta, rho, rho_old, dot_pq, bnrm2, check;
	int iter;
	floatType timeMatvec_s;
	floatType timeMatvec=0;


	/* allocate memory */
	r = (floatType*)malloc(n * sizeof(floatType));
	p = (floatType*)malloc(n * sizeof(floatType));
	q = (floatType*)malloc(n * sizeof(floatType));
	z = (floatType*)malloc(n * sizeof(floatType));
	diag = (floatType*)malloc(n * sizeof(floatType));

#pragma acc data create(r[0:n], q[0:n], diag[0:n], z[0:n]) copyin(data[0:n * maxNNZ], indices[0:n * maxNNZ], length[0:n], b[0:n]) copy(x[0:n])
{
	DBGMAT("Start matrix A = ", n, nnz, maxNNZ, data, indices, length)
	DBGVEC("b = ", b, n);
	DBGVEC("x = ", x, n);

	getDiag(n, nnz, maxNNZ, data, indices, length, diag);

	/* r(0)    = b - Ax(0) */
	timeMatvec_s = getWTime();
	matvec(n, nnz, maxNNZ, data, indices, length, x, r);
	timeMatvec += getWTime() - timeMatvec_s;
	xpay(b, -1.0, n, r);
	DBGVEC("r = b - Ax = ", r, n);

	diagMult(diag, r, n, z);
#pragma acc update host(z[0:n])

	memcpy(p, z, n * sizeof(floatType));
#pragma acc enter data copyin(p[0:n])
	
	/* Calculate initial residuum */
	nrm2(r, n, &bnrm2);

	/* p(0)    = r(0) */
	memcpy(p, r, n*sizeof(floatType));
	DBGVEC("p = r = ", p, n);

	/* rho(0)  = <r(0),z(0)> */ //hier anpassen
	vectorDot(r, z, n, &rho);
	vectorSquare(r, n, &check);
	printf("rho_0=%e/%e\n", rho, check);

	for(iter = 0; iter < sc->maxIter; iter++){
		DBGMSG("=============== Iteration %d ======================\n", iter);
		/* q(k)   = A * p(k) */
		timeMatvec_s = getWTime();
		matvec(n, nnz, maxNNZ, data, indices, length, p, q);
		timeMatvec += getWTime() - timeMatvec_s;
		DBGVEC("q = A * p= ", q, n);

		/* dot_pq  = <p(k),q(k)> */
		vectorDot(p, q, n, &dot_pq);
		DBGSCA("dot_pq = <p, q> = ", dot_pq);

		/* alpha   = rho(k) / dot_pq */
		alpha = rho / dot_pq;
		DBGSCA("alpha = rho / dot_pq = ", alpha);

		/* x(k+1)  = x(k) + alpha*p(k) */
		axpy(alpha, p, n, x);
		DBGVEC("x = x + alpha * p= ", x, n);

		/* r(k+1)  = r(k) - alpha*q(k) */
		axpy(-alpha, q, n, r);
		DBGVEC("r = r - alpha * q= ", r, n);


		rho_old = rho;
		DBGSCA("rho_old = rho = ", rho_old);


		/* rho(k+1) = <r(k+1), z(k+1)> */ //hier anpassen
		diagMult(diag, r, n, z);

		vectorDot(r, z, n, &rho);
		vectorSquare(r, n, &check);
		DBGSCA("rho = <r, r> = ", rho);

		/* Normalize the residual with initial one */
		sc->residual = sqrt(check) * bnrm2;
  	


		/* Check convergence ||r(k+1)||_2 < eps
		 * If the residual is smaller than the CG
		 * tolerance specified in the CG_TOLERANCE
		 * environment variable our solution vector
		 * is good enough and we can stop the 
		 * algorithm. */
		printf("res_%d=%e\n", iter+1, sc->residual);
		if(sc->residual < sc->tolerance) {
			break;
		}
		

		/* beta   = rho(k+1) / rho(k) */
		beta = rho / rho_old;
		DBGSCA("beta = rho / rho_old= ", beta);

		/* p(k+1)  = r(k+1) + beta*p(k) */
		xpay(z, beta, n, p);
		DBGVEC("p = r + beta * p> = ", p, n);
	}
	}//end data region



	/* Store the number of iterations and the 
	 * time for the sparse matrix vector
	 * product which is the most expensive 
	 * function in the whole CG algorithm. */
	sc->iter = iter;
	sc->timeMatvec = timeMatvec;

	/* Clean up */
	free(r);
	free(p);
	free(q);
}
