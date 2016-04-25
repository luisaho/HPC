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


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#ifndef _WIN32
# include <fcntl.h>
# include <unistd.h>
# include <sys/types.h>
# include <sys/stat.h>
# include <sys/mman.h>
#endif

#include "io.h"
#include "mmio.h"

/* Parse the matrix market file "filename" and return
 * the matrix in ELLPACK-R format in A. */
void parseMM(char *filename, int* n, int* nnz, int* maxNNZ, floatType** data, int** indices, int** length){
	int M,N;
	int i,j;
	int *I, *J, *offset;
	floatType *V;
	FILE *fp;
	MM_typecode matcode;

	printf("Start matrix parse.\n");

	/* Try to open the file and return a error if not possible */
	if ((fp = fopen(filename, "r")) == NULL) {
		printf("ERROR: Cant open file!\n");
		exit(1);
	}

	/* Read the banner of the matrix market matrix. You should
	 * not care about the details of the file format at this 
	 * point. Exit in case of unsupported files. */
	if (mm_read_banner(fp, &matcode) != 0) {
		printf("ERROR: Could not process Matrix Market banner.\n");
		exit(1);
	}

	/* This is how one can screen matrix types if their application 
	 * only supports a subset of the Matrix Market data types.
	 * Please dont care about the details! */
	if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
	    mm_is_sparse(matcode)) {
		printf("ERROR: Sorry, this application does not support ");
		printf("Market Market type: [%s]\n",
		    mm_typecode_to_str(matcode));
		exit(1);
	}

	/* Find out size of sparse matrix from the file */
	if (mm_read_mtx_crd_size(fp, &M, &N, nnz) != 0) {
		printf("ERROR: Could not read matrix size!\n");
		exit(1);
	}

	/* Exit for non square matrices. */
	if (N != M) {
		printf("ERROR: Naahhh. Come on, give me a NxN matrix!\n");
		exit(1);
	}

	printf("Start memory allocation.\n");

	/* if the matrix is stored in the symmetric format we will
	 * increase the number of nnz to store the upper and lower triangular */
	if (mm_is_symmetric(matcode)){

		/* store upper and lower triangular */
		(*nnz) = 2 * (*nnz) - N;
	}



	/* Allocate some of the memory for the ELLPACK-R matrix */
	*length = (int*) malloc(sizeof(int) * N);

	/* Check if the memory was allocated successfully */
	if (*length == NULL) {
		puts("Out of memory!");
		exit(1);
	}

	/* Initialize the length pointer of the matrix */
	for (i = 0; i < N + 1; i++) {
		(*length)[i] = 0;
	}

	/* Set the number of non zeros (nnz) and the dimension (n)
	 * of the matrix */
	*n = N;

	/* Alocate the temporary  memory for matrix market matrix which
	 * has to be converted to the ELLPACK-R format */
	I = (int*)malloc(sizeof(int) * (*nnz));
	J = (int*)malloc(sizeof(int) * (*nnz));
	V = (floatType*)malloc(sizeof(floatType) * (*nnz));

	/* Check if the memory was allocated successfully */
	if (I == NULL || J == NULL || V == NULL) {
		puts("Out of memory!");
		exit(1);
	}

	/* Allocate and initialize some more temporay memory */
	if ((offset = (int*)malloc(sizeof(int) * (*nnz))) == NULL) {
		puts("Out of memory!");
		exit(1);
	}
	memset(offset, 0, (*nnz) * sizeof(int));

	printf("Read from file.\n");

	/* Start reading the file and store the values */
	for (i = 0; i < (*nnz); i++) {
		fscanf(fp, "%d %d %lg\n", &I[i], &J[i], &V[i]);

		/* count double if entry is not on diag and in symmetric file format */
		if (I[i] != J[i] && mm_is_symmetric(matcode)){
			((*length)[I[i]-1])++;
			i++;
			I[i] = J[i-1];
			J[i] = I[i-1];

			I[i-1]--;  /* adjust from 1-based to 0-based */
			J[i-1]--;

			V[i] = V[i-1];
		}


		 /* Adjust from 1-based to 0-based which means that in
		  * the matrix market file format the first index is
		  * always 1, but in C the first index is always 0. */
		I[i]--; 
		J[i]--;

		/* Count entries in one row */
		((*length)[I[i]])++;
	}


	printf("Start converting from MM to ELLPACK-R.\n");

	/* Get maximum Number of NNZs per row for the ELLPACK-R format */
	*maxNNZ = 0;
	for (i = 0; i < N; i++) {
		if ((*length)[i] > (*maxNNZ)) {
			(*maxNNZ) = (*length)[i];
		}
	}

	/* Allocate the rest of the memory for the ELLPACK-R matrix */
	*data = (floatType*) malloc(sizeof(floatType) * N * (*maxNNZ));
	*indices = (int*) malloc(sizeof(int) * N * (*maxNNZ));

	/* Check if the memory was allocated successfully */
	if (*data == NULL || *indices == NULL) {
		puts("Out of memory!");
		exit(1);
	}

	/* Convert from MM to ELLPACK-R */
	for (j = 0; j < (*nnz); j++){
		i = I[j];

		/* Store data and indices in column-major order */
		(*data)[offset[i] * N + i] = V[j];
		(*indices)[offset[i] * N + i] = J[j];
		
		offset[i]++;
	}

	/* Insert 0's for padding in data and indices array */
	for (i = 0; i < N; i++) {
		for (j = (*length)[i]; j < (*maxNNZ); j++) {
			(*data)[j * N + i] = 0.0;
			(*indices)[j * N + i] = 0;
		}
	}

	printf("MM Parse done.\n");

	/* Clean up */
	free(offset);
	free(I);
	free(J);
	free(V);
	fclose(fp);
}

/* Free the complete memory of the matrix in ELLPACK-R format */
void destroyMatrix(floatType* data, int* indices, int* length) {
	free(data);
	free(indices);
	free(length);
}

/* Print out to std the first n elements of the vector x */
void printVector(const floatType *x, const int n) {
	int i;

	printf("(");
	for (i = 0; i < n; i++)
		printf("%d:%e' ", i, x[i]);
	printf(")\n");
}


/* Print out the whole ELLPACK-R matrix to std */
void printMatrix(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length) {
	int i, j, k;

	for (i = 0; i < n; i++) {
		if (i == 0) {
			printf("Row %d: [", 0);
		} 
		else {
			printf("]\nRow %d: [", i);
		}
		for (j = 0; j < length[i]; j++) {
			k = j * n + i;
			printf("%d:", indices[k]);
			printf("%f' ", data[k]);
		}
	}

	printf("]\n");
}


