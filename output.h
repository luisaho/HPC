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


#ifdef __cplusplus
extern "C" {
#endif
extern void output(char **argv, const char *name, char type, ...)
#ifdef __GNUC__
    __attribute__((__sentinel__(0)))
#endif
;
#ifdef __cplusplus
}
#endif
