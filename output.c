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
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <libgen.h>

#include "output.h"

#ifdef __INTEL_COMPILER
/* Useless warning: "transfer of control bypasses initialization of:" */
# pragma warning(disable: 589)
#endif

#define SKIP(list, type)					\
	switch (type) {						\
	case 's':;						\
		const char *SKIPtmp = va_arg(ap2, const char*);	\
		assert(SKIPtmp != NULL);			\
		(void)SKIPtmp;					\
		break;						\
	case 'i':						\
		(void)va_arg(ap2, int);				\
		break;						\
	case 'e':						\
	case 'f':						\
	case 'g':						\
		(void)va_arg(ap2, double);			\
		break;						\
	default:						\
		assert(0);					\
	}
#define LINE(longest, name, type)				\
	switch (type) {						\
	case 's':;						\
		const char *LINEvalue;				\
		LINEvalue = va_arg(ap, const char*);		\
		assert(LINEvalue != NULL);			\
		line(longest, name, type, LINEvalue);		\
		break;						\
	case 'i':						\
		line(longest, name, type, va_arg(ap, int));	\
		break;						\
	case 'e':						\
	case 'f':						\
	case 'g':						\
		line(longest, name, type, va_arg(ap, double));	\
		break;						\
	}

static void
line(size_t width, const char *name, char type, ...)
{
	va_list ap;
	size_t i, left = width - strlen(name);

	printf("\033[1m%s\033[0m: ", name);

	for (i = 0; i < left; i++)
		putchar(' ');

	va_start(ap, type);
	switch (type) {
	case 's':
		puts(va_arg(ap, const char*));
		break;
	case 'i':
		printf("%d\n", va_arg(ap, int));
		break;
	case 'e':
		printf("%.0le\n", va_arg(ap, double));
		break;
	case 'f':
		printf("%lf\n", va_arg(ap, double));
		break;
	case 'g':
		printf("%lg\n", va_arg(ap, double));
		break;
	default:
		assert(0);
	}
}

void output(char **argv, const char *name, char type, ...)
{
	va_list ap, ap2;
	size_t longest;
	const char *tmp;
	char *tmp2;
	char hostname[256];
	va_start(ap, type);
	va_copy(ap2, ap);

	longest = strlen("Build date");
	if (strlen(name) > longest)
		longest = strlen(name);
	SKIP(ap2, type)
	while ((tmp = va_arg(ap2, const char*)) != NULL) {
		assert(tmp != NULL);

		char t = va_arg(ap2, int);

		size_t len = strlen(tmp);

		if (len > longest)
			longest = len;

		SKIP(ap2, t)
	}

	tmp2 = strdup(argv[1]);
	line(longest, "Matrix", 's', basename(tmp2));
	free(tmp2);

	tmp2 = strdup(argv[0]);
	line(longest, "Version", 's', basename(tmp2));
	free(tmp2);

	line(longest, "Build date", 's', __DATE__ " " __TIME__);

	gethostname(hostname, 256);
	line(longest, "Hostname", 's', hostname);

	LINE(longest, name, type)
	while ((name = va_arg(ap, const char*)) != NULL) {
		type = va_arg(ap, int);

		LINE(longest, name, type)
	}
}
