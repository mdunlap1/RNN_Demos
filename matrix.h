#include <stdio.h>
#include <stdlib.h>

typedef struct matrix {
    int rows;
    int cols;
    double **d;
} Matrix;

/* Allocate memory for Matrix struct and internal data, set values.
 * Matrix data defaults to zeros 
 * Needs to be freed */
Matrix *init_matrix(int rows, int cols);

void free_matrix(Matrix *m);

void row_stream_set(Matrix *m, double *values);

void crude_print_matrix(Matrix *m);

/* Uses malloc needs to be freed */
double *vector_matrix_prod(double *vector, Matrix *m);

/* Uses malloc needs to be freed */
double *matrix_vector_prod(Matrix *m, double *vector);

void print_row_vector(double *vector, int len);

/* Mutate Matrix m1 to contain the matrix sum m1 + m2 */
void sum_m2_into_m1(Matrix *m1, Matrix *m2);

/* Mutate Matrix m by scalar multiplying c into it */
void mult_c_into_m(Matrix *m, double c);

/* ALLOCATES MEMORY NEED TO BE FREED
 * Returns the tranpose of the matrix
 * pointed to by m
 */
Matrix *tranpose(Matrix *m);

/* ALLOCATES MEMORY NEEDS TO BE FREED 
 * Calculates a component-wise product of two vectors
 */
double *c_prod(double *v1, double *v2, int len);
