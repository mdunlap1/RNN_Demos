/*  Author: Michael Dunlap
 *    Date: November 2023
 * Purpose: Basic matrix helper functions for small neural network project.
 *          Of course we could have used the routines from GNU gsl, this was
 *          done instead partly for fun.
 */

#include <stdio.h>
#include <stdlib.h>

typedef struct matrix {
    int rows;
    int cols;
    double **d;
} Matrix;


/* Allocate memory for Matrix struct and internal data, set values.
 * Matrix data defaults to zeros 
 *
 * Needs to be freed */
Matrix *init_matrix(int rows, int cols)
{
    Matrix *m = malloc(sizeof(Matrix));
    if (m == NULL) {
        exit(EXIT_FAILURE);
    }
    m->d = malloc(sizeof(double*) * rows);
    for (int i=0; i < rows; i++) {
        double *row = calloc(cols, sizeof(double)); // zeroed out
        if (row == NULL) {
            exit(EXIT_FAILURE);
        }

        m->d[i] = row;
    }

    m->rows = rows;
    m->cols = cols;

    return m;
}


/* Frees inernal data of Matrix struct and the pointer to the struct */
void free_matrix(Matrix *m)
{
    /* free the rows of the matrix first */
    for (int i = 0; i < m->rows; i++) {
        free(m->d[i]);
    }

    /* free the matrix outer array */
    free(m->d);

    /* free the matrix */
    free(m);

    return;
}


/* Takes an array of doubles and sets the values of a matrix to those values
 * in a row by row manner
 */
void row_stream_set(Matrix *m, double *values)
{
    int idx = 0;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->d[i][j] = values[idx++];
        }
    }

    return;
}


/* Basic printing function for a matrix */
void crude_print_matrix(Matrix *m)
{
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%.2e ", m->d[i][j]);
        }
        printf("\n");
    }
}


/* Uses malloc needs to be freed */
double *vector_matrix_prod(double *vector, Matrix *m)
{
    double *result = malloc(sizeof(double) * m->cols);
    for (int j=0; j < m->cols; j++) {
        double summa = 0.0;
        for (int i = 0; i < m->rows; i++) {
            summa += m->d[i][j] * vector[i];
        }
        result[j] = summa;
    }

    return result;

}


/* Uses malloc needs to be freed */
double *matrix_vector_prod(Matrix *m, double *vector)
{
    double *result = malloc(sizeof(double) * m->rows);
    for (int i=0; i < m->rows; i++) {
        double summa = 0.0;
        for (int j = 0; j < m->cols; j++) {
            summa += m->d[i][j] * vector[j];
        }
        result[i] = summa;
    }

    return result;
}


void print_row_vector(double *vector, int len)
{
    for (int i=0; i<len; i++) {
        printf("%g ", vector[i]);
    }

    printf("\n");
}


/* Mutate Matrix m1 to contain the matrix sum m1 + m2 */
void sum_m2_into_m1(Matrix *m1, Matrix *m2)
{
    if (m1->rows != m2->rows || m1->cols != m2-> cols) {
        fprintf(stderr, "Matrix dimensions do not match\nexiting...\n");
        exit(EXIT_FAILURE);
    }

    for (int i=0; i<m1->rows; i++) {
        for (int j=0; j<m1->cols; j++) {
            m1->d[i][j] += m2->d[i][j];
        }
    }

    return;
}


/* Mutate Matrix m by scalar multiplying c into it */
void mult_c_into_m(Matrix *m, double c)
{
    for (int i=0; i<m->rows; i++) {
        for (int j=0; j<m->cols; j++) {
            m->d[i][j] *= c;
        }
    }
}


/* Return the transpose of a matrix */
Matrix *tranpose(Matrix *m)
{
    Matrix *tmp = init_matrix(m->cols, m->rows);
    for (int i = 0; i < m->rows; i++) {
        for (int j=0; j < m->cols; j++) {
            tmp->d[j][i] = m->d[i][j];
        }
    }

    return tmp;
}


/* ALLOCATES MEMORY NEEDS TO BE FREED 
 * Calculates a component-wise product of two vectors
 */
double *c_prod(double *v1, double *v2, int len)
{
    double *prod = malloc(len * sizeof(double));
    for (int i=0; i < len; i++) {
        double x = v1[i] * v2[i];
        prod[i] = x;
    }

    return prod;
}
            

int test()
{
    Matrix *m = init_matrix(3,3);
    double values[9] = {1,2,3,4,5,6,7,8,9};
    row_stream_set(m, values);

    crude_print_matrix(m);

    double vector[3] = {1,0,0};
    double *result = vector_matrix_prod(vector, m);
    print_row_vector(result, 3);

    double *res2 = matrix_vector_prod(m,vector);
    print_row_vector(res2,3);

    free(result);
    free_matrix(m);

    return 0;
}


