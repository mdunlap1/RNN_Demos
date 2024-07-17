#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>


double sigmoid(double x);

double d_sigmoid(double x);


void alloc_rng(void);

void free_rng();

double **gen_epoch(int epoch_len, int num_external_inputs);

double symmetric_uniform(double val);

/* Wrapper function for sigmoid */
double f_(double x);

/* Wrapper function for d_sigmoid */
double d_f(double x);

/* */
double g_(double x);

/* */
double d_g(double x);

/* */
double h_(double x);

/* */
double d_h(double x);

double uniform();

int rando(int a, int b);


/* For testing... */
void test_symmetric_uniform(double val);

void test_rando();

void reset_seed(int k);

/* Allocates memory needs to be freed
 *
 * Generates an array of randomized weights.
 * Used to make random initilizations of the network
 *
 * Will make an array of length num_weights and fill with numbers
 * that are uniformly distributed on (-val, val)
 */
double *gen_weights(int num_weights, double val);

