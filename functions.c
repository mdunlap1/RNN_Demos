/*  Author: Michael Dunlap
 *    Date: November 2023
 * Purpose: Helper functions for small neural network project. 
 */

#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>

static gsl_rng *RNG;

/* reset the seed of RNG */
void reset_seed(int k)
{
    if (k == 0) {
        printf("DO NOT USE zero for RNG seed\n");
    }
    gsl_rng_set(RNG, k);
}

/* Allocates memory for RNG*/
void alloc_rng(void) {
    RNG = gsl_rng_alloc(gsl_rng_ranlux389);
    gsl_rng_set(RNG, 999996);
}


/* Frees RNG */
void free_rng() {
    gsl_rng_free(RNG);
}


/* Sigmoidal activation function */
double sigmoid(double x)
{
    x = -1 * x;
    return (double) 1 / (1 + exp(x));
}


/* Derivative of sigmoidal activation function */
double d_sigmoid(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}


/* Allocates memory needs to be freed 
 *
 * Gaussian RV generation modified and borrowed from: 
 *     https://www.math.arizona.edu/~stepanov/math_575b_notes.pdf
 *     (Page 26 Example 23.2 Idea 3)
 *
 * Generates a random epoch of inputs of length epoch_len. 
 * Each input is of length num_external_inputs.
 * 
 * Data is Gaussian mean 0 variance 1
 */
double **gen_epoch(int epoch_len, int num_external_inputs)
{
    double **epoch = malloc(sizeof(double*) * epoch_len);

    int i, j;
    double u[2], r, x, y;
    
    for (i = 0; i < epoch_len; i++)
    {
        double *input = malloc(sizeof(double) * num_external_inputs);
        int fill = 0;
        while (fill != num_external_inputs) {

            // Generate and store 2 Uniform(0,1) RVs
            for (j = 0; j < 2; j++)
                u[j] = gsl_rng_uniform(RNG);

            r = sqrt(-2. * log(u[0]));

            x = r * cos(2. * M_PI * u[1]);

            y = r * sin(2. * M_PI * u[1]);
            
            input[fill++] = x;
            if (fill < num_external_inputs) {
                input[fill++] = y;
            }
        }
        epoch[i] = input;
    }


    return epoch;

}


/* Wrapper function for sigmoid */
double f_(double x)
{
    return sigmoid(x);
}


/* Wrapper function for d_sigmoid */
double d_f(double x)
{
    return d_sigmoid(x);
}


/* Formula (5) pg 25 of Long Short-term Memory paper */
double g_(double x)
{
    return 4 * f_(x) - 2;
}


/* Derivative of g_ */
double d_g(double x)
{
    return 4*d_f(x);
}


/* Formula (4) pg 25 of Long Short-term Memory Paper */
double h_(double x)
{
    return 2*f_(x) - 1;
}


/* Derivative of h_ */
double d_h(double x)
{
    return 2*d_f(x);
}


/* Returns a uniform random value between -val and val */
double symmetric_uniform(double val)
{
    double x = gsl_rng_uniform(RNG);

    /* Scale by 2*val */
    x *= 2 * val;

    /* Shift to make fall in range between -val and val */
    x -= val;

    return x;
}


/* Wrapper function for gsl_rng_uniform */
double uniform()
{
    return gsl_rng_uniform(RNG);
}


/* Returns an integer between a and b (inclusive) */
int rando(int a, int b)
{
    int n = b - a + 1;

    /* Returns random int in [0, n-1] */
    int k = gsl_rng_uniform_int(RNG, n); 

    /* Shift: 0 + a = a and (b - a + 1) - 1 + a = b */
    return k + a;
}


/* Crude test of symmetric uniform */
void test_symmetric_uniform(double val)
{
    int trials = 1000;
    double x;
    for (int i=0; i < trials; i++) {
        x = symmetric_uniform(val);

        if (x > val || x < -1* val) {
            printf("FAILED test for val %f\n", val);
        }
    }

    printf("symmetric_uniform passed %d test cases with val=%f\n", trials, val);

    return;
}


/* Crude test of rando */
void test_rando()
{
    int trials = 1000;
    int a = 0;
    int b = 1;

    int k;
    for (int i=0; i < trials; i++) {
        k = rando(a,b);
        if (k < a || k > b) {
            printf("Failed test in rando\n");
        }
    }

    a = 3;
    b = 6;
    for (int i=0; i < trials; i++) {
        k = rando(a,b);
        if (k < a || k > b) {
            printf("Failed test in rando 2\n");
        }
    }

    for (int i=0; i < trials; i++) {
        k = rando(a,b);
        printf("%d\n", k);
    }

    /* Does this case work? */
    int ret_val = rando(5,5);
    if (ret_val != 5) {
        printf("Faied singletton test in rando 3\n");
    }
}


/* Allocates memory needs to be freed
 *
 * Generates an array of randomized weights.
 * Used to make random initilizations of the network
 *
 * Will make an array of length num_weights and fill with numbers
 * that are uniformly distributed on (-val, val)
 */
double *gen_weights(int num_weights, double val)
{
    double *weights = malloc(sizeof(double) * num_weights);

    for (int i=0; i < num_weights; i++) {
        double x = symmetric_uniform(val);
        weights[i] = x;
    }

    return weights;
}
