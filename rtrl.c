/*  Author: Michael Dunlap
 *    Date: November 2023
 * Purpose: Crude implementation of a single layer RNN and the Real Time 
 *          Recurrent Learning algorithm based on the paper "Gradient-Based 
 *          Learning Algorithms for Recurrent Networks and Their Computational 
 *          Complexity" by Williams and Zisper. 
 *
 *          Main objective is educational. Simply write the method and verify it
 *          using the finite difference method. 
 */ 

#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "functions.h"


typedef struct network {
    Matrix *w;
    int num_units;
    int num_external_inputs;
    double *activations;
    double ***p; // Sensitivities
    Matrix *g;   // Gradient (stored as a Matrix
} Network;


/*
 * Allocates memory needs to be freed
 *
 * Allocate memory for RTRL network struct and internal structures
 * returns a pointer to the struct.
 */
Network *init_network(int num_units, int num_external_inputs, double *weights)
{
    Network *n = malloc(sizeof(Network));
    if (n == NULL) {
        fprintf(stderr, "Failure to allocate Network memory\n");
        exit(EXIT_FAILURE);
    }

    int rows = num_units;
    int cols = num_units + num_external_inputs + 1; // Plus 1 for bias

    /* These default to zero matrix */
    Matrix *w = init_matrix(rows, cols); 
    Matrix *g = init_matrix(rows, cols);

    double *activations = calloc(num_units, sizeof(double));
    if (activations == NULL) {
        fprintf(stderr, "Failure to allocate activations memory\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate memory for network sensitivities */
    /* p will have the structure p[k][i][j] */
    double ***p = malloc(sizeof(double**) * num_units);
    for (int k = 0; k < num_units; k++) {
        double **m = malloc(sizeof(double*) * num_units);
        for (int i = 0; i < num_units; i++) {
            double *row = calloc(cols, sizeof(double));
            m[i] = row;
        }
        p[k] = m;
    }

    n->w = w;
    n->g = g;
    n->p = p;
    n->activations = activations;
    n->num_units = num_units;
    n->num_external_inputs = num_external_inputs;

    row_stream_set(n->w, weights);

    return n;
}


/*
 * Frees memory internal memory of Network and the struct itself
 */
void free_network(Network *n)
{
    free_matrix(n->w);
    free_matrix(n->g);
    free(n->activations);

    for (int k=0; k < n->num_units; k++) {
        for (int i = 0; i < n->num_units; i++) {
            free(n->p[k][i]);
        }
        free(n->p[k]);
    }
    free(n->p);

    free(n);
}


/* Run a single input through the network */
void run(Network *n, double *input)
{
    /* For convenience */
    int num_units = n->num_units;
    int num_external_inputs = n->num_external_inputs;

    /* Build input vector */
    double *x = malloc(sizeof(double) * (num_units + num_external_inputs + 1));

    /* External inputs */
    for (int i=0; i < num_external_inputs; i++) {
        x[i] = input[i];
    }
    /* Recurrent inputs */
    for (int i=0; i < num_units; i++) {
        x[num_external_inputs + i] = n->activations[i];
    }
    x[num_external_inputs + num_units] = 1.0; // Bias

    double *net_out = matrix_vector_prod(n->w, x);

    /* Use (33) to update the sensitivies */
    for (int i = 0; i < num_units; i++) {
        for (int j=0; j < num_units + num_external_inputs + 1; j++) {
            double *tmp = malloc(sizeof(double) * num_units);
            for (int k = 0; k < num_units; k++) {
                double summa = 0.0;
                if (i == k) {
                    summa += x[j];
                }
                /* We use l to keep with the paper but it can be confusing */
                for (int l = 0; l < num_units; l++) {
                    summa += n->w->d[k][l+num_external_inputs] * n->p[l][i][j];
                }

                summa *= d_sigmoid(net_out[k]);
                tmp[k] = summa;
            }
            /* Save new sensitivities for this i,j and all k */
            for (int k = 0; k < num_units; k++) {
                n->p[k][i][j] = tmp[k];
            }
            free(tmp);
        }
    }

    /* Update activations */
    for (int k = 0; k < num_units; k++ ) {
        n->activations[k] = sigmoid(net_out[k]);
    }
                
    free(net_out);
    free(x);
}


/* Calculate derivative with (32) */
void calc_grad_from_sensitivities( Network *n, double *target )
{

    /* For convenience */
    int num_units = n->num_units;
    int num_external_inputs = n->num_external_inputs;
   
    /* Get error signals */
    double *epsilons = malloc(sizeof(double) * num_units);
    for (int k=0; k < num_units; k++) {
        double e_k = target[k] - n->activations[k];
        epsilons[k] = e_k;
    }
    for (int i=0; i < num_units; i++) {
        for (int j=0; j < num_external_inputs + num_units + 1; j++) {
            double summa = 0.0;
            for (int k = 0; k < num_units; k++) {
                summa += epsilons[k] * n->p[k][i][j];
            }
            n->g->d[i][j] = summa;
        }
    }
    free(epsilons);
}


/* Print out a slice of sensitivities */
void print_slice_s(Network *n, int k)
{
    /* For convenience */
    int rows = n->num_units;
    int cols = n->num_external_inputs + n->num_units + 1;

    printf("Slice of network sensitivities through k = %d\n", k);
    for (int i=0; i < rows; i++) {
        for (int j=0; j < cols; j++) {
            printf("%g ", n->p[k][i][j]);
        }
        printf("\n");
    }
    printf("\n");

    return;
}


/* Print out a sensitivity */
void print_s_kij(Network *n, int k, int i, int j)
{
    /* For convenience */
    int rows = n->num_units;
    int cols = n->num_external_inputs + n->num_units + 1;

    printf("%g\n", n->p[k][i][j]);

    return;
}


/* Performs real time recurrent learning on the Network n. Returns the gradient
 * as a Matrix.
 */
Matrix *rtrl(Network *n,
             double **inputs,
             int input_len,
             double **targets,
             int *target_mask)
{
    /* For convenience */
    int num_units = n->num_units;
    int num_external_inputs = n->num_external_inputs;

    int rows = num_units;
    int cols = num_external_inputs + num_units + 1;

    Matrix *g = init_matrix(rows, cols); // Zero matrix

    for (int t=0; t < input_len; t++) {
        run(n,inputs[t]); // Network now stores output at t+1
        /* If we have a target, we calculate the gradient at this point */
        if (target_mask[t+1] == 1) {
            calc_grad_from_sensitivities(n, targets[t+1]);
            /* Update g (total gradient over input sequence */
            sum_m2_into_m1(g,n->g);
        }

        //print_s_kij(n,2,0,3); 
        //printf("%g\n", n->activations[0]);
    }

    return g;

}

/* Allocates memory needs to be freed
 *
 * Collect targets from network and input
 */
double **find_targets(Network *n, double **inputs, int input_len)
{
    /* One extra so that the times line up.
     * Recall inputs run from t0 to t1 - 1
     * outputs will be zero, then run from t0 + 1 to t1
     */
    double **outputs = calloc(input_len + 1, sizeof(double*));
    for (int t=0; t < input_len; t++) {
        run(n, inputs[t]);
        double *output = malloc(sizeof(double) * n->num_units);
        for (int k=0; k < n->num_units; k++) {
            output[k] = n->activations[k];
        }
        outputs[t+1] = output;
    }

    return outputs;
}


/* Allocates memory needs to be freed 
 *
 * Makes a target mask array of the right size such that all outputs from 
 * t0 + 1 to t1 are targets
 */
int *make_all_targets_mask(int epoch_len)
{
    int *mask = calloc(epoch_len + 1 , sizeof(int));
    for (int t=0; t < epoch_len; t++) {
        mask[t+1] = 1;
    }
    return mask;
}


/* Allocates memory needs to be freed 
 *
 * Makes a target mask array of the right size such that the only the 
 * output at t1 is a target.
 */
int *make_last_target_mask(int epoch_len)
{
    int *mask = calloc(epoch_len + 1 , sizeof(int));
    for (int t=0; t < epoch_len; t++) {
        mask[t] = 0;
    }
    mask[epoch_len] = 1;

    return mask;
}


/* Resets the activations of the network and the sensitivities */
void reset_network(Network *n)
{
    /* Reset activations */
    for (int k=0; k < n->num_units; k++) {
        n->activations[k] = 0.0;
    }

    /* Reset sensitivities */ 
    for (int k = 0; k < n->num_units; k++) {
        for (int i = 0; i < n->num_units; i++) {
            for (int j = 0; j < n->w->cols; j++) {
                n->p[k][i][j] = 0.0;
            }
        }
    }

    return;
}





int main()
{
    int num_units = 3;
    int num_external_inputs = 2;
    //int epoch_len = 20;
    int epoch_len = 50;

    /* TARGET NETWORK */
    double weights_correct [18] = {0.1, 0.2 , 0.01, -0.1 , 0.02, 1,
                                   0.2, 0.2 , 0.01,  0.11 -0.4,  1,
                                   0.3, 0.01, 0.02,  0.1 ,   1, -0.5};

    /* UPPER CORNER WAS 5 vs 1*/
    double weights[18] = {0.2, 0.2 , 0.01, -0.5 , 0.2, 5,
                          0.1, 0.2 , 0.01,  0.01 -0.4, 1,
                          0.7, 0.01, 0.02,  0.1 , 1,  -0.5};

    Network *tar_net = init_network(num_units, 
                                    num_external_inputs, 
                                    weights_correct);

    Network *n = init_network(num_units, 
                              num_external_inputs, 
                              weights);

    reset_network(n);

    /* Setup rng */
    alloc_rng();

    /* gen epoch */
    double **epoch = gen_epoch(epoch_len, num_external_inputs);

    double **targets = find_targets(tar_net, epoch, epoch_len);

    //int *tar_mask = make_all_targets_mask(epoch_len);
    int *tar_mask = make_last_target_mask(epoch_len);

    Matrix *g = rtrl(n, epoch, epoch_len, targets, tar_mask);

    crude_print_matrix(g);


    /* Free memory */
    for (int i=0; i < epoch_len; i++) {
        free(epoch[i]);
    }
    free(epoch);

    for (int i=0; i < epoch_len + 1; i++) {
        free(targets[i]);
    }
    free(targets);

    free(tar_mask);

    free_matrix(g);

    free_network(tar_net);
    free_network(n);

    free_rng();

    return 0;
}
