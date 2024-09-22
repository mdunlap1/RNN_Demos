/*  Author: Michael Dunlap 
 *    Date: "2024 01 17"
 * Purpose: Make a single cell LSTM network and study it.
 *          Network will use full back propagation through time. 
 *
 * TODO Put things like cell_idx and all in the struct itself 
 * TODO It was a bad idea to put the epoch_len in the network itself
 */ 


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "functions.h"

#define SHOW_DECAY 1
#define DONT_SHOW_DECAY 0
//#define SHOW_ACTIVATIONS
//#define SHOW_INTERNAL_STATE_CHANGES
//#define SQUASH_INTERNAL_STATE

/* Single cell LSTM network with only one recurrent layer and no output layer */
typedef struct network {

    double internal_state; // WARNING: this percludes multiple mem cells...

    int num_units; // Number of units in layer (only one layer)
    int num_external_inputs;
    int epoch_len; // Also the number of outputs...

    // Used to hold the weights
    Matrix *w;

    /* used to log the network output history (non-activated) */
    double **net_out_hist;

    /* used to log the activation history (redundant, used for clarity) */
    double **activation_hist;

    /* used to keep track of the current activations */
    double *activations;

    /* Used to keep track of the history of internal states */
    double *state_hist;

} Network;


/*
 * Allocates memory: will need to be freed.
 *
 * Initialize memory for single cell LSTM Network.
 * Set the weights randomly.
 * Preserves weight of 1 for self connection of memory cell
 *
 * Will init with zero vector for initial activations. 
 *
 * Row 0 is for memory cell
 * Row 1 if for input gate
 * Row 2 is for output gate
 *
 * Col 0 is for input
 * Cols 1-3 are for recurrent connections mem cell, input gate, output gate resp
 * Col 4 is for bias
 */
Network *init_network(int epoch_len)
{
    int num_units = 3; // input gate, output gate, memory cell
    int num_external_inputs = 1; // For simplicity of analysis
    /* Used to initialize the weights between +- w_limit
     * will need to re-set the self connection of memory cells to 1 */
    double w_limit = 0.9; // WORKS for 0,1 range of non-normalized inputs
    //double w_limit = 0.1;

    Network *n = malloc(sizeof(Network));

    n->num_units = num_units;     
    n->num_external_inputs = num_external_inputs; 
    n->epoch_len= epoch_len;
    n->internal_state = 0.0;

    int cols = num_units + num_external_inputs + 1; // plus 1 for bias
    int rows = num_units;

    Matrix *w = init_matrix(rows, cols);
    double *weights = gen_weights(rows*cols, w_limit); 
    row_stream_set(w, weights);
    n->w = w;
    free(weights);
    /* Set recurrent connection from memory cell to itself as 1 */
    n->w->d[0][1] = 1.0;
   

    /* more memory allocs here */

    /* The plus one since we are running from t = 0 to t = epoch_len */ 
    n->net_out_hist = malloc(sizeof(double*) * (epoch_len + 1));
    n->activation_hist = malloc(sizeof(double*) * (epoch_len + 1));
    for (int i = 0; i < epoch_len + 1; i++) {
        
        double *out = calloc(num_units, sizeof(double));
        n->net_out_hist[i] = out; 

        double *act = calloc(num_units, sizeof(double));
        n->activation_hist[i] = act;
    }

    n->activations = calloc(num_units, sizeof(double));

    n->state_hist  = calloc(epoch_len + 1, sizeof(double));

    return n;
}


/* Frees the internal memory allocated to the network as well as the network
 * struct itself.
 *
 * NOTE: Does NOT free the weights that were given to init_network.
 *       If weights were dynamically allocated and handed off to init_network,
 *       then that allocation needs to be freed as well.
 */
void free_network(Network *n)
{
    free_matrix(n->w);

    for (int i = 0; i < n->epoch_len + 1; i++) {
        free(n->net_out_hist[i]);
        free(n->activation_hist[i]);
    }

    free(n->activation_hist);
    free(n->net_out_hist);

    free(n->activations);
    free(n->state_hist);

    free(n);

    return;
}


/* Re-set the stored activations of the network */
void reset_network(Network *n)
{
    for (int i=0; i < n->num_units; i++) {
        n->activations[i] = 0.0;
    }

    n->internal_state = 0.0;
}


/* Runs the inputs from epoch through the network and stores the activations
 * and network outputs as it goes. 
 * Does NOT store the inputs themselves.
 * Resets the network prior to running but NOT after
 */
void run_network(Network *n, double **epoch)
{
    reset_network(n);

    /* For convenience */
    int num_external_inputs = n->num_external_inputs;
    int epoch_len = n->epoch_len;
    int num_units = n->num_units;
    int cell_idx = 0;
    int in_g_idx = 1;
    int out_g_idx = 2;

    double *input = malloc(sizeof(double)*(num_external_inputs + num_units + 1));
    input[num_units + num_external_inputs] = 1.0; // For the bias

    for (int t = 0; t < epoch_len; t++) {


        /* set external input */
        double *external_input = epoch[t];
        for (int idx = 0; idx < num_external_inputs; idx++) {
            input[idx] = external_input[idx];
        }


        /* set recurrent feedback input */
        for (int idx = 0; idx < num_units; idx++) {
            input[num_external_inputs + idx] = n->activations[idx];
        }


        /* NOTE bias was already set! */

        /* Calculate network output */
        double *net_out = matrix_vector_prod(n->w,input);

        /* From (6) of original LSTM paper */
        double n_in  = net_out[in_g_idx];
        double n_out = net_out[out_g_idx];
        double in_act = f_(n_in);
        double out_act = f_(n_out);

        /* Update internal state From (9) of original LSTM paper */
        n->internal_state = n->internal_state + in_act * g_(net_out[cell_idx]);

#ifdef SQUASH_INTERNAL_STATE
        n->internal_state = f_(n->internal_state);
#endif

        /* From (9) of original LSTM paper */
        double cell_act = out_act * h_(n->internal_state);

        /* Update current activations */
        n->activations[cell_idx] = cell_act;
        n->activations[in_g_idx] = in_act;
        n->activations[out_g_idx] = out_act;

#ifdef SHOW_ACTIVATIONS
        print_row_vector(n->activations, n->num_units);
#endif
#ifdef SHOW_INTERNAL_STATE_CHANGES
        //printf("%g %g\n", input[0], n->internal_state); // Magic number
        printf("%g\n", n->internal_state); 
#endif


        /* Save network outputs and activations to history */
        for (int i=0; i < num_units; i++) {
            n->net_out_hist[t+1][i] = net_out[i];
            n->activation_hist[t+1][i] = n->activations[i];
        }

        /* Save internal state to history */
        n->state_hist[t+1] = n->internal_state;

        free(net_out);
    }

    free(input);

    return;
}


/*
 * Perturbs the weight at row a column b at time step pert_step by adding
 * perturbation to it
 *
 * Runs the inputs from epoch through the network and stores the activations
 * and network outputs as it goes. 
 * Does NOT store the inputs themselves.
 * Resets the network prior to running but NOT after
 */
void perturbed_run_network(Network *n, 
                           double **epoch, 
                           double perturbation,
                           int pert_step,
                           int a,
                           int b
                           )
{
    reset_network(n);

    /* For convenience */
    int num_external_inputs = n->num_external_inputs;
    int epoch_len = n->epoch_len;
    int num_units = n->num_units;
    int cell_idx = 0;
    int in_g_idx = 1;
    int out_g_idx = 2;

    double *input = malloc(sizeof(double)*(num_external_inputs + num_units + 1));
    input[num_units + num_external_inputs] = 1.0; // For the bias

    for (int t = 0; t < epoch_len; t++) {

        /* IF pert_step, perturb the inputs 
         * THEN SET THEM BACK */
        double w_ij = 0.0;
        if (t == pert_step) {
            w_ij = n->w->d[a][b];
            n->w->d[a][b] = w_ij + perturbation;
            //crude_print_matrix(n->w);
        }


        /* set external input */
        double *external_input = epoch[t];
        for (int idx = 0; idx < num_external_inputs; idx++) {
            input[idx] = external_input[idx];
        }


        /* set recurrent feedback input */
        for (int idx = 0; idx < num_units; idx++) {
            input[num_external_inputs + idx] = n->activations[idx];
        }


        /* NOTE bias was already set! */

        /* Calculate network output */
        double *net_out = matrix_vector_prod(n->w,input);

        /* From (6) of original LSTM paper */
        double n_in  = net_out[in_g_idx];
        double n_out = net_out[out_g_idx];
        double in_act = f_(n_in);
        double out_act = f_(n_out);

        /* Update internal state From (9) of original LSTM paper */
        n->internal_state = n->internal_state + in_act * g_(net_out[cell_idx]);

/* XXX */
#ifdef SQUASH_INTERNAL_STATE
        n->internal_state = f_(n->internal_state);
#endif

        /* From (9) of original LSTM paper */
        double cell_act = out_act * h_(n->internal_state);

        /* Update current activations */
        n->activations[cell_idx] = cell_act;
        n->activations[in_g_idx] = in_act;
        n->activations[out_g_idx] = out_act;

#ifdef SHOW_ACTIVATIONS
        print_row_vector(n->activations, n->num_units);
#endif
#ifdef SHOW_INTERNAL_STATE_CHANGES
        //printf("%g %g\n", input[0], n->internal_state); //Magic number
        printf("%g\n",n->internal_state); //Magic number
#endif


        /* Save network outputs and activations to history */
        for (int i=0; i < num_units; i++) {
            n->net_out_hist[t+1][i] = net_out[i];
            n->activation_hist[t+1][i] = n->activations[i];
        }

        /* Save internal state to history */
        n->state_hist[t+1] = n->internal_state;

        free(net_out);

        /* Reset weight when done if at pert_step */
        /* IF pert_time reset weight to the original value */
        if (t == pert_step) {
            n->w->d[a][b] = w_ij;
        }
    }

    free(input);

    return;
}


/* 
 * Allocates memory that needs to be freed
 *
 * Create a target mask where only the last output is considered a target
 * 
 * TODO This is redundant as we ALWAYS assume (assert?) that there is 
 *      error to inject at the end of epoch, all zeros would be sufficient
 *      for our purposes here... (Better approach?)
 */
int *gen_end_only_tar_mask(int epoch_len)
{
    /* Recall that the last network activation will be at time t which is
     * where we are setting the target and this is at epoch_len */
    int *tar_mask = calloc(epoch_len + 1, sizeof(int));
    tar_mask[epoch_len] = 1;
    return tar_mask;
}


/* 
 * Allocates memory that needs to be freed
 *
 * Generates a target mask that identifies all output as targets
 */
int *gen_all_target_mask(int epoch_len)
{
    int *tar_mask = malloc(sizeof(int) * (epoch_len + 1));
    for (int i = 0; i <= epoch_len; i++) {
        tar_mask[i] = 1;
    }

    return tar_mask;
}


/* Allocates memory needs to be freed
 *
 * NORMALIZED
 * Generates a sequence of inputs for experiment 1
 *
 */
double **gen_norm_ex1_seq(int epoch_len)
{

    double **inputs = malloc(epoch_len * sizeof(double*)); 
    for (int i=0; i < epoch_len; i++) {
        double *input = calloc(1,sizeof(double));
        double x = uniform();
        x *= 0.5; // Force value to be less than 0.5
        if (x == 0.5) {
            printf("Assertion broken in gen_ex1_seq\nexiting...");
            exit(EXIT_FAILURE);
        }
        x = 2*x - 1;
        if (x >= 0.0) {
            printf("Assertion broken in gen_ex1_seq\nexiting...");
            exit(EXIT_FAILURE);
        }

        input[0] = x;
        inputs[i] = input;
    }

    double u = uniform();
    if (u >= 0.5) {
        int idx = rando(0,epoch_len - 1);
        u = u * 2 - 1;
        if (u < 0.0) {
            printf("Asserion broken in gen_norm_ex1_seq\nexiting...\n");
        }
        inputs[idx][0] = u;
    }

    return inputs;
}

/* Allocates memory needs to be freed
 *
 * NORMALIZED
 * Scans through the inputs from experiment 1, sets the target to 1
 * if a value greater than 0.5 has been encountered, 0 otherwise.
 *
 */
double **gen_norm_targets_ex_1(double **inputs, int epoch_len)
{
    /* We need epoch_len + 1 because we need the time indices to match */
    double **targets = malloc(sizeof(double*) * (epoch_len + 1));
    for (int i = 0; i < epoch_len + 1; i++) {
        double *target = calloc(sizeof(double), 1);
        targets[i] = target;
    }

    double tar = -1.0;
    targets[0][0] = tar; // Dummy target for time 0 (there is no target) 
    for (int t = 0; t < epoch_len; t++) {
        double input = inputs[t][0];
        if (input >= 0.0) {
            tar = 1.0;
        }
        targets[t + 1][0] = tar;
    }

    return targets;
}


/* Allocates memory needs to be freed
 *
 * Generates a sequence of inputs for experiment 1
 *
 */
double **gen_ex1_seq(int epoch_len)
{

    double **inputs = malloc(epoch_len * sizeof(double*)); 
    for (int i=0; i < epoch_len; i++) {
        double *input = calloc(1,sizeof(double));
        double x = uniform();
        x *= 0.5; // Force value to be less than 0.5
        if (x == 0.5) {
            printf("Assertion broken in gen_ex1_seq\nexiting...");
            exit(EXIT_FAILURE);
        }
        input[0] = x;
        inputs[i] = input;
    }

    double u = uniform();
    if (u >= 0.5) {
        int idx = rando(0,epoch_len - 1);
        inputs[idx][0] = u;
    }

    return inputs;
}


/* Allocates memory needs to be freed
 *
 * Scans through the inputs from experiment 1, sets the target to 1
 * if a value greater than 0.5 has been encountered, 0 otherwise.
 *
 */
double **gen_targets_ex_1(double **inputs, int epoch_len)
{
    /* We need epoch_len + 1 because we need the time indices to match */
    double **targets = malloc(sizeof(double*) * (epoch_len + 1));
    for (int i = 0; i < epoch_len + 1; i++) {
        double *target = calloc(sizeof(double), 1);
        targets[i] = target;
    }

    double tar = 0.0;
    targets[0][0] = tar; // Dummy target for time 0 (there is no target) 
    for (int t = 0; t < epoch_len; t++) {
        double input = inputs[t][0];
        if (input > 0.5) {
            tar = 1.0;
        }
        targets[t + 1][0] = tar;
    }

    return targets;
}


/*
 * Allocates memory that needs to be freed
 *
 * Full epoch-wise BPTT for a single cell original LSTM
 *
 * Calculates the gradient of network for one epoch using epochwise BPTT.
 * This is using the negative of the error function.
 *
 * NOTE: Requires that the network has already been run through the epoch
 *       and consequently has the activation_hist and net_out_hist set.
 */
Matrix *full_bptt(Network *n, 
                  double **inputs, 
                  double **targets, 
                  int *tar_mask,
                  int show_decay)
{
    /* For convenience */
    int num_units = n->num_units;
    int epoch_len = n->epoch_len;
    int num_external_inputs = n->num_external_inputs;
    int cell_idx = 0;
    int in_g_idx = 1;
    int out_g_idx = 2;
    /* For the weight matrix */
    int cell_col = cell_idx + 1;
    int in_col   = in_g_idx + 1;
    int out_col  = out_g_idx + 1;

    /* Matrix to hold gradient information */
    Matrix *g = init_matrix(n->w->rows, n->w->cols); // Zero matrix

    double *deltas   = malloc(sizeof(double) * num_units);
    double *epsilons = malloc(sizeof(double) * num_units);
    double epsilon = 0.0; 

    int t = epoch_len; // Will decrement to 1

    /* Calculate initial epsilon values */
    epsilon = n->activation_hist[t][cell_idx] - targets[t][cell_idx];
    
    /* dE/dy^cell(t) */
    epsilons[cell_idx] = epsilon; 
    /* dE/dy^out(t) */
    epsilons[out_g_idx] = epsilon * h_(n->state_hist[t]); 
    /* dE/dy^in(t) */
    epsilons[in_g_idx] = epsilon * n->activation_hist[t][out_g_idx] * d_h(n->state_hist[t]) * g_(n->net_out_hist[t][cell_idx]);

    /* Save reference to dE/dS(t) */
    double dE_dS = epsilon * 
                   n->activation_hist[t][out_g_idx] * 
                   d_h(n->state_hist[t]);

    /* Calculate initial delta values */

    /* Output gate delta */
    deltas[out_g_idx] = epsilons[out_g_idx] * d_f(n->net_out_hist[t][out_g_idx]);

    /* Input gate delta */
    deltas[in_g_idx] = epsilons[in_g_idx] * d_f(n->net_out_hist[t][in_g_idx]);

    /* Cell delta */
    deltas[cell_idx] = epsilons[cell_idx] * n->activation_hist[t][out_g_idx]*d_h(n->state_hist[t]) *
        n->activation_hist[t][in_g_idx] * d_g(n->net_out_hist[t][cell_idx]);

    double summa = 0;
    for (int k = 0; k < num_units; k++) {
        summa += n->w->d[k][cell_col] * deltas[k];
    }


    /* For input from timestep t-1 */
    double *input_t_1 = calloc(num_units + num_external_inputs + 1, sizeof(double));
    input_t_1[num_units + num_external_inputs] = 1.0; // For the bias

    /* load input */

    /* set external input */
    for (int idx = 0; idx < num_external_inputs; idx++) {
        input_t_1[idx] = inputs[t-1][idx];
    }

    /* set recurrent feedback input */
    for (int idx = 0; idx < num_units; idx++) {
        input_t_1[num_external_inputs + idx] = n->activation_hist[t-1][idx];
    }

    /* NOTE bias was already set! */


    /* Sum contributions to gradient */
    for (int i = 0; i < n->w->rows; i++) {
        for (int j = 0; j < n->w->cols; j++) {
            if (i == 0 && j==0) {
                //printf("summand: %.30f\n", deltas[i] * input_t_1[j]);
            }
            g->d[i][j] += deltas[i] * input_t_1[j];
            if ( i == 0 && j == 0) {
                //printf("grad:    %.30f\n", g->d[i][j]);
            }
            
        }
    }

    if (show_decay) {
        print_row_vector(deltas, num_units);
    }

    while (t > 1) {
        t--;

        /* Update epsilons */
        for (int unit = 0; unit < num_units; unit++) { 
            double summa = 0;
            for (int l = 0; l < num_units; l++) {
                /* Recall that the column for the unit in the weight matrix is plus one */
                summa += n->w->d[l][unit + num_external_inputs] * deltas[l];
            }
            epsilons[unit] = summa;
        }

        /* Update deltas */

        /* Output gate delta */
        deltas[out_g_idx] = epsilons[out_g_idx] * 
                            d_f(n->net_out_hist[t][out_g_idx]);
        deltas[out_g_idx] += epsilons[cell_idx] * 
                             d_f(n->net_out_hist[t][out_g_idx]) * 
                             h_(n->state_hist[t]);


        /* Input gate delta */
        deltas[in_g_idx] = epsilons[in_g_idx] * 
                           d_f(n->net_out_hist[t][in_g_idx]);

        deltas[in_g_idx] += epsilons[cell_idx] * 
                            n->activation_hist[t][out_g_idx] * 
                            d_h(n->state_hist[t]) * 
                            g_(n->net_out_hist[t][cell_idx]) * 
                            d_f(n->net_out_hist[t][in_g_idx]);

        deltas[in_g_idx] += dE_dS * 
                            g_(n->net_out_hist[t][cell_idx]) *
                            d_f(n->net_out_hist[t][in_g_idx]);


        /* Cell delta */
        deltas[cell_idx] = epsilons[cell_idx] * 
                           n->activation_hist[t][out_g_idx]*
                           d_h(n->state_hist[t]) *
                           n->activation_hist[t][in_g_idx] * 
                           d_g(n->net_out_hist[t][cell_idx]);

        deltas[cell_idx] += dE_dS * 
                            n->activation_hist[t][in_g_idx] *
                            d_g(n->net_out_hist[t][cell_idx]);


        dE_dS += epsilons[cell_idx] *
                 n->activation_hist[t][out_g_idx] *
                 d_h(n->state_hist[t]);

        if (show_decay) {
            print_row_vector(deltas, num_units);
        }

        /* load input */

        /* set external input */
        for (int idx = 0; idx < num_external_inputs; idx++) {
            input_t_1[idx] = inputs[t-1][idx];
        }

        /* set recurrent feedback input */
        for (int idx = 0; idx < num_units; idx++) {
            input_t_1[num_external_inputs + idx] = n->activation_hist[t-1][idx];
        }

        /* NOTE bias was already set! */


        /* Sum BASIC contributions to gradient */
        for (int i = 0; i < n->w->rows; i++) {
            for (int j = 0; j < n->w->cols; j++) {
                if (i == 0 && j==0) {
                    //printf("summand: %.30f\n", deltas[i] * input_t_1[j]);
                }
                g->d[i][j] += deltas[i] * input_t_1[j];
                
                if ( i == 0 && j == 0) {
                    //printf("grad:    %.30f\n", g->d[i][j]);
                }
            }
        }

    }

    free(deltas);
    free(epsilons);
    free(input_t_1);
    
    return g;
}


/* Negative of error at time t. The variable t is not explicitly handed to 
 * this function, it is implicit in where *result and *target came from
 */
double J(double *result, double *target, int num_units)
{
    /* XXX BUG (crude fix will be applied (instead of having targets for all 
     * units, we have only the one... */
    /*
    double summa = 0;
    for (int i = 0; i < num_units; i++) {
        double tmp = (target[i] - result[i]);
        tmp *= tmp;
        summa += tmp;
    }
    */
    double summa = target[0] - result[0];
    summa *= summa;


    return (0.5) * summa;
}


/* Runs through the calculation for J_total using J */
double J_total(double **results, 
               double **targets, 
                  int *tar_mask,
                  int epoch_len, 
                  int num_units)
{
    double summa = 0.0;
    for (int t = 1; t <= epoch_len; t++) {
        if (tar_mask[t] == 1) {
            double *result = results[t];
            double *target = targets[t];
            summa += J(result, target, num_units);
        }
    }

    return summa;
}


/* Calculates gradient by method of finite differences */
Matrix *finite_difference(Network *n,
                          double **epoch,
                          double **targets,
                             int *tar_mask,
                          double dx)
{
    /* For convenience */
    int num_units = n->num_units;
    int epoch_len = n->epoch_len;
    int num_external_inputs = n->num_external_inputs;

    /* Will be used to save current value of w_ij */
    double w_ij;

    /* Matrix to hold gradient information */
    Matrix *g = init_matrix(n->w->rows, n->w->cols); // Zero matrix

    /* Used to store numerator for difference */
    double J_plus  = 0.0;
    double J_minus = 0.0;

    for (int i = 0; i < n->w->rows; i++) {
        for (int j=0; j<n->w->cols;j++) {
            w_ij = n->w->d[i][j]; // Save value here

            n->w->d[i][j] = w_ij + dx; // Increment UP
            run_network(n, epoch);
            J_plus = J_total(n->activation_hist, 
                             targets,
                             tar_mask,
                             epoch_len,
                             num_units);

            n->w->d[i][j] = w_ij - dx; // Increment DOWN 
            run_network(n, epoch);
            J_minus = J_total(n->activation_hist, 
                              targets,
                              tar_mask,
                              epoch_len,
                              num_units);


            /* Find finite difference */
            
            double fd = (J_plus - J_minus) / (2*dx); 
            g->d[i][j] = fd;
            
            /* Reset w_ij */
            n->w->d[i][j] = w_ij;

        }
    }

    return g;

}


/* Determines whether or not the network output is "correct" or not.
 * Determination is based on the criteria outlined in the LSTM paper.
 * From page 20: "error > 0.3 for at least one output unit"
 *
 * MODIFIED for the single cell 
 *
 * Returns 1 if correct, 0 otherwise. 
 */
int is_correct(Network *n, double *target)
{
    double threhold = 0.3;

    double d = target[0] - n->activations[0];
    if (d < 0.0) {
        d *= -1;
    }

    if (d < 0.0) {
        fprintf(stderr, "is_correct is producing negative error\n");
        fprintf(stderr, "exiting...\n");
        exit(EXIT_FAILURE);
    }

    if (d > threhold) {
        return 0;
    }

    return 1;
}


/* Update the nework weights via Stochastic Gradient Descent using the
 * specified alpha
 */
void update(Network *n, Matrix *g, double alpha) 
{
    for (int row=0; row < n->w->rows; row++) {
        for (int col=0; col < n->w->cols; col++) {
            /* Recurrent self connection */
            if (row == 0 && col == 1) {
                continue;
            }
            n->w->d[row][col] -= alpha * g->d[row][col];
        }
    }
}


/* Train the network on experiment one */
int train(Network *n) {

    n->w->d[2][4] = -2.0;

    int epoch_len = n->epoch_len;

    int streak = 0;
    int tar_streak = 100;

    double err_cutoff = 0.1;
    double *errors = calloc(tar_streak, sizeof(double));

    double window_error = 0.0;
    for (int i=0; i < tar_streak; i++) {
        window_error += errors[i];
    }
    window_error /= tar_streak;

    int *tar_mask = gen_end_only_tar_mask(epoch_len);

    //crude_print_matrix(n->w); 
    //printf("\n");

    int trials = 0;
    while (streak < tar_streak || window_error > err_cutoff) {

        double **inputs = gen_norm_ex1_seq(epoch_len);
        double **targets = gen_norm_targets_ex_1(inputs, epoch_len);
               
        run_network(n, inputs);
        Matrix *g = full_bptt(n, inputs, targets, tar_mask, DONT_SHOW_DECAY);

        update(n,g,0.5);

        int correct = is_correct(n, targets[epoch_len]);
        if (correct) {
            streak++;
        }
        else {
            streak = 0;
        }


        double error = n->activations[0] - targets[epoch_len][0];
        error *= 0.5 * error;
        errors[trials % tar_streak] = error;

        /* Update error for streak window */
        window_error = 0.0;
        for (int i=0; i < tar_streak; i++) {
            window_error += errors[i];
        }
        window_error /= tar_streak;
        /* Update trials count */
        trials++;
        //XXX
        /*
        if (trials == 50) {
            exit(1);
        }
        */


        if (trials % 1000 == 0) {
            printf("trial %d\n", trials);
            crude_print_matrix(n->w);
            printf("%d %g\n", streak, window_error);
            printf("%g\n", n->internal_state);
            printf("\n");

        }
                

        free_matrix(g);
        for (int i=0; i < epoch_len; i++) {
            free(inputs[i]);
        }
        free(inputs);
        for (int i = 0; i < epoch_len + 1; i++) {
            free(targets[i]);
        }
        free(targets);


    }

    printf("Obtained in %d trials\n", trials);

    free(errors);
    free(tar_mask);

    return 0;
}


/* Write the network weights to a file */
void record_weights(Network *n, char *filename)
{
    FILE *fp = fopen(filename, "w");
    for (int i=0; i < n->w->rows; i++) {
        for (int j=0; j < n->w->cols; j++) {
            fprintf(fp, "%f ", n->w->d[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    return;

}


/* Load the weights found in filename to the network n */
void load_weight(Network *n, char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Failed to open file in load_weight\nexiting...\n");
        exit(EXIT_FAILURE);
    }

    /* Get a count of how large the array needs to be */
    int count = 0;
    double x;
    while ( (fscanf(fp, "%lf", &x)) != EOF) {
        count++;
    }

    int num_weights = n->w->rows * n->w->cols;

    if (num_weights != count) {
        printf("Failed to load network from file.\n"
                "Network size and weights from file are mismatched\n"
                "exiting...\n");
        exit(EXIT_FAILURE);
    }

    rewind(fp);

    double *array = malloc(count * sizeof(double));

    /* Load the array */
    int idx = 0;
    while ( (fscanf(fp, "%lf", &x)) != EOF) {
        array[idx++] = x;
    }

    /* Set the values */
    row_stream_set(n->w, array);

    free(array);

    fclose(fp);

}


/* Run the network through tests for experiment 1 */
void test_ex1(Network *n, int trials)
{
    int epoch_len = n->epoch_len;

    int *tar_mask = gen_end_only_tar_mask(epoch_len);


    int num_success = 0;
    double total_error = 0;
    int test_num = 0;
    while (test_num < trials) {

        test_num++;
        //printf("Test %d\n", test_num);

        double **inputs = gen_ex1_seq(epoch_len);
        double **targets = gen_targets_ex_1(inputs, epoch_len);
               
        run_network(n, inputs);

        int correct = is_correct(n, targets[epoch_len]);
        if (correct) {
            num_success++;
            
        }

        double error = n->activations[0] - targets[epoch_len][0];
        error *= 0.5 * error;
        total_error += error;

        /*
        if (!correct && targets[epoch_len][0] == 1) {
            printf("%f\n", n->activations[0]);
        }
        */


        for (int i=0; i < epoch_len; i++) {
            free(inputs[i]);
        }
        free(inputs);
        
        for (int i = 0; i < epoch_len + 1; i++) {
            free(targets[i]);
        }
        free(targets);


    }

    double success_rate = (double) num_success / trials;
    double avg_error    = total_error / trials;

    printf("Success rate: %g\n"
            "Average error: %g\n", success_rate, avg_error);


    free(tar_mask);

}

int preliminary_test_against_finite_difference()
{
    int epoch_len = 20;
    //int epoch_len = 100; // For experiment 1

    alloc_rng();
    
    Network *n = init_network(epoch_len);

    double **inputs = gen_ex1_seq(epoch_len);
    double **targets = gen_targets_ex_1(inputs, epoch_len);
    int *tar_mask = gen_end_only_tar_mask(epoch_len);

    //train(n);

    //record_weights(n, "__one_cell_weights.txt");
   
    /*
    printf("Loading weights...\n");
    load_weight(n, "__one_cell_weights.txt");
    printf("Loaded!\n");
    reset_seed(39977651);
    test_ex1(n, 100);
    */
    

    
    run_network(n, inputs);
    
    /*
    perturbed_run_network(n, 
                          inputs, 
                          1000,
                          1,
                          0,
                          1);
                          */

    /* Finite difference test */
     
    Matrix *g = full_bptt(n, inputs, targets, tar_mask, DONT_SHOW_DECAY);
    Matrix *actual = finite_difference(n, inputs, targets, tar_mask, 1e-8);

    printf("bptt:\n");
    crude_print_matrix(g);
    printf("\n");

    
    printf("finite difference:\n");
    crude_print_matrix(actual);
    printf("\n");
    /* END Finite difference test */
    

    for (int i = 0; i < epoch_len; i++) {
        free(inputs[i]);
        free(targets[i]);
    }
    free(targets[epoch_len]);

    free(inputs);
    free(targets);
    free(tar_mask);

    free_rng();

    free_network(n);
    free_matrix(g);
    free_matrix(actual);

}


/* Basic experiment to see how a perturbation to the network decays over time */
int analyze_perturbation()
{
    alloc_rng();

    int epoch_len = 200; // For experiment 1

    Network *n = init_network(epoch_len);

    double **inputs = gen_epoch(epoch_len, 1); // Gaussian noise input

#ifdef SHOW_INTERNAL_STATE_CHANGES
    printf("Network internal states\n"); 
#endif
    run_network(n,inputs);

    //crude_print_matrix(n->w);

    int cell_idx = 0;
    double *activations = malloc( (epoch_len + 1) * sizeof(double) );
    for (int t=0; t <= epoch_len; t++) {
        activations[t] = n->activation_hist[t][cell_idx];
    }

    int timestep = 3;
#ifdef SHOW_INTERNAL_STATE_CHANGES
    printf("Perturbed network internal states\n");
#endif
    perturbed_run_network(n,
                          inputs,
                          999999,
                          timestep,
                          0,
                          1);

    /* Print the differences of the outputs... */
    printf("Differences between network activations and perturbed network activations:\n");
    for (int t = 0; t <= epoch_len; t++) {
        double diff = activations[t] - n->activation_hist[t][cell_idx];
        printf("%g\n",diff);
    }


    printf("Analyzing decay rate of perturbation...\n");
    int streak = 0;
    int done = 0;
    for (int t = epoch_len; t >= 0; t--) {
        double diff = activations[t] - n->activation_hist[t][cell_idx];
        if (activations[t] >= 1.0 || n->activation_hist[t][cell_idx] >= 1.0) {
            printf("ERROR: Broke the range of sigmoidal activation function\n");
            exit(EXIT_FAILURE);
        }
        if (done == 0 && diff == 0.0) {
            streak++;
        }
        else {
            done = 1;
        }
    }

    printf("Decays in: %d steps\n", epoch_len - timestep - streak);


    free_rng();
    free_network(n);
    free(activations);
    for (int t=0; t < epoch_len; t++) {
        //printf("%f\n", inputs[t][0]);
        free(inputs[t]);
    }
    free(inputs);

    return 0;
}


/* Initiliazes two random single cell networks.
 * One will be trated as the target (gives correct behavior)
 * the other one will have back propagation through time performed on it
 * (comparing the output against the target network). 
 * The decay of the error signal (so-called "vanishing of the gradient")
 * will be printed out
 */
int show_decay_of_error_signal()
{
    alloc_rng();

    int epoch_len = 100; // For experiment 1

    
    Network *n = init_network(epoch_len);

    double **inputs = gen_epoch(epoch_len, 1); // Gaussian noise input

    run_network(n,inputs);

    /* Make target network and run on same inputs */
    Network *t = init_network(epoch_len);
    run_network(t, inputs);

    int *tar_mask = gen_end_only_tar_mask(epoch_len);

    printf("Will print out delta values in the following order:\n"
           "  cell input_gate output_gate.\n"
           "Each row corresponds to a time step in BPTT\n"
           "The closer to the last row, the further back in time we are in\n"
           "the algorithm.\n");

    /* Calculate the gradient of the Network n against the target Network
     * t using back propagation through time. */
    Matrix *grad = full_bptt(n,
                             inputs,
                             t->activation_hist,
                             tar_mask,
                             SHOW_DECAY);
    
    free_rng();
    free_network(n);
    free_network(t);
    free(tar_mask);
    free_matrix(grad);
    for (int t=0; t < epoch_len; t++) {
        //printf("%f\n", inputs[t][0]);
        free(inputs[t]);
    }
    free(inputs);


    return 0;
}


int main(int argc, char *argv[]) 
{
    if (argc == 1) {
        printf("Program to study a single memory cell LSTM network.\n");
        printf("Useage:"
               "\n-p"
               "   Analyze perturbations\n"
               "\n-d"
               "   Show the decay of the error signal\n"
               "\n-c"
               "   Check full bptt against finite difference method\n");
    
    }

    if (argc > 1 && strcmp(argv[1], "-p") == 0) {
        analyze_perturbation();
    }

    else if (argc > 1 && strcmp(argv[1], "-d") == 0) {
        show_decay_of_error_signal();
    }

    else if (argc > 1 && strcmp(argv[1], "-c") == 0) {
        preliminary_test_against_finite_difference();
    }

    return 0;

}
