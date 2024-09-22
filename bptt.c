/*  Author: Michael Dunlap
 *    Date: November 2023
 * Purpose: Crude implementation of a single layer epochwise RNN and Back 
 *          Propagation Through Time based on the paper "Gradient-Based 
 *          Learning Algorithms for Recurrent Networks and Their 
 *          Computational Complexity" by Williams and Zisper. 
 *
 *          Main objective is educational. Simply write the method and verify it
 *          using the finite difference method. 
 */ 

#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "functions.h"

//#define SHOW_DECAY
//#define SHOW_ACTIVATIONS

/* Basic single layer RNN run epochwise. Stores the information necessary
 * to run Backpropagation Through Time (BPTT) on it to calculate the gradient.
 * That is, after running the network through an epoch of inputs, the network
 * outputs and activations will be stored in the struct for further processing.
 */
typedef struct network {

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

} Network;


/*
 * Initialize memory for Network. 
 * Weights will be read from *weights in a row by row manner.
 *   ie:
 *      double *weights = [1,2,3,4]
 *      yields
 *      [ [1,2],
 *        [3,4],
 *      ]
 *
 * Will init with zero vector for initial activations. 
 */
Network *init_network(int num_units,
                      int num_external_inputs,
                      int epoch_len,
                      double *weights)
{
    Network *n = malloc(sizeof(Network));

    n->num_units = num_units;
    n->num_external_inputs = num_external_inputs;
    n->epoch_len= epoch_len;

    int cols = num_units + num_external_inputs + 1; // plus 1 for bias
    int rows = num_units;

    Matrix *w = init_matrix(rows, cols);
    row_stream_set(w, weights);
    n->w = w;

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

    free(n);

    return;
}


/* Re-set the stored activations of the network */
void reset_network(Network *n)
{
    for (int i=0; i < n->num_units; i++) {
        n->activations[i] = 0.0;
    }
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

        /* Save net_out NOTE time is considered +1
         * ALSO save activations to hist
         * AND save current activations */
        for (int i = 0; i < num_units; i++) {
            n->net_out_hist[t+1][i] = net_out[i];
            double a_i = sigmoid(net_out[i]);
            n->activations[i] = a_i;
            n->activation_hist[t+1][i] = a_i;
        }

        /* XXX */
        //printf("At t = %d\n",t);
#ifdef SHOW_ACTIVATIONS
        print_row_vector(n->activations, n->num_units);
#endif

        free(net_out);
    }

    free(input);

    return;
}


/* EXPERIMENT: modification of run_network, perturbs a weight on the first
 *             input, or last, depending on the settings...
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

        /* IF pert_step, perturb the inputs 
         * THEN SET THEM BACK */
        double w_ij = 0.0;
        if (t == pert_step) {
            w_ij = n->w->d[a][b];
            n->w->d[a][b] = w_ij + perturbation;
            //crude_print_matrix(n->w);
        }
        /* Calculate network output */
        double *net_out = matrix_vector_prod(n->w,input);

        /* IF pert_time reset weight to the original value */
        if (t == pert_step) {
            n->w->d[a][b] = w_ij;
        }

        /* Save net_out NOTE time is considered +1
         * ALSO save activations to hist
         * AND save current activations */
        for (int i = 0; i < num_units; i++) {
            n->net_out_hist[t+1][i] = net_out[i];
            double a_i = sigmoid(net_out[i]);
            n->activations[i] = a_i;
            n->activation_hist[t+1][i] = a_i;
        }

#ifdef SHOW_ACTIVATIONS
        print_row_vector(n->activations, n->num_units);
#endif

        free(net_out);
    }

    free(input);

    return;
}


/* Allocates memory that needs to be freed
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


/* Allocates memory that needs to be freed
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


/* Allocates memory that needs to be freed
 *
 * Calculates the gradient of network for one epoch using epochwise BPTT.
 * This is using the negative of the error function.
 *
 * NOTE: Requires that the network has already been run through the epoch
 *       and consequently has the activation_hist and net_out_hist set.
 */
Matrix *bptt(Network *n, double **inputs, double **targets, int *tar_mask)
{
    /* For convenience */
    int num_units = n->num_units;
    int epoch_len = n->epoch_len;
    int num_external_inputs = n->num_external_inputs;

    /* Matrix to hold gradient information */
    Matrix *g = init_matrix(n->w->rows, n->w->cols); // Zero matrix

    double *deltas = malloc(sizeof(double) * n->num_units);
    double *epsilons = malloc(sizeof(double) * n->num_units);

    int t = epoch_len; // Will decrement to 1

    /* calculate initial epsilon values using (17) AND
     * initial delta values using (18) */
    for (int k = 0; k < num_units; k++) {
        double e_k = targets[t][k] - n->activation_hist[t][k]; // (17)
        epsilons[k] = e_k;
        double delta_k = d_sigmoid(n->net_out_hist[t][k]) * e_k; // (18)
        deltas[k] = delta_k;
#ifdef SHOW_DECAY
        printf("%g ", deltas[k]); //XXX
#endif
    }
#ifdef SHOW_DECAY
    printf("\n"); //XXX
#endif

    
    /* Sum contributions to gradient (see (20))*/
    for (int i = 0; i < n->w->rows; i++) {
        for (int j = 0; j < n->w->cols; j++) {

            /* x_j is from external input */
            if (j >= 0 && j < num_external_inputs) {
                g->d[i][j] += deltas[i] * inputs[t-1][j];
                            }
            /* x_j is from recurrent feedback input */
            else if (j >= num_external_inputs && 
                     j < num_external_inputs + num_units) {
                g->d[i][j] += deltas[i] * n->activation_hist[t-1][j-num_external_inputs];
            }
            /* x_j is fixed input of one for bias */
            else {
                g->d[i][j] += deltas[i]; // Bias
            }
        }
    }
   
   
    /* Loop over (19), (18), (20) decrementing time from epoch_len - 1 to 1 */
    for (int t = epoch_len - 1; t > 0; t--) {
        /* update epsilons using (19) */
        for (int k = 0; k < num_units; k++) {
            double summa = 0;
            /* We will use the variable l to match (19) although it 
             * can be confusing to use as a variable (see GSL guidelines)*/
            for (int l = 0; l < num_units; l++) {
                /* Recall deltas and epsilons are at t+1
                 * NOTE: k is in U and so regarding the columns of the matrix
                 *       needs to be ranging from num_external_inputs to
                 *       num_external_inputs + num_units 
                 */ 
                summa += n->w->d[l][k+num_external_inputs] * deltas[l]; 
            }

            double e_k = 0.0;
            /* If we have a target we inject error otherwise we leave this
             * as 0 */
            if (tar_mask[t] == 1) {
                /* inject error from t */
                e_k = targets[t][k] - n->activation_hist[t][k];
            }

            /* combine with sum to finish (19) */
            e_k += summa;


            /* Save new epsilon values values */
            epsilons[k] = e_k;
        }

        /* Use (18) to update deltas */
        for (int k = 0; k < num_units; k++) {
            double delta_k = d_sigmoid(n->net_out_hist[t][k]) * epsilons[k];
            deltas[k] = delta_k;
#ifdef SHOW_DECAY
            printf("%g ", deltas[k]); //XXX
#endif
        }
#ifdef SHOW_DECAY
        printf("\n"); //XXX
#endif


        /* Sum contributions to gradient (see (20))*/
        for (int i = 0; i < n->w->rows; i++) {
            for (int j=0; j < n->w->cols; j++) {
                /* x_j is from external input */
                if (j >= 0 && j < num_external_inputs) {
                    g->d[i][j] += deltas[i] * inputs[t-1][j];
                    
                }
                /* x_j is from recurrent feedback */
                else if (j >= num_external_inputs && 
                         j < num_external_inputs + num_units) {
                    g->d[i][j] += deltas[i] * n->activation_hist[t-1][j-num_external_inputs];
                }
                /* x_j is 1 for bias */
                else {
                    g->d[i][j] += deltas[i]; // Bias
                }
            }
        }
    }

    free(deltas);
    free(epsilons);
    

    return g;
}


/* Negative of error at time t. The variable t is not explicitly handed to 
 * this function, it is implicit in where *result and *target came from
 */
double J(double *result, double *target, int num_units)
{
    double summa = 0;
    for (int i = 0; i < num_units; i++) {
        double tmp = (target[i] - result[i]);
        tmp *= tmp;
        summa += tmp;
    }

    return (-1 * 0.5) * summa;
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



/* Basic tests of code */
int preliminary()
{
    int num_units = 3;
    int num_external_inputs = 2;
    //int epoch_len = 1;
    int epoch_len = 100;

    /* TARGET NETWORK */
    double weights_correct [18] = {0.1, 0.2 , 0.01, -0.1 , 0.02, 1,
                                   0.2, 0.2 , 0.01,  0.11 -0.4,  1,
                                   0.3, 0.01, 0.02,  0.1 ,   1, -0.5};

    double weights[18] = {0.2, 0.2 , 0.01, -0.5 , 0.2, 5,
                          0.1, 0.2 , 0.01,  0.01 -0.4, 1,
                          0.7, 0.01, 0.02,  0.1 , 1,  -0.5};

    Network *tar = init_network (num_units,
                                num_external_inputs,
                                epoch_len,
                                weights_correct);

    Network *n = init_network(num_units, 
                              num_external_inputs, 
                              epoch_len,
                              weights);

    
    Matrix *grad_bptt = init_matrix(num_units, num_units+num_external_inputs+1);
    Matrix *grad_diff = init_matrix(num_units, num_units+num_external_inputs+1);
    
    alloc_rng();
    int trials = 1;
    double dx = 1e-8;
    for (int i = 0; i < trials; i++) {

        /* gen epoch */
        double **epoch = gen_epoch(epoch_len, num_external_inputs);
        /* gen tar_mask */
        int *tar_mask = gen_end_only_tar_mask(epoch_len);
        //int *tar_mask = gen_all_target_mask(epoch_len);

        /* get targets */
        run_network(tar, epoch);
        
        /* get current outputs */
        printf("Start normal output\n"); //XXX
        run_network(n, epoch);
        printf("End normal output\n"); //XXX

        /* bptt */
        Matrix *g_bptt = bptt(n, epoch, tar->activation_hist, tar_mask);
        sum_m2_into_m1(grad_bptt, g_bptt);
        free_matrix(g_bptt);

        /* Get finite diff result */
        Matrix *g_fd = finite_difference(n, 
                                         epoch, 
                                         tar->activation_hist, 
                                         tar_mask,
                                         dx);

        sum_m2_into_m1(grad_diff,g_fd);
        free_matrix(g_fd);

        /* SIDE QUEST */
        printf("Output of network:\n");
        print_row_vector(n->activations, n->num_units);

        printf("START\n");
        perturbed_run_network(n, epoch,999999,1,1,3);
        printf("END\n");

        printf("Perturbed output:\n");
        print_row_vector(n->activations, n->num_units);
        

        
        for (int j = 0; j < epoch_len; j++) {
            free(epoch[j]);
        }
        free(epoch);
        free(tar_mask);
    }

    /* re-scale for total batch gradient */
    double c = (double) 1 / trials;

    mult_c_into_m(grad_bptt, c);
    mult_c_into_m(grad_diff, c);

    printf("bptt:\n");
    crude_print_matrix(grad_bptt);
    printf("\n");

    printf("finite difference:\n");
    crude_print_matrix(grad_diff);
    printf("\n");


        

    free_rng();

    free_network(n);
    free_network(tar);

    free_matrix(grad_bptt);
    free_matrix(grad_diff);


}


/**/
int main()
{

    alloc_rng();

    int num_units = 3;
    int num_external_inputs = 1;
    int epoch_len = 100;

    /* Experiment with an apples to apples RNN vs LSTM
     * hence we fix one of the recurrent self-connections to 1.0
     */
    /*
    double weights[15] = {0.1, 1 , 0.01, -0.1 , 0.02,
                          0.2, 0.2 , 0.01,  0.11 -0.4, 
                          0.3, 0.01, 0.02,  0.1 , 0.1,};
    */

    /* For uniformity let us use the same as in single_cell_lstm */
    double val = 0.9;
    double *weights = gen_weights(15 , val);
    

    Network *n = init_network (num_units,
                               num_external_inputs,
                               epoch_len,
                               weights);

    double fixed_recurrent = 0.5;

    n->w->d[0][1] = fixed_recurrent;

    crude_print_matrix(n->w);

    Network *n_perturbed = init_network(num_units, 
                                        num_external_inputs, 
                                        epoch_len,
                                        weights);

    n_perturbed->w->d[0][1] = fixed_recurrent;

    free(weights);

    /* gen epoch, (recall epoch is Gaussian N(0,1) */
    double **epoch = gen_epoch(epoch_len, num_external_inputs);


    run_network(n,epoch);

    int timestep = 3;
    perturbed_run_network(n_perturbed, // Network to perturb
                                epoch, // data to run through network
                               999999, // perturbation
                             timestep, // time step to apply perturbation
                                    0, // row of weight matrix to perturb
                                    1);// col of weight matrix to perturb

    /* Print the differences in chronological order */
    printf("Differences in network outputs:\n");
    for (int t = 0; t <= epoch_len; t++) {
        double diff = n->activation_hist[t][0] - n_perturbed->activation_hist[t][0];
        printf("%g\n", diff);
    }

    /* Analyze the decay rate of the perturbation */
    printf("Analyzing decay rate of perturbation...\n");
    int streak = 0;
    int done = 0;
    for (int t = epoch_len; t >= 0; t--) {
        double diff = n->activation_hist[t][0] - n_perturbed->activation_hist[t][0];
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
    free_network(n_perturbed);

    for (int t = 0; t < epoch_len; t++) {
        //printf("%f\n", epoch[t][0]);
        free(epoch[t]);
    }
    free(epoch);

    return 0;
}

