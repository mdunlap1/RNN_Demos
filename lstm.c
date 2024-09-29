/*  Author: Michael Dunlap
 * Purpose: Educational study of the paper Long Short-term Memory by Sepp 
 *          Hochreiter and Jürgen Schmidhuber.
 *          In particular create a network like the one used for task 6a from 
 *          the paper. The code is not meant to be highly performative, yet
 *          aims to be clear and easy to read in the context of the paper. 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include "functions.h"
#include "matrix.h"


/* Holds the data for a hidden layer of an LSTM network. 
 *
 * Weight matrix should have the following structure:
 *
 * Rows go memory block by memory block and cell by cell so for 2 memory blocks
 * each with 2 cells the first 4 rows should correspond to c11, c12, c21, c22; 
 * where cmn is the mth block nth memory cell
 *
 * then all input gates (in increasing order),
 *
 * then all output gates (in increasing order).
 *
 * So for the above example we would have in1, in2, out1, out2.
 *
 * Columns start with the external input, then follow the same pattern as rows
 * for recurrent inputs with an additional column for the bias. We will include
 * a fixed value of 1 for all input fed to the weights of the network. 
 *
 */
typedef struct h_layer {

    int num_external_inputs;
    int num_units;
    int num_weights; /* For convenience */

    int num_blocks; /* Number of memory blocks */
    int block_size; /* Number of memory cells per block */
    int num_cells;  /* Number of total memory cells */


    /* Dimensions will be num_units by num_external_inputs + num_units + 1 */
    Matrix *w; /* Hidden Layer Weights */

    /* Buffers for convenience and calculation of gradient and sensitivities */
    double *input_buff;      /* Stores most recent inputs / previous activations*/
    double *net_out_buff;    /* Stores the most recent net_out */
    double *activations;     /* Stores the most recent activations */
    double *internal_states; /* For the internal states of the memory cells */

    double **s_g; /* Sensitivities with respect to connections to input gates */
    double **s_c; /* Sensitivities for weights connecting to memory cells */


} H_layer;


/* Holds the data for an output layer of an LSTM network */
typedef struct o_layer {

    int num_h_inputs; /* From the hidden layer */
    int num_units;
    int num_weights;

    /* num_units by num_inputs + 1 */
    Matrix *w; /* Output layer weights */

    /* Buffers for convenience and calculation of gradient and sensitivities */
    double *net_out_buff; /* Stores the most recent net_out */
    double *activations;  /* Stores the most recent activations */
    double *input_buff;   /* Stores the input */

} O_layer;


/* Holds the data of an LSTM network */
typedef struct network {
    H_layer *hid; /* Hidden layer  */
    O_layer *out; /* Output layer  */
    double alpha; /* Learning rate */

} Network;


/* Used to store the gradient of the network as two matrices */
typedef struct gradient {
    Matrix *hW; /* Hidden layer portion of gradient */
    Matrix *oW; /* Output layer portion of gradient */
} Gradient;


/* Holds a sequence of inputs, target value and other essential information to
 * be able to run the sequence through the network. Can also be used to run a
 * sequence without a target
 */
typedef struct sequence {

    int seq_len;     /* Length of sequence of inputs */
    int input_len;   /* Size of each input */ 
    int max_seq_len; /* Maximum size sequence can be: Used to reduce mem alloc*/
    double **seq;    /* Sequence data */

    int tar_len;     /* Size of target */
    double *tar;     /* Target */

} Sequence;
    

/* Allocates memory needs to be freed
 *
 * Allocates the memory for a Gradient struct.
 * Defaults the gradient matrices to zero matrices. 
 */
Gradient *init_gradient(Network *n)
{
    Gradient *grad = malloc(sizeof(Gradient));

    int h_rows = n->hid->num_units;
    int h_cols = n->hid->num_external_inputs + n->hid->num_units + 1;

    Matrix *hW = init_matrix(h_rows, h_cols); // Zero matrix

    int o_rows = n->out->num_units;
    int o_cols = n->out->num_h_inputs + 1;

    Matrix *oW = init_matrix(o_rows, o_cols); // Zero matrix

    /* Put Matrices in Gradient struct */
    grad->oW = oW;
    grad->hW = hW;

    return grad;
}


/* Frees the memory allocated to a Gradient struct and the struct itself */
void free_gradient(Gradient *grad)
{
    free_matrix(grad->oW);
    free_matrix(grad->hW);
    free(grad);

    return;
}


/* Allocates memory needs to be freed
 *
 * Allocates memory for an H_layer 
 */
H_layer *init_h_layer(int num_external_inputs,
                       int num_units,
                       int num_blocks,
                       int block_size)
{
    /* Allocate struct */
    H_layer *hid = malloc(sizeof(H_layer));

    /* Set fields */
    hid->num_external_inputs = num_external_inputs;
    hid->num_units  = num_units;
    hid->num_blocks = num_blocks;
    hid->block_size = block_size;
    hid->num_cells  = block_size * num_blocks;

    /* For convenience */
    int rows = num_units;
    int cols = num_external_inputs + num_units + 1; // 1 for bias
    int num_cells = num_blocks * block_size;

    Matrix *w = init_matrix(rows, cols);
    hid->w = w;

    double *input_buff      = calloc(cols, sizeof(double));
    double *net_out_buff    = calloc(rows, sizeof(double));
    double *activations     = calloc(rows, sizeof(double));
    double *internal_states = calloc(num_blocks * block_size, sizeof(double));

    input_buff[cols - 1] = 1.0; // Bias

    hid->input_buff      = input_buff;
    hid->net_out_buff    = net_out_buff;
    hid->activations     = activations;
    hid->internal_states = internal_states;


    double **s_g = malloc(sizeof(double*) * num_cells);
    double **s_c = malloc(sizeof(double*) * num_cells);
    int total_input_len = num_external_inputs + num_units + 1;
    for (int i=0; i < num_cells; i++) {
        double *s_g_row = calloc(total_input_len, sizeof(double));
        double *s_c_row = calloc(total_input_len, sizeof(double));
        s_g[i] = s_g_row;
        s_c[i] = s_c_row;
    }

    hid->s_g = s_g;
    hid->s_c = s_c;
    
    /* Calculate total number of weights (for convenience)*/
    hid->num_weights = rows * cols;

    return hid;
}


/* Frees the memory allocated to the hidden layer and the struct itself */
void free_h_layer(H_layer *hid)
{
    free_matrix(hid->w);
    free(hid->input_buff);
    free(hid->net_out_buff);
    free(hid->activations);
    free(hid->internal_states);

    /* Free sensitivities */
    int num_cells = hid->block_size * hid->num_blocks;
    for (int i=0; i < num_cells; i++) {
        free(hid->s_g[i]);
        free(hid->s_c[i]);
    }
    free(hid->s_g);
    free(hid->s_c);

    free(hid);

    return;
}


/* Allocates memory needs to be freed 
 *
 * Allocates the memory for an output layer O_layer struct 
 * and its internal data
 */
O_layer *init_o_layer(int num_h_inputs, int num_units)
{
    /* Allocate the struct */
    O_layer *out = malloc(sizeof(O_layer));

    out->num_h_inputs = num_h_inputs;
    out->num_units = num_units;

    int rows = num_units;
    int cols = num_h_inputs + 1;

    Matrix *w = init_matrix(rows,cols);
    out->w = w;

    double *net_out_buff = calloc(rows, sizeof(double));
    double *activations  = calloc(rows, sizeof(double));
    double *input_buff   = calloc(cols, sizeof(double));
    input_buff[num_h_inputs] = 1.0; // Bias

    out->net_out_buff = net_out_buff;
    out->activations  = activations;
    out->input_buff   = input_buff;

    /* For convenience */
    out->num_weights = rows * cols;

    return out;
}


/* Frees the memory allocated to an output layer and the struct itself */
void free_o_layer(O_layer *out)
{
    free_matrix(out->w);
    free(out->net_out_buff);
    free(out->activations);
    free(out->input_buff);
    free(out);

    return;
}


/* Allocates memory needs to be freed
 *
 * Allocates memory for an LSTM network modeled after the network for 
 * Task 6a from the LSTM paper.
 */
Network *init_6a_network()
{
    /* Allocate memory for hidden layer */
    H_layer *hid = init_h_layer(8, // external inputs
                                8, // units in the hidden layer
                                2, // memory blocks
                                2  // memory cells per block
                               );

    /* Get random weights for hidden layer.
     * The LSTM paper states: 
     * "All weights are initialized in the range [-0.1, 0.1]" 
     * This is being interpreted as coming from a uniform distribution. 
     * Other interpretations are possible however.
     */
    double *h_weights = gen_weights(hid->num_weights, 0.1);
    row_stream_set(hid->w, h_weights); // A bit wasteful but easy to read 
    free(h_weights);
   
    /* Fix the self connections in cell blocks */
    for (int i=0; i < hid->num_blocks * hid->block_size; i++) {
        int j=i + hid->num_external_inputs; // Column of self connection 
        hid->w->d[i][j] = 1.0;
    }

    /* Fix the biases of the input gates to -2 and -4 respectively */
    int bias_column = hid->num_external_inputs +
                      hid->num_units;

    /* Go past the memory cells */
    int in_gate_start = hid->num_blocks * hid->block_size;

    /* Admittedly this is not a very good bit of code here... */
    double bias = -2.0;
    for (int i=0; i < hid->num_blocks; i++) {
        hid->w->d[in_gate_start + i][bias_column] = bias; 
        bias -= 2;
    }

    /* Allocate o_layer for output layer */
    O_layer *out = init_o_layer(4,4); // Dimensions are per the paper

    /* Get random weights for the output layer */
    double *o_weights = gen_weights(out->num_weights, 0.1);
    row_stream_set(out->w, o_weights); 
    free(o_weights);

    /* Allocate Network struct */
    Network *n = malloc(sizeof(Network));

    /* plug in the two layers */
    n->hid = hid;
    n->out = out;

    /* Set value of alpha to value from LSTM paper for task 6a */
    n->alpha = 0.5;

    return n; 
}


/* Frees memory allocated to LSTM network and the struct itself */
void free_network(Network *n)
{
    free_h_layer(n->hid);
    free_o_layer(n->out);
    free(n);
}


/* Resets the network activations, internal states of memory cells and 
 * sensitivities. 
 */
void reset(Network *n)
{
    /* Reset hidden layer activations */
    for (int i=0; i < n->hid->num_units; i++) {
        n->hid->activations[i] = 0.0;
    }

    /* Reset output layer activations */
    for (int i=0; i < n->out->num_units; i++) {
        n->out->activations[i] = 0.0;
    }

    /* Reset internal states */
    for (int i=0; i < n->hid->num_cells; i++) {
       n->hid->internal_states[i] = 0.0; 
    }

    /* Reset sensitivities */
    int total_input_len = n->hid->num_external_inputs + n->hid->num_units + 1;
    for (int i=0; i < n->hid->num_cells; i++) {
        for (int m=0; m < total_input_len; m++) {
            n->hid->s_g[i][m] = 0.0;
            n->hid->s_c[i][m] = 0.0;
        }
    }

    return;
}


/* Run an input through the network */
void forward_pass(Network *n, double *input)
{
    /* For convenience */
    /* Layers */
    H_layer *hL = n->hid;
    O_layer *oL = n->out;
    /* Starting positions within hL->activations */
    int in_gate_start  = hL->num_blocks * hL->block_size;
    int out_gate_start = in_gate_start + hL->block_size;
    int num_cells      = hL->num_blocks * hL->block_size;
    int block_size     = hL->block_size;

    /* Write current activations (recurrent input) to input buffer */
    for (int i=0; i < hL->num_units; i++) {
        /* We are putting recurrent inputs at the end of the buffer */
        hL->input_buff[i + hL->num_external_inputs] = hL->activations[i];
    }

    /* Write external input to input buffer */
    for (int i=0; i < hL->num_external_inputs; i++) {
        hL->input_buff[i] = input[i];
    }

    /* Calculate network output and update net_out_buff */
    double *net_out = matrix_vector_prod(hL->w, hL->input_buff);
    for (int i=0; i < hL->num_units; i++) {
        hL->net_out_buff[i] = net_out[i];
    }
    free(net_out);

    /* Evaluate input gate and output gate activations and save to buffer */
    /* The indexing works here since the hidden units are all either:
     *   memory cells
     *   input  gates
     *   output gates
     * So if we start at the input_gate_start and run through them they will 
     * all be gates
     */
    for (int i = in_gate_start; i < hL->num_units; i++) {
        hL->activations[i] = sigmoid(hL->net_out_buff[i]);
    }

    /* For convenience */
    double *act = hL->activations; 

    /* Update internal states of memory cells */
    for (int b=0; b < hL->num_blocks; b++) {
        for (int k=0; k < hL->block_size; k++) {
            /* idx of cell in hL->activations */
            int cell_idx = b * hL->block_size + k; 

            /* current state */
            double *cs = &(hL->internal_states[cell_idx]); 
            /* new state */
            double ncs; 
            ncs = *cs + act[in_gate_start + b]*g_(hL->net_out_buff[cell_idx]);

            /* Update memory state */
            *cs = ncs; 

            /* Calculate activation of memory cell and save */
            hL->activations[cell_idx] = act[out_gate_start + b] * h_(ncs);

        }
    }


    /* Update the sensitivities */
    int total_input_len = hL->num_external_inputs + hL->num_units + 1;
    for (int j=0; j < hL->num_blocks; j++) {
      int in_gate_idx = in_gate_start + j;
      for (int nu=0; nu < hL->block_size; nu++) {
        for (int m=0; m < total_input_len; m++) {
          /* From (25) */
          hL->s_g[j*block_size + nu][m] = hL->s_g[j*block_size + nu][m] + 
                                          g_(hL->net_out_buff[j*block_size + nu]) *
                                          d_f(hL->net_out_buff[in_gate_idx]) *
                                          hL->input_buff[m];
                           
          /* From (27) */
          hL->s_c[j*block_size + nu][m] = hL->s_c[j*block_size + nu][m] +
                                          d_g(hL->net_out_buff[j*block_size + nu]) *
                                          f_(hL->net_out_buff[in_gate_idx]) *
                                          hL->input_buff[m];
        }
      }
    }

    /* Write output of hidden layer to input_buff of output layer */
    for (int i=0; i < num_cells; i++) {
        oL->input_buff[i] = hL->activations[i];
    }

    /* Compute network output of output layer */
    double *o_net_out = matrix_vector_prod(oL->w, oL->input_buff);

    /* Write network outputs to net_out_buff and activations to activations */
    for (int i=0; i < oL->num_h_inputs; i++) {
        oL->net_out_buff[i] = o_net_out[i];
        oL->activations[i]  = f_(o_net_out[i]);
    }
    free(o_net_out);

    return;
}


/* Allocates memory needs to be freed
 *
 * Note: 
 *   This deviates from the paper. The paper uses the backward pass to compute 
 *   weight updates. This function will instead return the truncated gradient. 
 *   The purpose of this is to be able to check it against the finite difference
 *   method to see how big of a discrepancy there is between the two. 
 *   Hence all delta_wij formulas will be -(1/alpha) multiplied by the formula 
 *   instead. Other forumlas will be unchanged.
 */
Gradient *backward_pass(Network *n, double *target)
{
    
    /* Starting positions within n->hid->activations */
    int in_gate_start   = n->hid->num_blocks * n->hid->block_size;
    int out_gate_start  = in_gate_start + n->hid->block_size;
    int num_cells       = n->hid->num_blocks * n->hid->block_size;
    int num_blocks      = n->hid->num_blocks;
    int block_size      = n->hid->block_size;
    int total_input_len = n->hid->num_external_inputs + n->hid->num_units + 1;

    Gradient *grad = init_gradient(n);

    /* (19) */
    double *errors = malloc(sizeof(double) * n->out->num_units);
    for (int k=0; k < n->out->num_units; k++) {
        double err = target[k] - n->out->activations[k]; // FIXME (works better backwards? act - tar ? then add not subtact in update?
        err *= d_f(n->out->net_out_buff[k]);
        errors[k] = err;
    }

    /* (22) Calculate the gradient for output layer (note: -(1/alpha) * (22))
     * Also not that l is being used only to match as much as possible the paper
     * in general l should be avoided as a variable name */
    for (int l=0; l < n->out->num_units; l++) {
        for (int m=0; m < n->out->num_h_inputs + 1; m++) {
            grad->oW->d[l][m] = -1 * errors[l] * n->out->input_buff[m];
        }
    }

    /* The rest is for the gradient of the hidden layer weights */

    /* These sums appear in (21) and (23) hence we do them once and save to
     * an array.
     */
    double *err_sums = malloc(sizeof(double) * num_cells);
    for (int nu = 0; nu < num_cells; nu++) {
        double summa = 0.0;
        for (int k=0; k < n->out->num_units; k++) {
            summa += n->out->w->d[k][nu] * errors[k];
        }
        err_sums[nu] = summa; // Note we will have to take j into account later
    }

    /* Calculate errors for the output gates using (21) */
    for (int j = 0; j < num_blocks; j++) {
        double e_out_j = 0.0;
        for (int nu=0; nu < block_size; nu++) {
            e_out_j += h_(n->hid->internal_states[j*block_size + nu]) *
                       err_sums[j*block_size + nu];
        }
        e_out_j *= d_f(n->hid->net_out_buff[out_gate_start + j]);

        /* Use e_out_j to calc gradients using -(1/alpha) * (22) */
        for (int m=0; m < total_input_len; m++) {
            grad->hW->d[out_gate_start + j][m] = -1 * e_out_j * 
                                                 n->hid->input_buff[m];
        }
    }

    /* (23) */
    double *err_cells = malloc(num_cells * sizeof(double));
    for (int j = 0; j < num_blocks; j++) {
        for (int nu=0; nu < block_size; nu++) {
            double e_s = 1.0;
            e_s *= f_(n->hid->net_out_buff[out_gate_start + j]);
            e_s *= d_h(n->hid->internal_states[j*block_size + nu]);
            e_s *= err_sums[j*block_size + nu];
            err_cells[j*block_size + nu] = e_s;
        }
    }

    /* Calculate gradient for connections to input gate weights -(1/alpha)*(26) */
    for (int j = 0; j < num_blocks; j++) {
        for (int m=0; m < total_input_len; m++) {
            double summa = 0.0;
            for (int nu=0; nu < block_size; nu++) {
                summa += err_cells[j*block_size + nu] * 
                         n->hid->s_g[j*block_size + nu][m];
            }
            grad->hW->d[in_gate_start + j][m] = -1 * summa;
        }
    }

    /* Calculate the gradient for connections to memory cells -(1/alpha)*(28) */
    for (int nu=0; nu < num_cells; nu++) {
        for (int m=0; m < total_input_len; m++) {
            grad->hW->d[nu][m] = err_sums[nu] * n->hid->s_c[nu][m] * -1;
        }
    }

    free(err_cells);
    free(errors);
    free(err_sums);


    return grad;
}


/* Allocates memory needs to be freed 
 *
 * Creates a sequence struct to hold the sequence data
 * Intended to be re-used rather than allocated and deallocated 40000+ times.
 * Hence the max_seq_len field.
 */
Sequence *init_seq(int max_seq_len, int input_len, int tar_len)
{
    Sequence *seq = malloc(sizeof(Sequence));
    seq->max_seq_len = max_seq_len;
    seq->input_len = input_len;
    seq->tar_len = tar_len;

    /* A serparate funciton will fill data and set this value appropriately */
    seq->seq_len = 0;  

    double **sequence = malloc(max_seq_len * sizeof(double*));
    for (int i=0; i < max_seq_len; i++) {
        double *input = calloc(input_len, sizeof(double));
        sequence[i] = input;
    }

    double *target = calloc(tar_len, sizeof(double));

    seq->seq = sequence;
    seq->tar = target;

    return seq;
}


/* Frees the data held by the Sequence and the struct itself */
void free_sequence(Sequence *seq)
{
    for (int i=0; i < seq->max_seq_len; i++) {
        free(seq->seq[i]);
    }
    free(seq->seq);

    free(seq->tar);
    free(seq);

    return;
}


/* Load a task 6a type sequence into a Sequence struct
 *
 * We will use:
 *   1,0,0,0,0,0,0,0 as start
 *   0,0,0,0,0,0,0,1 as end
 *   0,1,0,0,0,0,0,0 as X
 *   0,0,1,0,0,0,0,0 as Y
 *
 *   The rest is noise
 *
 *   TODO Make this not rely on the assertion that input_len is 8
 */
void load_seq(Sequence *seq)
{
    /* Choose t1, t2, seq_len */
    int t1  = rando(10,20); /* Technically off by one but that should be ok */
    int t2  = rando(50,60);
    int len = rando(100,110); /* Assert: max_seq_len is appropriate */

    /* Choose 'X' or 'Y' for t1 and t2 */
    int input_t1 = rando(1,2); // NOTE 1 is for X, 2 is for Y
    int input_t2 = rando(1,2);

    /* Zero out the sequence */
    for (int i=0; i < seq->max_seq_len; i++) {
        for (int j=0; j < seq->input_len; j++) {
            seq->seq[i][j] = 0.0;
        }
    }

    /* Set seq_len */
    seq->seq_len = len;

    /* Set 0th element to start */
    seq->seq[0][0] = 1.0;

    int t = 1;
    while (t < t1) {
        int k = rando(3,6); // Noise index range
        seq->seq[t][k] = 1.0;
        t++;
    }

    /* Set seq at t1 to non-noise value */
    seq->seq[t1][input_t1] = 1.0;

    t++; // Move t to t1 + 1
    while (t < t2) {
        int k = rando(3,6); // Noise index range
        seq->seq[t][k] = 1.0;
        t++;
    }

    /* Set seq at t2 to non-noise value */
    seq->seq[t2][input_t2] = 1.0; 
    t++; // Move t to t2 + 1

    while (t < len - 1) {
        int k = rando(3,6); // Noise index range
        seq->seq[t][k] = 1.0;
        t++;
    }

    /* Note: seq[len] is not set, we run from 0 to len - 1 */
    seq->seq[t][7] = 1.0; 

    /* Zero out target */
    for (int i=0; i < seq->tar_len; i++) {
        seq->tar[i] = 0.0;
    }

    /* Set target */
    /* Case XX */
    if (input_t1 == 1 && input_t2 == 1) {
        seq->tar[0] = 1.0;
    }
    /* Case XY */
    else if (input_t1 == 1 && input_t2 == 2) {
        seq->tar[1] = 1.0;
    }
    /* Case YX */
    else if(input_t1 == 2 && input_t2 == 1) {
        seq->tar[2] = 1.0;
    }
    /* Case YY */
    else if (input_t1 == 2 && input_t2 == 2) {
        seq->tar[3] = 1.0;
    }
    
    return;
}


/* Load a task "generalized" 6a type sequence into a Sequence struct
 *
 * NOTE:
 *   There will now now longer be the same narrow window in which t1 might occur
 *
 * We will use:
 *   1,0,0,0,0,0,0,0 as start
 *   0,0,0,0,0,0,0,1 as end
 *   0,1,0,0,0,0,0,0 as X
 *   0,0,1,0,0,0,0,0 as Y
 *
 */
void load_seq_no_window(Sequence *seq)
{
    /* Choose seq_len first as t1 and t2 are now more broadly ranged */
    int len = rando(100,110); /* Assert: max_seq_len is appropriate */

    /* Choose t1, t2 */
    int t1  = rando(1,len-3);  /* len - 1 is END so we need to leave room for t2 and end */
    int t2  = rando(t1 + 1, len-2);

    /* Choose 'X' or 'Y' for t1 and t2 */
    int input_t1 = rando(1,2); // NOTE 1 is for X, 2 is for Y
    int input_t2 = rando(1,2);

    /* Zero out the sequence */
    for (int i=0; i < seq->max_seq_len; i++) {
        for (int j=0; j < seq->input_len; j++) {
            seq->seq[i][j] = 0.0;
        }
    }

    /* Set seq_len */
    seq->seq_len = len;

    /* Set 0th element to start */
    seq->seq[0][0] = 1.0;

    int t = 1;
    while (t < t1) {
        int k = rando(3,6); // Noise index range
        seq->seq[t][k] = 1.0;
        t++;
    }

    /* Set seq at t1 to non-noise value */
    seq->seq[t1][input_t1] = 1.0;

    t++; // Move t to t1 + 1
    while (t < t2) {
        int k = rando(3,6); // Noise index range
        seq->seq[t][k] = 1.0;
        t++;
    }

    /* Set seq at t2 to non-noise value */
    seq->seq[t2][input_t2] = 1.0; 
    t++; // Move t to t2 + 1

    while (t < len - 1) {
        int k = rando(3,6); // Noise index range
        seq->seq[t][k] = 1.0;
        t++;
    }

    /* Note: seq[len] is not set, we run from 0 to len - 1 */
    seq->seq[t][7] = 1.0; 

    /* Zero out target */
    for (int i=0; i < seq->tar_len; i++) {
        seq->tar[i] = 0.0;
    }

    /* Set target */
    /* Case XX */
    if (input_t1 == 1 && input_t2 == 1) {
        seq->tar[0] = 1.0;
    }
    /* Case XY */
    else if (input_t1 == 1 && input_t2 == 2) {
        seq->tar[1] = 1.0;
    }
    /* Case YX */
    else if(input_t1 == 2 && input_t2 == 1) {
        seq->tar[2] = 1.0;
    }
    /* Case YY */
    else if (input_t1 == 2 && input_t2 == 2) {
        seq->tar[3] = 1.0;
    }
    
    return;
}


/* Updates the weights of the network using steepest descent */
void update(Network *n, Gradient *grad)
{
    /* Update the hidden layer */
    int rows, cols;
    int num_cells = n->hid->num_cells;
    int num_external_inputs = n->hid->num_external_inputs;

    /* Check that dimensions match */
    if (n->hid->w->rows != grad->hW->rows || n->hid->w->cols != grad->hW->cols){
        fprintf(stderr, "Hidden layer and gradient dimensions do not match\n");
        fprintf(stderr, "Exiting...\n");
        exit(EXIT_FAILURE);
    }

    rows = n->hid->w->rows;
    cols = n->hid->w->cols;

    for (int i=0; i < rows; i++) {
        for (int j=0; j < cols; j++) {
            /* Do NOT update the memory cell self connections */
            if (i < num_cells && j == num_external_inputs + i) {
                continue;
            }
            /* SEE FIXME in backward_pass */
            n->hid->w->d[i][j] -= n->alpha * grad->hW->d[i][j];
        }
    } 

    /* Update the output weights */
    
    /* Check that dimensions match */
    if (n->out->w->rows != grad->oW->rows || n->out->w->cols != grad->oW->cols){
        fprintf(stderr, "Output layer and gradient dimensions do not match\n");
        fprintf(stderr, "Exiting...\n");
        exit(EXIT_FAILURE);
    }

    rows = n->out->w->rows;
    cols = n->out->w->cols;

    for (int i=0; i < rows; i++) {
        for (int j=0; j < cols; j++) {
            /* SEE FIXME in backward_pass */
            n->out->w->d[i][j] -=  n->alpha * grad->oW->d[i][j];
        }
    }

    return;
}


/* Computes the mean squared error of the network output activations as compared
 * to the target
 */
double MSE(Network *n, double *target)
{
    double summa = 0.0;
    for (int i=0; i < n->out->num_units; i++) {
        double tmp = target[i] - n->out->activations[i];
        tmp *= tmp;
        summa += tmp;
    }

    return (0.5) * summa;
}


/* Determines whether or not the network output is "correct" or not.
 * Determination is based on the criteria outlined in the LSTM paper.
 * Incorrect is defined from page 20: "error > 0.3 for at least one output unit".
 * Here it is not clear if the authors mean absolute error or mean squared error.
 * We will use the more stringent absolute error.
 *
 * Returns 1 if correct, 0 otherwise. 
 */
int is_correct(Network *n, double *target)
{
    double threhold = 0.3;

    int num_units = n->out->num_units;

    for (int i=0; i < num_units; i++) {
        double d = target[i] - n->out->activations[i];
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
    }

    return 1;
}


/* Trains the network n on random task 6a type sequences.
 * Returns the number of trials it took to train, or -1 if 
 * max_iter is reached without successful training.
 */
int train(Network *n, int max_iter)
{
    Sequence *seq = init_seq(110, 8, 4);

    /* Tracks what trial we are on */
    int trials = 0;

    /* Counts the current tally of how many correct responses the
     * network has generated in a row */
    int streak = 0;

    /* Determines the window in which we will check the stopping criteria */
    int window = 2000;

    /* Used to track the average error for the most recent trials */
    double *window_errors = calloc(window, sizeof(double));
    int e_idx = 0;


    /* Used to track total of all errors */
    double all_errors = 0.0;

    while (trials < max_iter) {
        trials++;
        load_seq(seq);
        //load_seq_no_window(seq);
        reset(n);
        for (int t=0; t < seq->seq_len; t++) {
            forward_pass(n, seq->seq[t]);
        }

        int correct = is_correct(n, seq->tar);

        /* We will only evalute the average for the most recent trials 
         * as determined by window */
        double err = MSE(n, seq->tar);
        all_errors += err;
        window_errors[e_idx] = err;
        e_idx = (e_idx + 1) % window;
        double total_window_error = 0.0;
        for (int i=0; i < window; i++) {
            total_window_error += window_errors[i];
        }
        double avg_error = all_errors / trials;
        double window_avg_error = total_window_error / window;


        if (correct) {
            streak++;
        }
        else {
            streak = 0;
        }


        
        if (streak >= window && window_avg_error < 0.1) {
            break;
        }

        /* Perform gradient descent */
        Gradient *grad = backward_pass(n, seq->tar);
        
        update(n, grad);
        free_gradient(grad);

        printf("%d\t%d\t%g\n", trials, streak, window_avg_error);

    }

    free_sequence(seq);
    free(window_errors);

    if (trials == max_iter) {
        return -1;
    }
    else {
        return trials;
    }

}


/* test the network on new data... */
void test_network(Network *n, int num_tests, double seed)
{
    Sequence *seq = init_seq(110, 8, 4);

    reset_seed(seed);

    /* Tracks what trial we are on */
    int trials = 0;


    /* Used to track total of all errors */
    double errors = 0.0;

    int successes = 0;

    while (trials < num_tests) {
        trials++;
        load_seq(seq);
        reset(n);
        for (int t=0; t < seq->seq_len; t++) {
            forward_pass(n, seq->seq[t]);
        }

        int correct = is_correct(n, seq->tar);

        double err = MSE(n, seq->tar);
        errors += err;


        if (correct) {
            successes++;
        }
        else {
        }



        //printf("%d\t%d\t%g\n", trials, streak, window_avg_error);

    }

    free_sequence(seq);

    double avg_success = (double) successes / num_tests;
    double avg_error   = errors / num_tests;

    printf("Had %d successes out of %d trials\n", successes, num_tests);
    printf("For a success rate of %f\n", avg_success);
    printf("Average error rate was %f\n", avg_error);


}


void weight_guessing(int seed)
{
    reset_seed(seed);

    /* Network has architecture described in LSTM paper for task 6a
     * weights have been intilized accordingly. Presumably the random 
     * weights were uniformly distrbuted however they may have used a 
     * different distribution. */
    Network *n = init_6a_network();

    /* Allocate memory for task 6a seq struct */
    Sequence *seq = init_seq(110, 8, 4);

    /* Tracks what trial we are on */
    int trials = 0;

    /* Counts the current tally of how many correct responses the
     * network has generated in a row */
    int streak = 0;

    /* Determines the window in which we will check the stopping criteria */
    int window = 100;

    /* Cut-off to stop program if we have not successfully trained the network */
    int max_iter = 200000;

    /* Used to track the average error for the most recent trials */
    double all_errors = 0.0;
    double *window_errors = calloc(window, sizeof(double));
    int e_idx = 0;

    while (trials < max_iter) {
        trials++;
        load_seq(seq);
        //load_seq_no_window(seq);
        reset(n);
        for (int t=0; t < seq->seq_len; t++) {
            forward_pass(n, seq->seq[t]);
        }

        int correct = is_correct(n, seq->tar);

        /* We will only evalute the average for the most recent trials 
         * as determined by window */
        double err = MSE(n, seq->tar);
        window_errors[e_idx] = err;
        e_idx = (e_idx + 1) % window;
        double total_window_error = 0.0;
        for (int i=0; i < window; i++) {
            total_window_error += window_errors[i];
        }

        double avg_window_error;
        if (trials < window) {
            avg_window_error = total_window_error / trials;
        }
        else {
            avg_window_error = total_window_error / window;
        }

        if (correct) {
            streak++;
                    }
        else {
            streak = 0;
            /* If no good, trash network and start over */
            free_network(n);
            n = init_6a_network();
        }

        
        if (streak >= 2000 && avg_window_error < 0.1) {
            //printf("Successfully trained in %d iterations\n", trials);
            break;
        }
        printf("%d %d %f\n", trials, streak, avg_window_error);


    }

    if (trials == max_iter) {
        printf("Failed to train network in %d trials\n", trials);
    }
    else {
        printf("%d\n", trials);
    }

    free_sequence(seq);
    free_network(n);
    //free_rng();
    free(window_errors);

    return;
}


void save_weights(Network *n, char *filename)
{

    printf("Saving network to: %s\n", filename);
    FILE *fp = fopen(filename, "w");

    /* Write the number weights in the hidden and output layers */
    int num_h_weights = n->hid->w->rows * n->hid->w->cols;
    int num_o_weights = n->out->w->rows * n->out->w->cols;

    fprintf(fp, "%d %d\n", num_h_weights, num_o_weights);

    /* Write hidden layer weights */
    for (int i=0; i < n->hid->num_units; i++) {
        for (int j=0; j < n->hid->num_units + n->hid->num_external_inputs + 1; j++) {
            fprintf(fp, "%f ", n->hid->w->d[i][j]);
        }
        fprintf(fp, "\n");
    }
    /* Write output layer */
    for (int i=0; i < n->out->num_units; i++) {
        for (int j=0; j < n->out->num_h_inputs + 1; j++) {
            fprintf(fp, "%f ", n->out->w->d[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    return;

}



/* Loads weights to a network from a file */
void load_weights_from(Network *n, char *filename) 
{
    FILE *fp = fopen(filename, "r");

    int s1 = 0;
    int s2 = 0;

    fscanf(fp, "%d%d", &s1, &s2);

    int num_h_weights = n->hid->w->rows * n->hid->w->cols;
    int num_o_weights = n->out->w->rows * n->out->w->cols;

    if (s1 != num_h_weights || s2 != num_o_weights) {
        printf("Error, weight file does not have proper configuration\n");
        printf("Exiting...\n");
        exit(EXIT_FAILURE);

    }

    double *h_weights = malloc(num_h_weights * sizeof(double));
    double *o_weights = malloc(num_o_weights * sizeof(double));

    int count = 0;
    double x;
    while (count < num_h_weights) {
        int ret_val = fscanf(fp, "%lf", &x);
        if (ret_val == EOF) {
            printf("Error, weight file does not have proper configuration\n");
            printf("Exiting...\n");
            exit(EXIT_FAILURE);
        }
        //printf("%f\n", x);
        h_weights[count] = x;
        count++;
    }

    count = 0;
    while (count < num_o_weights) {
        int ret_val = fscanf(fp, "%lf", &x);
        if (ret_val == EOF) {
            printf("Error, weight file does not have proper configuration\n");
            printf("Exiting...\n");
            exit(EXIT_FAILURE);
        }
        //printf("%f\n", x);
        o_weights[count] = x;
        count++;
    }

    row_stream_set(n->hid->w, h_weights);
    row_stream_set(n->out->w, o_weights);

    fclose(fp);
    free(h_weights);
    free(o_weights);

    return;
}


/*set input*/
void set_input(double *input, int len, char c)
{
    /* zero out input */
    for (int i=0; i < len; i++) {
        input[i] = 0.0;
    }

    int idx = 0;
    switch(c) {
        case 's': 
            idx = 0;
            break;
        case 'e':
            idx = 7;
            break;
        case 'a':
            idx = 3;
            break;
        case 'b':
            idx = 4;
            break;
        case 'c':
            idx = 5;
            break;
        case 'd':
            idx = 6;
            break;
        case 'x':
            idx = 1;
            break;
        case 'y':
            idx = 2;
            break;
    }


    /* Set appropraite value to 1.0 */
    input[idx] = 1.0;

    return;
}


/* */
void repl_run(Network *n, char *mode)
{
    char *message =  
    "Enter:\n"
    "  s        to reset the network and send start of seq\n"
    "  e        to send the end of seq\n"
    "  a,b,c,d  to send the noise input\n"
    "  x,y      meaningful input\n";

    printf("%s", message);

    char seen[2] = {'\0', '\0'};
    int rec_idx = 0;

    reset(n);
    double *input = calloc(n->hid->num_external_inputs , sizeof(double));
    char c = '\0';
    int t = 0;
    while(1) {
        scanf(" %c", &c);
        if (c == 's') {
            printf("\nresetting network and giving start of seq\n");
            t = 0;
            reset(n);
            set_input(input, n->hid->num_external_inputs, c);
            forward_pass(n,input);
            seen[0] = '\0';
            seen[1] = '\0';
            rec_idx = 0;
        }
        else if (c == 'a' || c == 'b' || c == 'c' || c == 'd' || c == 'e')
        {
            set_input(input, n->hid->num_external_inputs, c);
            print_row_vector(input,n->hid->num_external_inputs);
            forward_pass(n,input);
        }
        else if (c == 'x' || c =='y') 
        {
            set_input(input, n->hid->num_external_inputs, c);
            print_row_vector(input,n->hid->num_external_inputs);
            forward_pass(n,input);
            if (rec_idx < 2) {
                seen[rec_idx++] = c;
            }
        }
        

        else {
            continue;
        }

        t++;

        printf("t = %d\n", t);
        printf("Cell states:\n");
        for (int i=0; i < n->hid->num_cells; i++) {
            printf("%f ", n->hid->internal_states[i]);
        }
        printf("\n");
        printf("Activations output layer\n");
        for (int i=0; i < n->out->num_units; i++) {
            printf("%f ", n->out->activations[i]);
        }
        printf("\n");


        /* If end of signal is sent determine if output is correct */
        if (c == 'e')
        {
            set_input(input, n->hid->num_external_inputs, c);
            print_row_vector(input,n->hid->num_external_inputs);
            forward_pass(n,input);

            /* This should match load_seq */
            double tar[4] = {0,0,0,0};
            if (seen[0] == 'x' && seen[1] == 'x') {
               tar[0] = 1.0;
            }
            else if (seen[0] == 'x' && seen[1] == 'y') {
                tar[1] = 1.0;
            }
            else if (seen[0] == 'y' && seen[1] == 'x') {
                tar[2] = 1.0;
            }
            else {
                tar[3] == 1.0;
            }

            int correct = is_correct(n, tar);
            printf("\n");
            if (correct) {
                printf("Network output is correct:\nSignal is: %c%c\n",seen[0], seen[1]);
            }
            else {
                printf("Network output is NOT correct:\nSignal is %c%c\n", seen[0], seen[1]);
                double max = 0.0;
                int m_idx = 0;
                for (int i=0; i < n->out->num_units; i++) {
                    if (n->out->activations[i] > max) {
                        max = n->out->activations[i];
                        m_idx = i;
                    }
                }

                printf("Output is: ");
                if (m_idx == 0) {
                    printf("xx\n");
                }
                else if (m_idx == 1) {
                    printf("xy\n");
                }
                else if (m_idx == 2) {
                    printf("yx\n");
                }
                else {
                    printf("yy\n");
                }

                if (max < 0.7) {
                    printf("Output error too high\n");
                }
            }

        }


    }

        


}






int main(int argc, char *argv[])
{
    if (argc == 1) {
        printf("Useage:"
                "\n-t"
                "  train a single LSTM network on task 6a and get the number\n"
                "    of trials back\n"
                "\n-m"
                "  train 30 LSTM networks on task 6a and get the average\n"
                "    number of trails to train\n"
                "\n-s"
                "  train an LSTM and save the network weights to __lstm_weights.txt\n"
                "\n-l"
                "  loads the weights from __lstm_weights.txt and runs in a REPL\n"
                "\n-x"
                "  loads the weights from __lstm_weights.txt and tests the network\n"
                "\n");
        exit(EXIT_SUCCESS);
    }

    alloc_rng();

    /* Stop training if number of trials reaches this value */
    int max_iter = 200000;
    

    /* Train a network */
    if (argc > 1 && strcmp(argv[1],"-t") == 0) {

        /* Network has architecture described in LSTM paper for task 6a
         * weights have been intilized accordingly. Presumably the random 
         * weights were uniformly distrbuted however they may have used a 
         * different distribution. */
        Network *n = init_6a_network();

        int k = train(n,max_iter);
        if (k != -1) {
            printf("Network trained in %d trials\n", k);
        }
        else if (k == -1){
            printf("Failed to train network in %d trials\n", max_iter);
        }
        free_network(n);
    }

    /* Train 30 networks and get the average time to train */
    else if (argc > 1 && strcmp(argv[1], "-m") == 0) {

        int seeds[20] = {37, 370, 17,   16,  15,
                                 100,3,   1000, 880, 10,
                                 390, 790, 3000, 444, 700,
                                 951,451 ,101 ,560 ,1039};
                                 

        int summa = 0;
        for (int i=0; i < 20; i++) {

            reset_seed(seeds[i]);
            Network *n = init_6a_network();

            int k = train(n, max_iter);
            if (k != -1) {
                printf("Network trained in %d trials\n", k);
            }
            else if (k == -1){
                printf("Failed to train network in %d trials\n", max_iter);
            }
            summa += k;

            free_network(n);
        }

        double average = (double) summa / 20.0;

        printf("Average training time is %.2f trials\n", average);
    }

    /* Train a network and save the weights to a file */
    else if ( argc > 1 && strcmp(argv[1],"-s") == 0) {

        Network *n = init_6a_network();

        printf("Training network...\n");

        int k = train(n,max_iter);
        if (k != -1) {
            printf("Network trained in %d trials\n", k);
        }
        else if (k == -1){
            printf("Failed to train network in %d trials\n", max_iter);
        }

        /* Save network */
        char *filename = "__lstm_weights.txt";
        save_weights(n, filename);


        free_network(n);

    }

    /* Loads the network saved to __lstm_weights.txt and runs it in a REPL */
    else if ( argc > 1 && strcmp(argv[1],"-l") == 0 ) {
        char *filename = "__lstm_weights.txt";

        printf("Loading network from %s\n", filename);

        Network *n = init_6a_network();

        load_weights_from(n, filename);

        repl_run(n, "");

        free_network(n);
    }

    /* Loads the network saved to __lstm_weights.txt and tests it */
    else if ( argc > 1 && strcmp(argv[1],"-x") == 0 ) {
        char *filename = "__lstm_weights.txt";

        printf("Loading network from %s\n", filename);

        Network *n = init_6a_network();

        load_weights_from(n, filename);

        /* Get random seed */
        time_t curr_time = time(NULL);
        int seed = (int) curr_time % INT_MAX;

        int num_trials = 3000;

        printf("Running %d tests using seed: %d\n", num_trials, seed);
        test_network(n, num_trials, seed);

        free_network(n);
    }

    
    free_rng();

    return 0;
}
