/*
 * Adaline Network with BitNet Ternary Quantization {-1, 0, +1}
 *
 * Modern rewrite of the classic Adaline (Widrow-Hoff) network using
 * BitNet-style ternary weight quantization. Classifies digit patterns
 * (0-9) rendered on a 5x7 grid.
 *
 * Key changes from the original:
 *   - Weights quantized to ternary {-1, 0, +1} via absmean scaling
 *   - Master weights in double precision for gradient updates (STE)
 *   - Forward pass uses integer MACs (multiply-accumulate) with scale
 *   - Same delta-rule learning preserved through Straight-Through Estimator
 *
 * Author:     Karsten Kutza (original), Modern rewrite
 * Date:       15.4.96 (original), 2026-05-15 (BitNet rewrite)
 * Reference:  B. Widrow, M.E. Hoff - Adaptive Switching Circuits, 1960
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

/* ------------------------------------------------------------------ */
/*  Constants                                                         */
/* ------------------------------------------------------------------ */

#define NUM_DATA       10U          /* number of training patterns     */
#define X_DIM          5U           /* grid width                      */
#define Y_DIM          7U           /* grid height                     */
#define N_INPUT        (X_DIM * Y_DIM)  /* total input features         */
#define M_OUTPUT       10U          /* output classes (digits 0-9)   */

#define BIAS           1            /* bias input value                */
#define HI             1            /* high ternary state              */
#define LO            -1            /* low ternary state               */

#define FALSE          0
#define TRUE           1

/* ------------------------------------------------------------------ */
/*  Ternary weight type                                               */
/* ------------------------------------------------------------------ */

typedef int8_t ternary_t;   /* {-1, 0, +1} -- fits in signed char     */

/* ------------------------------------------------------------------ */
/*  Layer structure                                                   */
/* ------------------------------------------------------------------ */

struct layer {
    uint32_t       units;      /* number of neurons in this layer         */
    int           *output;     /* integer output of each neuron           */
    double        *activation; /* floating-point activation (pre-clip)    */
    double        *error;      /* error term for backprop                 */

    /* BitNet ternary quantization fields */
    double        *master_w;   /* full-precision master weights (STE)     */
    ternary_t     *quant_w;    /* quantized ternary weights {-1,0,+1}     */
    double         scale;      /* per-layer absmean scale factor          */
};

/* ------------------------------------------------------------------ */
/*  Network structure                                                 */
/* ------------------------------------------------------------------ */

struct network {
    struct layer *input_layer;   /* input layer (with bias)               */
    struct layer *output_layer;  /* output layer                          */
    double        eta;           /* learning rate                         */
    double        error;         /* total network error                   */
    double        epsilon;       /* convergence threshold                 */
};

typedef struct layer   LAYER;
typedef struct network NET;

/* ------------------------------------------------------------------ */
/*  Macros                                                            */
/* ------------------------------------------------------------------ */

#define SQR(x)          ((x) * (x))
#define MIN(a, b)       ((a) < (b) ? (a) : (b))
#define MAX(a, b)       ((a) > (b) ? (a) : (b))

/* ------------------------------------------------------------------ */
/*  Random number helpers                                             */
/* ------------------------------------------------------------------ */

static void init_randoms(void)
{
    srand(4711);
}

static int random_int(int low, int high)
{
    return rand() % (high - low + 1) + low;
}

static double random_double(double low, double high)
{
    return ((double)rand() / (double)RAND_MAX) * (high - low) + low;
}

/* ------------------------------------------------------------------ */
/*  BitNet absmean quantization                                       */
/*
 *  Quantize master weights to ternary {-1, 0, +1}:
 *
 *    scale = mean(|W_master|)   over all weight elements
 *    W_q[i] = clamp(round(W_master[i] / scale), -1, +1)
 *
 *  This is the absmean scheme used in BitNet b1.58.
 *  round() maps:  |x| >= 0.5 -> sign(x),  |x| < 0.5 -> 0
 */

static void quantize_weights(LAYER *layer, uint32_t input_units)
{
    const uint32_t feat = input_units;
    double abs_sum = 0.0;
    uint32_t total = layer->units * feat;
    uint32_t i, j;

    /* Step 1: compute scale = mean(|W|) */
    for (i = 0; i < layer->units; i++) {
        for (j = 0; j < feat; j++) {
            abs_sum += fabs(layer->master_w[i * feat + j]);
        }
    }
    layer->scale = abs_sum / (double)total;

    /* Step 2: quantize to ternary via round(W/scale) */
    for (i = 0; i < layer->units; i++) {
        for (j = 0; j < feat; j++) {
            double ratio = layer->master_w[i * feat + j] / layer->scale;
            int rounded = (int)(ratio >= 0.0 ? ratio + 0.5 : ratio - 0.5);

            if (rounded > 0)
                layer->quant_w[i * feat + j] = HI;
            else if (rounded < 0)
                layer->quant_w[i * feat + j] = LO;
            else
                layer->quant_w[i * feat + j] = 0;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Application-specific data                                         */
/* ------------------------------------------------------------------ */

static const char pattern[NUM_DATA][Y_DIM][X_DIM + 1] = {
    /* Digit 0 */
    { " OOO ",
      "O   O",
      "O   O",
      "O   O",
      "O   O",
      "O   O",
      " OOO " },

    /* Digit 1 */
    { "  O  ",
      " OO  ",
      "O O  ",
      "  O  ",
      "  O  ",
      "  O  ",
      "  O  " },

    /* Digit 2 */
    { " OOO ",
      "O   O",
      "    O",
      "   O ",
      "  O  ",
      " O   ",
      "OOOOO" },

    /* Digit 3 */
    { " OOO ",
      "O   O",
      "    O",
      " OOO ",
      "    O",
      "O   O",
      " OOO " },

    /* Digit 4 */
    { "   O ",
      "  OO ",
      " O O ",
      "O  O ",
      "OOOOO",
      "   O ",
      "   O " },

    /* Digit 5 */
    { "OOOOO",
      "O    ",
      "O    ",
      "OOOO ",
      "    O",
      "O   O",
      " OOO " },

    /* Digit 6 */
    { " OOO ",
      "O   O",
      "O    ",
      "OOOO ",
      "O   O",
      "O   O",
      " OOO " },

    /* Digit 7 */
    { "OOOOO",
      "    O",
      "    O",
      "   O ",
      "  O  ",
      " O   ",
      "O    " },

    /* Digit 8 */
    { " OOO ",
      "O   O",
      "O   O",
      " OOO ",
      "O   O",
      "O   O",
      " OOO " },

    /* Digit 9 */
    { " OOO ",
      "O   O",
      "O   O",
      " OOOO",
      "    O",
      "O   O",
      " OOO " },
};

static int input_data[NUM_DATA][N_INPUT];
static int target_output[NUM_DATA][M_OUTPUT];

static FILE *log_file;

/* ------------------------------------------------------------------ */
/*  Application initialisation                                        */
/* ------------------------------------------------------------------ */

static void init_application(NET *net)
{
    unsigned n, i, j;

    net->eta     = 0.01;
    net->epsilon = 0.5;

    for (n = 0; n < NUM_DATA; n++) {
        for (i = 0; i < Y_DIM; i++) {
            for (j = 0; j < X_DIM; j++) {
                input_data[n][i * X_DIM + j] =
                    (pattern[n][i][j] == 'O') ? HI : LO;
            }
        }
    }

    /* One-hot targets: class k -> +1 at position k, -1 elsewhere */
    for (n = 0; n < NUM_DATA; n++) {
        for (unsigned c = 0; c < M_OUTPUT; c++) {
            target_output[n][c] = (c == n) ? HI : LO;
        }
    }

    log_file = fopen("ADALINE.txt", "w");
}

/* ------------------------------------------------------------------ */
/*  Write helpers for logging                                         */
/* ------------------------------------------------------------------ */

static void write_input(const NET *net, const int *inp)
{
    (void)net;
    for (unsigned i = 0; i < N_INPUT; i++) {
        if (i % X_DIM == 0)
            fputc('\n', log_file);
        fputc((inp[i] == HI) ? 'O' : ' ', log_file);
    }
    fprintf(log_file, " -> ");
}

static void write_output(const NET *net, const int *out)
{
    (void)net;
    int count = 0, index = -1;

    for (unsigned i = 0; i < M_OUTPUT; i++) {
        if (out[i] == HI) {
            count++;
            index = (int)i;
        }
    }

    if (count == 1)
        fprintf(log_file, "%d\n", index);
    else
        fprintf(log_file, "invalid\n");
}

static void finalize_application(NET *net)
{
    (void)net;
    fclose(log_file);
}

/* ------------------------------------------------------------------ */
/*  Network creation                                                  */
/* ------------------------------------------------------------------ */

static void generate_network(NET *net)
{
    const uint32_t feat = N_INPUT + 1U;   /* inputs + bias */

    /* Input layer */
    net->input_layer = malloc(sizeof(LAYER));
    if (!net->input_layer) {
        fprintf(stderr, "Error: failed to allocate input layer\n");
        exit(EXIT_FAILURE);
    }
    net->input_layer->units  = N_INPUT;
    net->input_layer->output = calloc(feat, sizeof(int));
    if (!net->input_layer->output) {
        fprintf(stderr, "Error: failed to allocate input outputs\n");
        exit(EXIT_FAILURE);
    }
    net->input_layer->output[0] = BIAS;   /* bias unit */

    /* Output layer */
    net->output_layer = malloc(sizeof(LAYER));
    if (!net->output_layer) {
        fprintf(stderr, "Error: failed to allocate output layer\n");
        exit(EXIT_FAILURE);
    }
    net->output_layer->units     = M_OUTPUT;
    net->output_layer->activation = calloc(feat, sizeof(double));
    net->output_layer->output     = calloc(feat, sizeof(int));
    net->output_layer->error      = calloc(feat, sizeof(double));

    /* BitNet: master weights (double) + quantized weights (ternary) */
    net->output_layer->master_w = calloc((size_t)feat * M_OUTPUT, sizeof(double));
    net->output_layer->quant_w  = calloc((size_t)feat * M_OUTPUT, sizeof(ternary_t));

    if (!net->output_layer->activation ||
        !net->output_layer->output     ||
        !net->output_layer->error      ||
        !net->output_layer->master_w   ||
        !net->output_layer->quant_w) {
        fprintf(stderr, "Error: failed to allocate output layer arrays\n");
        exit(EXIT_FAILURE);
    }

    net->eta     = 0.1;
    net->epsilon = 0.01;
}

/* ------------------------------------------------------------------ */
/*  Initialise master weights randomly, then quantise                 */
/* ------------------------------------------------------------------ */

static void random_weights(NET *net)
{
    const uint32_t feat = net->input_layer->units + 1U;
    uint32_t i, j;

    for (i = 0; i < net->output_layer->units; i++) {
        for (j = 0; j < feat; j++) {
            net->output_layer->master_w[i * feat + j] =
                random_double(-0.5, 0.5);
        }
    }

    /* Quantise to ternary */
    quantize_weights(net->output_layer, net->input_layer->units + 1U);
}

/* ------------------------------------------------------------------ */
/*  Set / get input and output                                        */
/* ------------------------------------------------------------------ */

static void set_input(NET *net, const int *inp, int protocol)
{
    uint32_t i;

    for (i = 1; i <= net->input_layer->units; i++) {
        net->input_layer->output[i] = inp[i - 1];
    }
    if (protocol)
        write_input(net, inp);
}

static void get_output(NET *net, int *out, int protocol)
{
    uint32_t i;

    for (i = 1; i <= net->output_layer->units; i++) {
        out[i - 1] = net->output_layer->output[i];
    }
    if (protocol)
        write_output(net, out);
}

/* ------------------------------------------------------------------ */
/*  Forward propagation with ternary weights                          */
/*
 *  For each output neuron:
 *      activation = scale * sum_j( quant_w[j] * input[j] )
 *      output     = +1 if activation >= 0, else -1
 *
 *  The multiply is free (ternary -> add/subtract or nothing).
 */

static void propagate_net(NET *net)
{
    const LAYER *in   = net->input_layer;
    LAYER       *out  = net->output_layer;
    const uint32_t feat = in->units + 1U;
    uint32_t i, j;

    for (i = 0; i < out->units; i++) {
        double sum = 0.0;

        for (j = 0; j < feat; j++) {
            ternary_t w = out->quant_w[i * feat + j];
            int x = in->output[j];

            /* Ternary multiply: only add/subtract, no FMUL */
            if (w == HI)
                sum += (double)x;
            else if (w == LO)
                sum -= (double)x;
            /* w == 0 -> skip */
        }

        out->activation[i] = out->scale * sum;
        out->output[i]     = (out->activation[i] >= 0.0) ? HI : LO;
    }
}

/* ------------------------------------------------------------------ */
/*  Compute error (MSE-like, same as original)                        */
/* ------------------------------------------------------------------ */

static void compute_output_error(NET *net, const int *target)
{
    LAYER *out = net->output_layer;
    double err;
    uint32_t i;

    net->error = 0.0;

    for (i = 0; i < out->units; i++) {
        err = (double)target[i] - out->activation[i];
        out->error[i] = err;
        net->error += 0.5 * SQR(err);
    }
}

/* ------------------------------------------------------------------ */
/*  Adjust master weights (delta rule via STE)                        */
/*
 *  Straight-Through Estimator:
 *      W_master += eta * error * input
 *
 *  Master weights stay in full precision during training.
 *  Quantization only happens before forward passes.
 */

static void adjust_weights(NET *net)
{
    LAYER *out = net->output_layer;
    const uint32_t feat = net->input_layer->units + 1U;
    uint32_t i, j;

    for (i = 0; i < out->units; i++) {
        for (j = 0; j < feat; j++) {
            double grad = net->eta * out->error[i] *
                          (double)net->input_layer->output[j];
            out->master_w[i * feat + j] += grad;
        }
    }

    /* Re-quantise after update so next forward pass uses ternary */
    quantize_weights(out, net->input_layer->units + 1U);
}

/* ------------------------------------------------------------------ */
/*  Single-sample simulation                                          */
/* ------------------------------------------------------------------ */

static void simulate_net(NET *net, const int *inp, const int *target,
                         int training, int protocol)
{
    int output[M_OUTPUT];

    set_input(net, inp, protocol);
    propagate_net(net);
    get_output(net, output, protocol);

    compute_output_error(net, target);

    if (training)
        adjust_weights(net);
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(void)
{
    NET net;
    double error;
    int stop;
    unsigned n, m;

    init_randoms();
    generate_network(&net);
    random_weights(&net);
    init_application(&net);

    do {
        error = 0.0;
        stop  = TRUE;

        for (n = 0; n < NUM_DATA; n++) {
            simulate_net(&net, input_data[n], target_output[n],
                         FALSE, FALSE);
            error = MAX(error, net.error);
            stop  = stop && (net.error < net.epsilon);
        }

        error = MAX(error, net.epsilon);
        printf("Training %.0f%% completed ...\n",
               (net.epsilon / error) * 100.0);

        if (!stop) {
            for (m = 0; m < 10U * NUM_DATA; m++) {
                n = (unsigned)random_int(0, (int)NUM_DATA - 1);
                simulate_net(&net, input_data[n], target_output[n],
                             TRUE, FALSE);
            }
        }
    } while (!stop);

    /* Final classification of all patterns */
    for (n = 0; n < NUM_DATA; n++) {
        simulate_net(&net, input_data[n], target_output[n],
                     FALSE, TRUE);
    }

    finalize_application(&net);

    return EXIT_SUCCESS;
}
