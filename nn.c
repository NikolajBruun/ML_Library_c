#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "LogReg.h"
#include "nn.h"
#include "evaluation.h"

double relu(double x)
{
    return fmax(0, x);
}

double find_max(struct matrix *x)
{
    double max = x->elements[0];
    for (int i = 1; i < x->cols * x->rows; i++)
    {
        if (max < x->elements[i])
        {
            max = x->elements[i];
        }
    }
    return max;
}
struct matrix *softmax(struct matrix *input)
{
    // først finder vi max
    double max = find_max(input);
    struct matrix *output = matrix_create(input->rows, 1);
    // først konvertere vi til exp
    for (int i = 0; i < input->rows; i++)
    {
        output->elements[i] = exp(input->elements[i] - max);
    }
    double sum = 0.0;
    for (int i = 0; i < input->rows; i++)
    {
        // udregner summen -> bruges i nævner
        sum += output->elements[i];
    }

    for (int i = 0; i < input->rows; i++)
    {
        output->elements[i] = output->elements[i] / sum;
    }
    return output;
}

struct matrix *matrix_lin_comb(struct matrix *x, struct matrix *w)
{
    if (x->cols != 1)
    {
        fprintf(stderr, "x skal være søjlevektor");
        exit(1);
    }
    struct matrix *z = matrix_mul(w, x);
    return z;
}

struct matrix *compute_layer_output(struct layer *layer, struct matrix *input)
{
    // vi udvider først x og w så bias bliver built-in
    // vi starter med x som skal have tilføjet et 1
    struct matrix *x_with_bias = matrix_create(input->rows + 1, 1);
    for (int i = 0; i < input->rows; i++)
    {
        x_with_bias->elements[i] = input->elements[i];
    }
    x_with_bias->elements[x_with_bias->rows - 1] = 1;

    // og nu udvider vi w med en kolonne som er bias
    // vi bruger vores hjælpefunktion fra matrix.h

    struct matrix *w_with_bias = expand_matrix_with_col(layer->weights, layer->bias);

    // nu er vi klar til at udregne Z=wx

    struct matrix *wx = matrix_mul(w_with_bias, x_with_bias);
    layer->pre_activation = wx;
    matrix_free(x_with_bias);
    matrix_free(w_with_bias);
    switch (layer->activation_type)
    {
    case ReLu:
    {
        struct matrix *relu_wx = matrix_create(wx->rows, wx->cols);
        for (int i = 0; i < wx->rows; i++)
        {
            relu_wx->elements[i] = relu(wx->elements[i]);
        }
        layer->output = relu_wx;
        return relu_wx;
    }
    case SoftMax:
    {
        struct matrix *sm = softmax(wx);
        layer->output = sm;
        return sm;
    }
    default:
        fprintf(stderr, "Ukendt aktiveringsfunktion\n");
        return NULL;
    }
}

struct matrix *forward_propagation(struct network nn, struct matrix *input)
{
    struct matrix *current = input;

    for (int i = 0; i < nn.antal_layers; i++)
    {
        struct matrix *output =
            compute_layer_output(&nn.layers[i], current);

        current = output;
    }

    return current;
}

int predict(struct matrix *prop_vector)
{
    int arg_max = 0;
    double max = prop_vector->elements[0];
    for (int i = 1; i < prop_vector->rows; i++)
    {
        if (max < prop_vector->elements[i])
        {
            max = prop_vector->elements[i];
            arg_max = i;
        }
    }
    return arg_max;
}

double cross_entropy_one_datapoint(struct matrix *y, struct matrix *y_hat)
{
    if (y->rows != y_hat->rows)
    {
        fprintf(stderr, "not same dims");
        exit(1);
    }
    if (y->cols != 1)
    {
        fprintf(stderr, "too many cols, y");
        exit(1);
    }
    if (y_hat->cols != 1)
    {
        fprintf(stderr, "too many cols, y_hat");
        exit(1);
    }

    int find_one = -1;
    for (int i = 0; i < y->rows; i++)
    {
        if (y->elements[i] == 1)
        {
            find_one = i;
            break;
        }
    }
    if (find_one == -1)
    {
        fprintf(stderr, "target vector contains no 1's");
        exit(1);
    }
    return -log(y_hat->elements[find_one]);
}

double cross_entropy_dataset(struct matrix **y, struct matrix **y_hat, int size_of_dataset)
{
    double loss = 0;

    for (int i = 0; i < size_of_dataset; i++)
    {
        loss += cross_entropy_one_datapoint(y[i], y_hat[i]);
    }
    return loss / size_of_dataset;
}

// backprop
//     δL  / δ z_2
struct matrix *output_gradient(struct matrix *y, struct matrix *y_hat)
{
    if (y->cols != y_hat->cols)
    {
        fprintf(stderr, "cols do not match");
        exit(1);
    }
    if (y->rows != y_hat->rows)
    {
        fprintf(stderr, "rows do not match");
        exit(1);
    }
    struct matrix *y_hat_copy = copy_matrix(y_hat);

    matrix_sub(y_hat_copy, y);
    return y_hat_copy;
}

//    δL / δw2 = δL / δz_2 * h^T
struct matrix *gradient_w(struct matrix *h, struct matrix *grad_z)
{
    if (h->cols != grad_z->cols)
    {
        fprintf(stderr, "cols do not match, gradient_w");
        exit(1);
    }

    struct matrix *h_T = transpose(h);
    struct matrix *grad = matrix_mul(grad_z, h_T);
    matrix_free(h_T);
    return grad;
}

//    δL / δb = δL / δz
struct matrix *gradient_b(struct matrix *grad_z)
{
    return grad_z;
}

//    δL / δh^{(l-1)} = (w^{(l})^T * δL / δz^{(l)}
struct matrix *gradient_h(struct matrix *w, struct matrix *grad_z)
{
    if (w->rows != grad_z->rows)
    {
        fprintf(stderr, "rows do not match, gradient_h\n");
        exit(1);
    }

    struct matrix *w_T = transpose(w);
    struct matrix *grad = matrix_mul(w_T, grad_z);
    matrix_free(w_T);
    return grad;
}

struct matrix *relu_diff(struct matrix *z)
{
    struct matrix *relu_diff_z = matrix_create(z->rows, z->cols);
    for (int i = 0; i < z->rows; i++)
    {
        if (z->elements[i] > 0)
        {
            relu_diff_z->elements[i] = 1;
        }
        else
            relu_diff_z->elements[i] = 0;
    }
    return relu_diff_z;
}
int count_alive_neurons(struct matrix *layer_output)
{
    int count = 0;
    for (int i = 0; i < layer_output->rows * layer_output->cols; i++)
    {
        if (layer_output->elements[i] > 0)
        {
            count++;
        }
    }
    return count;
}

struct matrix *gradient_z(enum act_func activation, struct matrix *grad_h, struct matrix *z)
{
    switch (activation)
    {
    case ReLu:
    {
        struct matrix *ReLu_diff = relu_diff(z);
        if (ReLu_diff->cols != grad_h->cols)
        {
            fprintf(stderr, "forkert dim");
        }
        if (ReLu_diff->rows != grad_h->rows)
        {
            fprintf(stderr, "forkert dim");
        }
        struct matrix *grad_z = matrix_create(grad_h->rows, grad_h->cols);
        for (int i = 0; i < grad_h->rows; i++)
        {
            grad_z->elements[i] = grad_h->elements[i] * ReLu_diff->elements[i];
        }
        matrix_free(ReLu_diff);

        return grad_z;
    }

        // case SoftMax:;
        //     struct matrix *dL_dz = output_gradient(y, y_hat);
        //     return dL_dz;

    default:
        fprintf(stderr, "Ukendt aktiveringsfunktion\n");
        return NULL;
    }
}

void backprop_layer(struct layer *layer, struct matrix *grad_h, struct matrix *y, struct matrix *h_prev)
{
    // lad os først komme igennem activation function:
    // vi tester om det er et output layer
    // så kan vi nemlig bruge softmax + CE tricket
    // ellers ved vi det er relu (for now)
    if (layer->output_layer)
    {
        struct matrix *dL_dZ = output_gradient(y, layer->output);
        // nu udregner vi dL/dW
        struct matrix *dL_dW = gradient_w(h_prev, dL_dZ);

        // udregner bias grad
        struct matrix *dL_db = gradient_b(dL_dZ);

        // udregner det vi sender videre
        struct matrix *dL_dh_prev = gradient_h(layer->weights, dL_dZ);

        layer->weight_gradient = dL_dW;
        layer->bias_gradient = dL_db;
        layer->input_gradient = dL_dh_prev;
    }
    else
    {
        struct matrix *dL_dZ = gradient_z(layer->activation_type, grad_h, layer->pre_activation);

        // nu udregner vi dL/dW
        struct matrix *dL_dW = gradient_w(h_prev, dL_dZ);

        // udregner bias grad
        struct matrix *dL_db = gradient_b(dL_dZ);

        // udregner det vi sender videre
        struct matrix *dL_dh_prev = gradient_h(layer->weights, dL_dZ);

        layer->weight_gradient = dL_dW;
        layer->bias_gradient = dL_db;
        layer->input_gradient = dL_dh_prev;
    }
}

void backward_propagation(struct network nn, struct matrix *y, struct matrix *input)
{
    nn.layers[nn.antal_layers - 1].output_layer = 1;

    backprop_layer(&nn.layers[nn.antal_layers - 1], NULL, y, nn.layers[nn.antal_layers - 2].output);

    for (int i = nn.antal_layers - 2; i >= 0; i--)
    {
        if (i == 0)
        {
            struct matrix *h_prev = input;
            backprop_layer(&nn.layers[i], nn.layers[i + 1].input_gradient, y, h_prev);
        }
        else
        {
            struct matrix *h_prev = nn.layers[i - 1].output;

            backprop_layer(&nn.layers[i], nn.layers[i + 1].input_gradient, y, h_prev);
        }
    }
}

void update_weights_and_bias(struct network nn, double lr)
{
    for (int i = 0; i < nn.antal_layers; i++)
    {
        matrix_scaler(nn.layers[i].weight_gradient, lr);
        matrix_scaler(nn.layers[i].bias_gradient, lr);

        matrix_sub(nn.layers[i].weights, nn.layers[i].weight_gradient);
        matrix_sub(nn.layers[i].bias, nn.layers[i].bias_gradient);
    }
}

struct matrix *one_hot_encode(int label, int rows)
{
    if (label + 1 > rows)
    {
        fprintf(stderr, "label oob\n");
        exit(1);
    }
    struct matrix *vec = matrix_create(rows, 1);
    for (int i = 0; i < rows; i++)
    {
        if (i == label)
        {
            vec->elements[i] = 1;
        }
        else
        {
            vec->elements[i] = 0;
        }
    }
    return vec;
}

void free_forward_pass(struct network nn)
{

    for (int i = 0; i < nn.antal_layers; i++)
    {
        matrix_free(nn.layers[i].pre_activation);
        nn.layers[i].pre_activation = NULL;

        matrix_free(nn.layers[i].output);
        nn.layers[i].output = NULL;
    }
}
void free_backward_pass(struct network nn,
                        struct matrix *input,
                        struct matrix *y)
{
    for (int i = 0; i < nn.antal_layers; i++)
    {

        matrix_free(nn.layers[i].weight_gradient);
        nn.layers[i].weight_gradient = NULL;

        matrix_free(nn.layers[i].bias_gradient);
        nn.layers[i].bias_gradient = NULL;

        matrix_free(nn.layers[i].input_gradient);
        nn.layers[i].input_gradient = NULL;
    }

    matrix_free(input);
    matrix_free(y);
}

double train_one_epoch_minibatch_sgd(struct network nn, int batch_size, double lr, struct matrix *feature_matrix, struct matrix *label_matrix)
{
    // vi opretter først matricer til at indeholde den løbende opdaterende gradient
    struct matrix **weight_gradient_acc =
        malloc(nn.antal_layers * sizeof(struct matrix *));

    for (int i = 0; i < nn.antal_layers; i++)
    {
        weight_gradient_acc[i] =
            matrix_create(nn.layers[i].weights->rows,
                          nn.layers[i].weights->cols);

        zero_matrix(weight_gradient_acc[i]);
    }

    struct matrix **bias_gradient_acc =
        malloc(nn.antal_layers * sizeof(struct matrix *));
    for (int i = 0; i < nn.antal_layers; i++)
    {
        bias_gradient_acc[i] =
            matrix_create(nn.layers[i].bias->rows,
                          nn.layers[i].bias->cols);

        zero_matrix(bias_gradient_acc[i]);
    }

    // nu gennemgår vi hele feature matricen

    double epoch_loss = 0.0;

    for (int i = 0; i < feature_matrix->rows; i += batch_size)
    {
        int current_batch_size = batch_size;

        if (i + batch_size > feature_matrix->rows)
            current_batch_size = feature_matrix->rows - i;

        for (int k = 0; k < current_batch_size; k++)
        {
            int index = i + k;

            struct matrix *row_i = copy_row_i(feature_matrix, index);
            int label_i = label_matrix->elements[index];
            struct matrix *one_hot_encode_label_i = one_hot_encode(label_i, 10);
            // the nn will expect input to be a column therefore we transpose
            struct matrix *input_i = transpose(row_i);
            matrix_free(row_i);

            struct matrix *y_hat = forward_propagation(nn, input_i);
            epoch_loss += cross_entropy_one_datapoint(one_hot_encode_label_i, y_hat);

            backward_propagation(nn, one_hot_encode_label_i, input_i);

            for (int j = 0; j < nn.antal_layers; j++)
            {
                matrix_add_inplace(
                    weight_gradient_acc[j],
                    nn.layers[j].weight_gradient);

                matrix_add_inplace(
                    bias_gradient_acc[j],
                    nn.layers[j].bias_gradient);
            }
            free_forward_pass(nn);
            free_backward_pass(nn, input_i, one_hot_encode_label_i);
            if (index % 5000 == 0)
            {
                printf("inputs processed: %d, progress (for this epoch): %f\n", index, (float)index / (float)feature_matrix->rows);
            }
        }

        for (int l = 0; l < nn.antal_layers; l++)
        {

            matrix_scaler(weight_gradient_acc[l], lr / current_batch_size);
            matrix_scaler(bias_gradient_acc[l], lr / current_batch_size);

            matrix_sub(nn.layers[l].weights,
                       weight_gradient_acc[l]);

            matrix_sub(nn.layers[l].bias,
                       bias_gradient_acc[l]);

            zero_matrix(weight_gradient_acc[l]);
            zero_matrix(bias_gradient_acc[l]);
        }
    }
    for (int i = 0; i < nn.antal_layers; i++)
    {
        matrix_free(weight_gradient_acc[i]);
        matrix_free(bias_gradient_acc[i]);
    }
    free(weight_gradient_acc);
    free(bias_gradient_acc);
    return epoch_loss;
}

void train_network(struct network nn, double lr, struct matrix *feature_matrix, struct matrix *label_matrix, int epochs, int batchsize)
{
    printf("starting training network\n");
    printf("number of epochs: %d\n", epochs);
    printf("number of inputs in every epoch: %d\n", feature_matrix->rows);
    for (int j = 1; j < epochs + 1; j++)
    {
        double epoch_loss = train_one_epoch_minibatch_sgd(nn, batchsize, lr, feature_matrix, label_matrix);
        if (j % 100 == 0)
        {
            printf("epoch %d, avg loss: %f\n", j, epoch_loss / feature_matrix->rows);
            printf("%d out of %d epochs completed\n", j, epochs);
        }
    }
}

struct matrix *make_prediction_vector_nn(struct network nn, struct matrix *test_matrix, struct matrix *label_matrix)
{
    struct matrix *prediction_vector = matrix_create(test_matrix->rows, 1);
    if (label_matrix->cols != prediction_vector->cols || label_matrix->rows != prediction_vector->rows)
    {
        fprintf(stderr, "wroooong dimensions");
    }

    for (int i = 0; i < test_matrix->rows; i++)
    {

        struct matrix *row_i = copy_row_i(test_matrix, i);
        // the nn will expect input to be a column therefore we transpose
        struct matrix *input_i = transpose(row_i);
        matrix_free(row_i);
        struct matrix *prop_vector = forward_propagation(nn, input_i);
        int prediction = predict(prop_vector);
        prediction_vector->elements[i] = prediction;
        free_forward_pass(nn);
        matrix_free(input_i);
    }
    return prediction_vector;
}
