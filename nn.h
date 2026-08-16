#ifndef NN_H
#define NN_H

enum act_func
{
    ReLu,
    SoftMax
};
struct network
{
    int antal_layers;
    struct layer *layers;
};
struct layer
{
    // dimensions of layer
    int in_dim;
    int out_dim;

    // weights and bias
    struct matrix *weights;
    struct matrix *bias;

    // type of activation function
    enum act_func activation_type;

    // saved values from forward prop.
    // truct matrix *input;
    struct matrix *output;
    struct matrix *pre_activation;

    // gradients from backprop
    struct matrix *weight_gradient;
    struct matrix *bias_gradient;
    struct matrix *input_gradient;

    // for om jeg kan bruge softmax+ cross entropy tricket
    int output_layer;
};

// laver forward prop.
struct matrix *forward_propagation(struct network nn, struct matrix *input);

// laver prediction udfra prop vector dvcs fx fra softmax vector
int predict(struct matrix *prop_vector);

// træner hele netværk med forward og backward prop
void train_network(struct network nn, double lr, struct matrix *feature_matrix, struct matrix *label_matrix, int epochs, int batchsize);

struct matrix *make_prediction_vector_nn(struct network nn, struct matrix *test_matrix, struct matrix *label_matrix);

// void gradient_check(struct network nn, struct matrix *input, struct matrix *y, int num_checks_per_layer);

// struct matrix *one_hot_encode(int label, int rows);
int count_alive_neurons(struct matrix *layer_output);
void free_forward_pass(struct network nn);

#endif