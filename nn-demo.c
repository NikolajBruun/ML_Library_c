#include <stdio.h>
#include "matrix.h"
#include "datasetloader.h"
#include "LinReg.h"
#include "LogReg.h"
#include <math.h>
#include <stdlib.h>
#include "evaluation.h"
#include "nn.h"
#include <time.h>

int main()
{
    unsigned int seed = 42;
    srand(seed);
    // vi loader først dataen ind
    struct matrix *mnist_train = load_from_csv("data/mnist_train.csv");
    struct matrix *mnist_test = load_from_csv("data/mnist_test.csv");

    // struktur i csv er label | feature1 feature2 ... feature784
    // derfor er kolonne 1 vores labels og resten
    struct dataset labels_and_feature_train = x_and_y(mnist_train, 1);
    struct dataset labels_and_feature_test = x_and_y(mnist_test, 1);

    matrix_free(mnist_train);
    matrix_free(mnist_test);

    // x_and_y er lidt bagvendt og regner med feature1 feature2 ... feature784 | label
    // derfor er navngivning i følgende forvirrende
    struct matrix *label_train = labels_and_feature_train.X;
    struct matrix *feature_train = labels_and_feature_train.y;
    normalize_matrix(feature_train, 255);

    struct matrix *label_test = labels_and_feature_test.X;
    struct matrix *feature_test = labels_and_feature_test.y;

    normalize_matrix(feature_test, 255);

    // nu kan vi lave vores NN
    // strukturen bliver 784 -> 128(relu) -> 10 (softmax)
    // altså input-> hidden -> output
    // strukturen i dette netværk medberegner dog ikke input lag som et egentlig lag

    // vi opretter bias som vi sætter til 0.1 for lille positiv impact
    // vægt er er he-init som er bedre for relu
    struct matrix *b_1 = matrix_create(128, 1);
    matrix_fill(b_1, 0.1);

    struct matrix *w_1 = matrix_create(128, 784);
    matrix_random(w_1);
    matrix_scaler(w_1, sqrt(2.0 / 784)); // He-init for ReLU

    struct layer hidden_layer =
        {
            .activation_type = ReLu,
            .bias = b_1,
            .weights = w_1,
            .in_dim = 784,
            .out_dim = 128,
            .output_layer = 0, // nej vores hidden layer er ikke vores output_layer
            .bias_gradient = NULL,
            //.input = NULL,
            .input_gradient = NULL,
            .output = NULL,
            .pre_activation = NULL,
            .weight_gradient = NULL};

    // nu opretter vi bias og vægt for det næste lag, her er bias bare 0
    struct matrix *b_2 = matrix_create(10, 1);
    zero_matrix(b_2);

    struct matrix *w_2 = matrix_create(10, 128);
    matrix_random(w_2);
    matrix_scaler(w_2, sqrt(2.0 / 128));
    struct layer output_layer =
        {
            .activation_type = SoftMax,
            .bias = b_2,
            .bias_gradient = NULL,
            .in_dim = 128,
            //.input = NULL,
            .input_gradient = NULL,
            .out_dim = 10,
            .output = NULL,
            .output_layer = 1, // yes det her er output layer
            .pre_activation = NULL,
            .weight_gradient = NULL,
            .weights = w_2};

    // nu kan vi oprette vores netværk
    struct network nn_mnist;
    nn_mnist.layers = malloc(2 * sizeof(struct layer));

    nn_mnist.layers[0] = hidden_layer;
    nn_mnist.layers[1] = output_layer;

    nn_mnist.antal_layers = 2;

    // netværket er done!!
    // vi vælger følgende hyperparametre
    double lr = 0.1;
    int epochs = 1;
    int batchsize = 100;

    struct matrix *pred_vector_bef = make_prediction_vector_nn(nn_mnist, feature_test, label_test);

    double acc = ev_accuracy(pred_vector_bef, label_test);
    matrix_free(pred_vector_bef);

    printf("accuracy before: %f\n", acc);

    train_network(nn_mnist, lr, feature_train, label_train, epochs, batchsize);

    printf("vi tester accuracy efter træning:\n");

    struct matrix *pred_vector_aft = make_prediction_vector_nn(nn_mnist, feature_test, label_test);

    acc = ev_accuracy(pred_vector_aft, label_test);
    matrix_free(pred_vector_aft);
    printf("accuracy after: %f\n", acc);

    // lad os tilsidst free'e alt vi har oprettet
    matrix_free(w_2);
    matrix_free(w_1);
    matrix_free(b_2);
    matrix_free(b_1);
    dataset_free(labels_and_feature_test);
    dataset_free(labels_and_feature_train);

    free(nn_mnist.layers);

    return 0;
}