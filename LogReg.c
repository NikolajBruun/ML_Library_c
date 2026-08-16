#include "datasetloader.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include "math.h"

double LinComb(struct matrix *w, struct matrix *x, double b)
{
    struct matrix *z_1x1 = matrix_mul(x, w);
    double z = z_1x1->elements[0];
    matrix_free(z_1x1);
    return z + b;
}
double sigmoid(double z)
{
    return 1 / (1 + exp(-z));
}

double logistic_log_likelihood(double b, struct matrix *w, struct matrix *x, struct matrix *y)
{

    int n = x->rows;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {

        // first half of log-likelihood
        //-log(1+exp(w_xi+b))
        struct matrix *x_i = copy_row_i(x, i);
        struct matrix *w_xi = matrix_mul(x_i, w);
        double lin_comb = w_xi->elements[0] + b;
        double log_part = log(1.0 + exp(lin_comb));
        sum -= log_part;

        // second part
        // y_i(w_xi+b)
        double y_i = y->elements[i];
        double prod = y_i * lin_comb;

        sum += prod;
        matrix_free(x_i);
        matrix_free(w_xi);
    }

    return sum;
}
//\sum_i=1^n(y_i-p_i)x_i
// svare til X^T(p-y)
// passer med J(B) = -l(b)
struct matrix *gradient(struct matrix *x, struct matrix *y, struct matrix *w, double b, double *grad_b_out)
{
    if (y->rows != x->rows)
    {
        return NULL;
    }
    struct matrix *X_T = transpose(x);

    // lad os udregne p-vektoren
    struct matrix *p = matrix_create(x->rows, 1);
    double grad_b = 0.0;
    for (int i = 0; i < x->rows; i++)
    {
        struct matrix *row_i = copy_row_i(x, i);
        double lin_comb = LinComb(w, row_i, b);
        p->elements[i] = sigmoid(lin_comb);
        matrix_free(row_i);

        grad_b += p->elements[i] - y->elements[i];
    }

    matrix_sub(p, y);
    struct matrix *gradient = matrix_mul(X_T, p);
    matrix_free(p);
    matrix_free(X_T);
    *grad_b_out = grad_b;
    return gradient;
}

void gradient_descent(int iter, double lr_rate, struct matrix *x, struct matrix *y, struct matrix *w, double *b)
{
    printf("init log-likeli: %f\n", logistic_log_likelihood(*b, w, x, y));
    // w_new = w_old - lr * grad
    for (int i = 0; i < iter; i++)
    {
        double grad_b = 0.0;
        struct matrix *grad = gradient(x, y, w, *b, &grad_b);
        matrix_scaler(grad, lr_rate);

        for (int j = 0; j < w->rows; j++)
            w->elements[j] -= grad->elements[j];

        *b -= lr_rate * grad_b;
        matrix_free(grad);

        if (i % 1000 == 0)
        {
            printf("iteration: %d, log-likeli: %f\n", i, logistic_log_likelihood(*b, w, x, y));
        }
    }
}

int log_reg_predict_datapoint(struct matrix *x, struct matrix *w, double b)
{
    double z = LinComb(w, x, b);
    double sig = sigmoid(z);
    if (sig >= 0.5)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

struct matrix *log_reg_predict(struct matrix *X, struct matrix *w, double b)
{
    struct matrix *y_hat = matrix_create(X->rows, 1);

    for (int i = 0; i < X->rows; i++)
    {
        struct matrix *row_i = copy_row_i(X, i);
        y_hat->elements[i] = log_reg_predict_datapoint(row_i, w, b);
        matrix_free(row_i);
    }
    return y_hat;
}
