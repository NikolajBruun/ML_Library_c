#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "datasetloader.h"
#include "LinReg.h"

struct matrix *LinearRegression(struct dataset data)
{
    struct matrix *X = data.X;
    struct matrix *y = data.y;

    struct matrix *X_T = transpose(X);
    struct matrix *square = matrix_mul(X_T, X);

    int inv = inverse(square);
    // tilføj error handling her ->
    // if !=0 report fejl og hvofor det sker

    if (inv)
    {
        fprintf(stderr, "fejl i invers");
        return NULL;
    }
    struct matrix *X_t_y = matrix_mul(X_T, y);
    struct matrix *w_hat = matrix_mul(square, X_t_y);

    // lad os free alt intermediate
    matrix_free(X_t_y);
    matrix_free(square);
    matrix_free(X_T);
    return w_hat;
}

struct cmp_vector test_LinReg(struct dataset data, struct matrix *w_hat)
{
    // xTxw
    struct matrix *X = data.X;
    struct matrix *y = data.y;
    struct matrix *trans = transpose(X);
    struct matrix *xT_x = matrix_mul(trans, X);
    struct matrix *est = matrix_mul(xT_x, w_hat);

    // xTy
    struct matrix *xTy = matrix_mul(trans, y);

    struct cmp_vector cmp = {est, xTy};

    matrix_free(trans);
    matrix_free(xT_x);
    return cmp;
}