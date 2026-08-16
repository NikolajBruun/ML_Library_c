#include <stdio.h>
#include "matrix.h"
#include "datasetloader.h"
#include "LinReg.h"
#include "LogReg.h"
#include <math.h>
#include <stdlib.h>
#include "evaluation.h"

int main()
{
    // lad os først gøre matricen klar til brug

    struct matrix *mat_with_ids = load_from_csv("data/wdbc.data");
    // print_matrix(mat_with_ids);
    struct matrix *wdbc_uden_id = rm_col_from_mat(mat_with_ids, 0);

    struct dataset train_test = train_test_split(wdbc_uden_id, 0.8);

    struct matrix *train_mat = train_test.X;
    struct matrix *test_mat = train_test.y;

    struct dataset train_tar_and_fea = x_and_y(train_mat, 1);
    struct dataset test_tar_and_fea = x_and_y(test_mat, 1);

    struct matrix *train_target = train_tar_and_fea.X;
    struct matrix *train_features = train_tar_and_fea.y;

    // nu standardisere vi train feature matricen og funktionen returnere også det vi skal bruge
    struct mean_and_std mean_std = standardize_matrix(train_features);

    struct matrix *test_target = test_tar_and_fea.X;
    struct matrix *test_features = test_tar_and_fea.y;

    // nu standardisere vi test feature matricen
    standardize_matrix_with_mean_and_std(test_features, mean_std);
    free(mean_std.mean);
    free(mean_std.std);

    // lad os nu oprette vægtmatricen
    struct matrix *w = matrix_create(train_features->cols, 1);
    matrix_random(w);
    matrix_scaler(w, 0.01);

    double b = 0;

    struct matrix *row_i = copy_row_i(train_features, 0);

    matrix_free(row_i);

    gradient_descent(80000, 0.005, train_features, train_target, w, &b);

    struct matrix *y_hat = log_reg_predict(test_features, w, b);

    double acc = ev_accuracy(y_hat, test_target);

    double prec = ev_precision(y_hat, test_target);
    double f1 = ev_f1(y_hat, test_target);
    // for (int i = 0; i < test_features->rows; i++)
    // {
    //     struct matrix *row_i = copy_row_i(test_features, i);
    //     errors += abs((int)log_reg_predict(row_i, w, b) - (int)test_target->elements[i]);
    //     matrix_free(row_i);
    // }
    printf("accuracy %f \n", acc);

    printf("precision %f \n", prec);
    printf("f1 %f \n", f1);

    // print_matrix(y_hat);
    matrix_free(y_hat);

    matrix_free(w);
    matrix_free(wdbc_uden_id);
    matrix_free(mat_with_ids);
    dataset_free(train_tar_and_fea);
    dataset_free(test_tar_and_fea);
    matrix_free(train_mat);
    matrix_free(test_mat);

    return 0;
}
