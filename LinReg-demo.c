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
    // først indlæser vi datasæt til matrice og splitter det som x og y
    struct matrix *BH_matrix = load_from_csv("data/BostonHousing.csv");

    struct dataset features_and_medv = x_and_y(BH_matrix, BH_matrix->cols - 1);
    printf("rows:%d, cols: %d\n", features_and_medv.y->rows, features_and_medv.y->cols);

    // vi laver analytisk lin-reg og får vægtmatrice:
    struct matrix *lin_reg = LinearRegression(features_and_medv);

    // nedenstående burde være nogenlunde ens -> er egentlig bare et sanity check
    struct cmp_vector est_xty = test_LinReg(features_and_medv, lin_reg);

    printf("nedenfor burde vi få cirka 0 vektor\n");
    matrix_sub(est_xty.m1, est_xty.m2);
    print_matrix(est_xty.m1);

    matrix_free(est_xty.m1);
    matrix_free(est_xty.m2);
    matrix_free(lin_reg);
    dataset_free(features_and_medv);
    matrix_free(BH_matrix);

    return 0;
}
