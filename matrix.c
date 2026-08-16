#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"

// create:
// ide er bare at initialisere plads til matricen
// ved ikke om det skal være sizeof double eller sizeofmatrix
struct matrix *matrix_create(int rows, int cols)
{
    if (rows <= 0 || cols <= 0)
    {
        fprintf(stderr, "rows or columns must be positive");
        return NULL;
    }
    // først allokere vi plads til strukturen
    struct matrix *mat = malloc(sizeof(struct matrix));
    if (mat == NULL)
        return NULL;

    // dernæst allokere vi plads til elementerne i strukturen:
    mat->elements = malloc(rows * cols * sizeof(double));
    if (mat->elements == NULL)
    {
        free(mat);
        return NULL;
    }
    mat->cols = cols;
    mat->rows = rows;
    return mat;
}

struct matrix *simple_matrix()
{
    struct matrix *m = matrix_create(2, 2);
    m->elements[0] = 1;
    m->elements[1] = 2;
    m->elements[2] = 1;
    m->elements[3] = 3;
    return m;
}

void matrix_free(struct matrix *mat)
{
    free(mat->elements);
    free(mat);
}

double rand_double_minus1_plus1()
{
    double r = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
    return r;
}
void matrix_random(struct matrix *mat)
{
    for (int i = 0; i < mat->rows * mat->cols; i++)
    {
        double r = rand_double_minus1_plus1();
        mat->elements[i] = r;
    }
}

void print_matrix(struct matrix *mat)
{
    for (int i = 0; i < mat->rows; i++)
    {
        printf("[ ");

        for (int j = 0; j < mat->cols; j++)
        {
            printf("%8.3f ", mat->elements[i * mat->cols + j]);
        }

        printf("]\n");
    }
    printf("\n");
}

void zero_matrix(struct matrix *mat)
{
    for (int i = 0; i < mat->rows * mat->cols; i++)
    {

        mat->elements[i] = 0;
    }
}

struct matrix *matrix_add(struct matrix *a, struct matrix *b)
{
    if (a->cols != b->cols || a->rows != b->rows)
    {
        fprintf(stderr, "matricer ikke kompatible");
        return NULL;
    }

    struct matrix *res = matrix_create(a->rows, a->cols);
    for (int i = 0; i < a->cols * a->rows; i++)
    {
        res->elements[i] = a->elements[i] + b->elements[i];
    }
    return res;
}

void matrix_add_inplace(struct matrix *a, struct matrix *b)
{
    if (a->cols != b->cols || a->rows != b->rows)
    {
        fprintf(stderr, "matricer ikke kompatible");
        exit(1);
    }
    for (int i = 0; i < a->cols * a->rows; i++)
    {
        a->elements[i] += b->elements[i];
    }
}

void matrix_sub(struct matrix *a, struct matrix *b)
{
    if (a->cols != b->cols || a->rows != b->rows)
    {
        fprintf(stderr, "matricer ikke kompatible");
        return;
    }

    for (int i = 0; i < a->cols * a->rows; i++)
    {
        a->elements[i] -= b->elements[i];
    }
}

struct matrix *matrix_mul(struct matrix *a, struct matrix *b)
{
    if (a->cols != b->rows)
    {
        fprintf(stderr, "matrices not compatable");
        return NULL;
    }
    struct matrix *res = matrix_create(a->rows, b->cols);
    // looper over rækker i a
    for (int i = 0; i < a->rows; i++)
    {
        // looper over kolonner i b
        for (int j = 0; j < b->cols; j++)
        {
            // laver sum
            double sum = 0.0;
            for (int k = 0; k < a->cols; k++)
            {
                sum += a->elements[a->cols * i + k] * b->elements[b->cols * k + j];
            }
            res->elements[res->cols * i + j] = sum;
        }
    }
    return res;
}

struct matrix *transpose(struct matrix *mat)
{
    struct matrix *trans = matrix_create(mat->cols, mat->rows);
    for (int i = 0; i < mat->rows; i++)
    {
        for (int j = 0; j < mat->cols; j++)
        {
            trans->elements[trans->cols * j + i] = mat->elements[mat->cols * i + j];
        }
    }
    return trans;
}

void matrix_scaler(struct matrix *mat, double scalar)
{
    for (int i = 0; i < mat->rows * mat->cols; i++)
    {
        mat->elements[i] *= scalar;
    }
}

int gauss_jordan(struct matrix *mat)
{
    int cols = mat->cols;
    int rows = mat->rows;
    // først finder vi minimum af række og kollonner, dette er antal af totale iterationer i ydre loop
    int min = cols;
    if (rows < cols)
    {
        min = rows;
    }
    for (int i = 0; i < min; i++)
    {
        // vi finder pivotrække og pivot element
        double pivot = 0.0;
        int pivot_row = i;

        for (int j = i; j < rows; j++)
        {
            if (fabs(mat->elements[cols * j + i]) > fabs(pivot))
            {
                pivot = mat->elements[cols * j + i];
                pivot_row = j;
            }
        }
        // printf("i = %d, pivot = %.15e\n", i, pivot);
        if (fabs(pivot) <= 1e-12)
        {
            fprintf(stderr, "matricen er singulær\n");
            return -1;
        }

        // vi swapper rækker dvs pivotrow <=> row i
        double tmp_ele;
        for (int k = 0; k < cols; k++)
        {
            tmp_ele = mat->elements[cols * i + k];
            mat->elements[cols * i + k] = mat->elements[cols * pivot_row + k];
            mat->elements[cols * pivot_row + k] = tmp_ele;
        }

        // normalisere row i (dvs vores tidligere valgte pivotrække)
        double pivot_val = mat->elements[cols * i + i];
        for (int k = 0; k < cols; k++)
        {
            mat->elements[cols * i + k] /= pivot_val;
        }

        // eliminér andre rækker dvs andre elementer i pivotkolonnen bliver 0

        for (int k = 0; k < rows; k++)
        {
            if (k == i)
            {
                continue;
            }

            double scalar = mat->elements[cols * k + i];
            for (int l = 0; l < cols; l++)
            {
                mat->elements[cols * k + l] -= scalar * mat->elements[cols * i + l];
            }
        }
    }
    return 0;
}

int inverse(struct matrix *mat)
{
    int cols = mat->cols;
    int rows = mat->rows;
    if (cols != rows)
    {
        fprintf(stderr, "matricen skal være kvardratisk");
        return -1;
    }
    struct matrix *exp_matrix = matrix_create(rows, 2 * cols);

    // oprette [A|I]
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {

            exp_matrix->elements[(2 * cols) * i + j] = mat->elements[cols * i + j];
        }
        for (int j = cols; j < 2 * cols; j++)
        {
            if (j == cols + i)
            {
                exp_matrix->elements[(2 * cols) * i + j] = 1;
            }
            else
            {
                exp_matrix->elements[(2 * cols) * i + j] = 0;
            }
        }
    }

    // lav gauss jordan på denne
    int fail = gauss_jordan(exp_matrix);
    if (fail)
    {
        return -2;
    }

    // okay nu er følgende sket [A|I]-> [I|A^-1]
    // vi skal bare extracte A^-1
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            mat->elements[cols * i + j] = exp_matrix->elements[(2 * cols * i) + j + cols];
        }
    }
    matrix_free(exp_matrix);
    return 0;
}

struct matrix *copy_row_i(struct matrix *x, int j)
{
    // int rows = x->rows;
    int cols = x->cols;
    struct matrix *row = matrix_create(1, cols);
    for (int i = 0; i < cols; i++)
    {
        row->elements[i] = x->elements[cols * j + i];
    }
    return row;
}

struct matrix *expand_matrix_with_col(struct matrix *x, struct matrix *col)
{
    if (col->cols != 1)
    {
        fprintf(stderr, "col is not a column");
        exit(1);
    }
    if (col->rows != x->rows)
    {
        fprintf(stderr, "dim er forkerte");
        exit(1);
    }

    struct matrix *expanded_matrix = matrix_create(x->rows, x->cols + 1);
    for (int i = 0; i < x->rows; i++)
    {
        for (int j = 0; j < expanded_matrix->cols; j++)
        {
            if (j < x->cols)
            {
                expanded_matrix->elements[expanded_matrix->cols * i + j] = x->elements[x->cols * i + j];
            }
            else
            {
                expanded_matrix->elements[expanded_matrix->cols * i + j] = col->elements[i];
            }
        }
    }
    return expanded_matrix;
}

struct matrix *copy_matrix(struct matrix *m)
{
    struct matrix *copy = matrix_create(m->rows, m->cols);

    for (int i = 0; i < m->cols * m->rows; i++)
    {
        copy->elements[i] = m->elements[i];
    }
    return copy;
}
void matrix_fill(struct matrix *mat, double value)
{
    for (int i = 0; i < mat->rows * mat->cols; i++)
    {
        mat->elements[i] = value;
    }
}