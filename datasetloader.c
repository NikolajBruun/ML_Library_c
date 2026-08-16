#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "matrix.h"
#include "datasetloader.h"

struct dataset x_and_y(struct matrix *mat, int split)
{
    int cols = mat->cols;
    int rows = mat->rows;

    struct matrix *feature = matrix_create(rows, split);
    struct matrix *target = matrix_create(rows, cols - split);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (j < split)
            {
                feature->elements[split * i + j] =
                    mat->elements[cols * i + j];
            }
            else
            {
                target->elements[(cols - split) * i + (j - split)] =
                    mat->elements[cols * i + j];
            }
        }
    }

    struct dataset target_and_feature = {feature, target};
    return target_and_feature;
}

void dataset_free(struct dataset data)
{
    matrix_free(data.X);
    matrix_free(data.y);
}

char *clean_token(char *token)
{
    if (token[0] == '"')
    {
        token++;
    }

    int n = strlen(token);

    if (n > 0 && token[n - 1] == '"')
    {
        token[n - 1] = '\0';
    }

    return token;
}

struct matrix *load_from_csv(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        return NULL;
    }
    // random længde line -> skal bare være stor nok -> ingen dynamisk allokering nødvendig
    char line[10000];
    int number_col = 0;
    int number_row = 0;
    // vi læser headeren og bruger denne til at tælle linjer
    fgets(line, sizeof(line), file);
    char *container = strtok(line, ",");

    while (container != NULL)
    {
        number_col++;
        container = strtok(NULL, ",");
    }
    // tæler antal kolonner;
    // tæller bare antal rækker med simple increment
    while (fgets(line, sizeof(line), file))
    {
        number_row++;
    }

    // flytter læsehovedet tilbage til toppen
    rewind(file);
    // nedenstående er anden iteration gennem csv-filen
    struct matrix *csv_mat = matrix_create(number_row, number_col);
    // int i til at tælle række nr
    // int j til at tælle kolonne nr
    int i = 0;
    int j = 0;
    // lad os lige skipper headeren igen
    fgets(line, sizeof(line), file);
    // nu kan vi indsætte i matrice
    while (fgets(line, sizeof(line), file))
    {
        // læs hver værdi -> bliver som string
        char *token = strtok(line, ",");

        // gennemgår 1 værdi ad gangen
        while (token != NULL)
        {
            // tester lige om token er "clean" -> dvs ikke "1" men blot 1
            token = clean_token(token);

            // konverter string til double
            double x = strtod(token, NULL);

            csv_mat->elements[number_col * i + j] = x;
            token = strtok(NULL, ",");
            j++;
        }
        j = 0;
        i++;
    }
    fclose(file);
    return csv_mat;
}

void shuffle_matrix(struct matrix *x)
{
    // vi shuffler ved brug af fisher-yates
    // implementeret som jeg har forstået den
    int rows = x->rows;
    int cols = x->cols;
    for (int i = rows - 1; i >= 1; i--)
    {
        struct matrix *row_i = copy_row_i(x, i);
        int r = rand() % (i + 1);
        struct matrix *row_r = copy_row_i(x, r);

        for (int j = 0; j < cols; j++)
        {
            x->elements[cols * r + j] = row_i->elements[j];
            x->elements[cols * i + j] = row_r->elements[j];
        }

        matrix_free(row_i);
        matrix_free(row_r);
    }
}

struct dataset train_test_split(struct matrix *x, double split_ratio)
{
    int rows = x->rows;
    int cols = x->cols;

    int train_rows = rows * split_ratio;
    int test_rows = rows - train_rows;

    struct matrix *train = matrix_create(train_rows, cols);
    struct matrix *test = matrix_create(test_rows, cols);
    shuffle_matrix(x);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (i < train_rows)
            {
                train->elements[cols * i + j] = x->elements[cols * i + j];
            }

            else
            {
                test->elements[cols * (i - train_rows) + j] = x->elements[cols * i + j];
            }
        }
    }
    struct dataset train_and_test = {train, test};

    return train_and_test;
}

struct matrix *rm_col_from_mat(struct matrix *x, int col_number)
{
    int rows = x->rows;
    int cols = x->cols;
    int new_cols = cols - 1;

    struct matrix *new_mat = matrix_create(rows, new_cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (j < col_number)
            {
                new_mat->elements[new_cols * i + j] = x->elements[cols * i + j];
            }
            if (j > col_number)
            {
                new_mat->elements[new_cols * i + (j - 1)] = x->elements[cols * i + j];
            }
        }
    }
    return new_mat;
}

void normalize_matrix(struct matrix *mat, int max_value)
{
    for (int i = 0; i < mat->cols * mat->rows; i++)
    {
        mat->elements[i] = mat->elements[i] / max_value;
    }
}

struct mean_and_std standardize_matrix(struct matrix *mat)
{
    struct mean_and_std ms;
    ms.cols = mat->cols;
    ms.mean = malloc(mat->cols * sizeof(double));
    ms.std = malloc(mat->cols * sizeof(double));

    for (int i = 0; i < mat->cols; i++)
    {
        double counter = 0;
        for (int j = 0; j < mat->rows; j++)
            counter += mat->elements[mat->cols * j + i];
        double mean = counter / mat->rows;

        double sum = 0;
        for (int l = 0; l < mat->rows; l++)
            sum += pow(mat->elements[mat->cols * l + i] - mean, 2);
        double std = sqrt(sum / mat->rows);

        ms.mean[i] = mean;
        ms.std[i] = std;

        for (int h = 0; h < mat->rows; h++)
            mat->elements[mat->cols * h + i] = (mat->elements[mat->cols * h + i] - mean) / std;
    }
    return ms;
}

void standardize_matrix_with_mean_and_std(struct matrix *mat, struct mean_and_std ms)
{
    for (int i = 0; i < mat->rows; i++) // række
    {
        for (int j = 0; j < mat->cols; j++) // kolonne
        {
            mat->elements[mat->cols * i + j] =
                (mat->elements[mat->cols * i + j] - ms.mean[j]) / ms.std[j];
        }
    }
}