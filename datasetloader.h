#ifndef DATASETLOADER_H
#define DATASETLOADER_H

struct dataset
{
    struct matrix *X;
    struct matrix *y;
};

struct mean_and_std
{
    double *mean;
    double *std;
    int cols;
};

struct dataset x_and_y(struct matrix *mat, int split);
void dataset_free(struct dataset data);

// csv loader virker KUN for header csv filer -> header bliver udeladt
struct matrix *load_from_csv(const char *filename);

char *clean_token(char *token);

// blander din matrice, bruges oftest ifb med test_train splittet
// kan også bruges alene
void shuffle_matrix(struct matrix *x);

// splitter dit datasæt i train og test, splitratio er [0,1] og angiver
// hvor stor del der skal være train, resten er test
struct dataset train_test_split(struct matrix *x, double split_ratio);

// fjerner én kolonne fra din matrice, som angivet med col_number (0-indexed)
struct matrix *rm_col_from_mat(struct matrix *x, int col_number);

void normalize_matrix(struct matrix *mat, int max_value);
struct mean_and_std standardize_matrix(struct matrix *mat);

void standardize_matrix_with_mean_and_std(struct matrix *mat, struct mean_and_std ms);

#endif
