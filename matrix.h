#ifndef MATRIX_H
#define MATRIX_H

struct matrix
{
    int rows;
    int cols;
    double *elements;
};

struct matrix *matrix_create(int rows, int cols);
struct matrix *simple_matrix();

// returnere random double [-1,1]
double rand_double_minus1_plus1();

// random matrice med elementer [-1,1]
void matrix_random(struct matrix *mat);
// prints the matrix nicely to terminal
void print_matrix(struct matrix *mat);
// frees both elements and matrix stucture
void matrix_free(struct matrix *mat);
// sets all elements in matrix to 0
void zero_matrix(struct matrix *mat);
// adds matrix a with matrix b
struct matrix *matrix_add(struct matrix *a, struct matrix *b);
// does so in place for a
void matrix_add_inplace(struct matrix *a, struct matrix *b);

// subtracts matrix b from matrix a
void matrix_sub(struct matrix *a, struct matrix *b);

// a er nxm b er mxl
struct matrix *matrix_mul(struct matrix *a, struct matrix *b);
struct matrix *transpose(struct matrix *mat);
void matrix_scaler(struct matrix *mat, double scalar);
int gauss_jordan(struct matrix *mat);
int inverse(struct matrix *mat);
// er brugt til at kopiere række i af en matrice X
// j er det række nr man ønsker kopieret
struct matrix *copy_row_i(struct matrix *x, int j);

// udvideer matrice x med kolonnen col
// bruges til at gøre plads til bias i vægtmatrice
struct matrix *expand_matrix_with_col(struct matrix *x, struct matrix *col);

struct matrix *copy_matrix(struct matrix *m);

void matrix_fill(struct matrix *mat, double value);

#endif