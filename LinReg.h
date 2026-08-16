#ifndef LINREG_H
#define LINREG_H
struct cmp_vector
{
    struct matrix *m1;
    struct matrix *m2;
};
struct matrix *LinearRegression(struct dataset data);

// m1 returneret er X^TXw og m2 er X^Ty
struct cmp_vector test_LinReg(struct dataset data, struct matrix *w_hat);

#endif