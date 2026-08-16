#ifndef LOGREG_H
#define LOGREG_H
double LinComb(struct matrix *w, struct matrix *x, double b);
double sigmoid(double z);
// udregner log-likelihood
// smart så man kan se at gradient descnet virker
// b er intercept,w er vægt vektor
// x er datapunkter hvor datapunkt i er række i
// y er class 0/1
double logistic_log_likelihood(double b, struct matrix *w, struct matrix *x, struct matrix *y);

// udregner blot gradienten, ikke relevant til andet end gradient descent tror jeg
struct matrix *gradient(struct matrix *x, struct matrix *y, struct matrix *w, double b, double *grad_b_out);

// laver gradient descent på hele feature matricen, udfra givet antal iterationer og lr
void gradient_descent(int iter, double lr_rate, struct matrix *x, struct matrix *y, struct matrix *w, double *b);

// laver klassifikation udfra din matrice x (feature vektoren for 1 datapunkt) og w (vægt vektoren) og b (bias)
int log_reg_predict_datapoint(struct matrix *x, struct matrix *w, double b);

// predicter ud fra en hel matrice X (ikke blot 1 datapunkt)
struct matrix *log_reg_predict(struct matrix *X, struct matrix *w, double b);

#endif
