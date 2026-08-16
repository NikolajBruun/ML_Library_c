#ifndef EVALUATION_H
#define EVALUATION_H

// udregner accuracy mængden af korrekt / alle
double ev_accuracy(struct matrix *y_hat, struct matrix *y);

// udregner precision TP/(TP+FP), positive er assumet 1
double ev_precision(struct matrix *y_hat, struct matrix *y);

// udregner recall TP/(TP+FN)
double ev_recall(struct matrix *y_hat, struct matrix *y);

double ev_f1(struct matrix *y_hat, struct matrix *y);

#endif