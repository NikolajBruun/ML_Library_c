#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include <math.h>

double ev_accuracy(struct matrix *y_hat, struct matrix *y)
{
    if (y_hat->cols != y->cols || y_hat->rows != y->rows)
    {
        fprintf(stderr, "y hat og y har forskellige dimensioner\n");
        return -1;
    }

    int correct = 0;
    for (int i = 0; i < y->rows; i++)
    {
        if (fabs(y_hat->elements[i] - y->elements[i]) < 0.5)
        {
            correct++;
        }
    }

    double acc = (double)correct / (double)y->rows;
    return acc;
}

double ev_precision(struct matrix *y_hat, struct matrix *y)
{
    if (y_hat->cols != y->cols || y_hat->rows != y->rows)
    {
        fprintf(stderr, "y hat og y har forskellige dimensioner\n");
        return -1;
    }

    int TP = 0;
    int FP = 0;
    for (int i = 0; i < y->rows; i++)
    {
        if (y_hat->elements[i] > 0.5 && y->elements[i] > 0.5)
        {
            TP++;
        }
        if (y_hat->elements[i] > 0.5 && y->elements[i] < 0.5)
        {
            FP++;
        }
    }

    if (TP + FP == 0)
    {
        fprintf(stderr, "Ingen positive forudsigelser - precision udefineret\n");
        return -1; // eller 0.0, alt efter hvad der giver mening i din kontekst
    }
    double prec = (double)TP / (double)(TP + FP);
    return prec;
}

double ev_recall(struct matrix *y_hat, struct matrix *y)
{
    if (y_hat->cols != y->cols || y_hat->rows != y->rows)
    {
        fprintf(stderr, "y hat og y har forskellige dimensioner\n");
        return -1;
    }

    int TP = 0;
    int FN = 0;
    for (int i = 0; i < y->rows; i++)
    {
        if (y_hat->elements[i] > 0.5 && y->elements[i] > 0.5)
        {
            TP++;
        }
        if (y_hat->elements[i] < 0.5 && y->elements[i] > 0.5)
        {
            FN++;
        }
    }

    if (TP + FN == 0)
    {
        fprintf(stderr, "Ingen positive forudsigelser - precision udefineret\n");
        return -1;
    }
    double rec = (double)TP / (double)(TP + FN);
    return rec;
}

double ev_f1(struct matrix *y_hat, struct matrix *y)
{
    double rec = ev_recall(y_hat, y);
    double prec = ev_precision(y_hat, y);

    if (rec == -1 || prec == -1)
    {
        fprintf(stderr, "Kan ikke beregne F1 - precision eller recall er udefineret\n");
        return -1;
    }

    double f1 = 2 * (prec * rec / (prec + rec));
    return f1;
}
