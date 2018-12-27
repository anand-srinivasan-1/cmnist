#include "matrix.h"

#include <math.h>

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

void mat_mul(mlmatrix *dst, mlmatrix *x, mlmatrix *y) {
    int a = x->rows;
    int b = x->cols;
    int c = y->cols;
    for(int i = 0; i < a; i++) {
        for(int j = 0; j < c; j++) {
            float sum = 0.0f;
            for(int k = 0; k < b; k++) {
                sum += x->data[i*b + k] * y->data[k*c + j];
            }
            dst->data[i*c + j] = sum;
        }
    }
}

void mat_add(mlmatrix *dst, mlmatrix *x, mlmatrix *y) {
    for(int i = 0; i < dst->rows*dst->cols; i++) {
        dst->data[i] = x->data[i] + y->data[i];
    }
}

void mat_sub(mlmatrix *dst, mlmatrix *x, mlmatrix *y) {
    for(int i = 0; i < dst->rows*dst->cols; i++) {
        dst->data[i] = x->data[i] - y->data[i];
    }
}

void mat_copy(mlmatrix *dst, mlmatrix *src) {
    for(int i = 0; i < dst->rows*dst->cols; i++) {
        dst->data[i] = src->data[i];
    }
}

void mat_transpose(mlmatrix *dst, mlmatrix *src) {
    for(int i = 0; i < dst->rows; i++) {
        for(int j = 0; j < dst->cols; j++) {
            dst->data[i*dst->cols + j] = src->data[j*src->cols + i];
        }
    }
}

void mat_nl(mlmatrix *dst, mlmatrix *src) {
    for(int i = 0; i < dst->rows*dst->cols; i++) {
        dst->data[i] = sigmoid(src->data[i]);
    }
}

void mat_invnl(mlmatrix *dx, mlmatrix *y, mlmatrix *dy) {
    // dx = dy * NL(x) * (1 - NL(x)) = dy * y * (1 - y)
    for(int i = 0; i < dx->rows*dx->cols; i++) {
        dx->data[i] = dy->data[i] * y->data[i] * (1.0f - y->data[i]);
    }
}

void mat_param_update(mlmatrix *m, mlmatrix *deriv, float alpha) {
    for(int i = 0; i < m->rows*m->cols; i++) {
        m->data[i] -= alpha * deriv->data[i];
    }
}
