typedef struct {
    int rows;
    int cols;
    float *data;
} mlmatrix;

void mat_mul(mlmatrix *dst, mlmatrix *x, mlmatrix *y);
void mat_add(mlmatrix *dst, mlmatrix *x, mlmatrix *y);
void mat_sub(mlmatrix *dst, mlmatrix *x, mlmatrix *y);
void mat_transpose(mlmatrix *dst, mlmatrix *src);
void mat_copy(mlmatrix *dst, mlmatrix *src);
void mat_nl(mlmatrix *dst, mlmatrix *src);
void mat_invnl(mlmatrix *dx, mlmatrix *y, mlmatrix *dy);
void mat_param_update(mlmatrix *m, mlmatrix *deriv, float alpha);
