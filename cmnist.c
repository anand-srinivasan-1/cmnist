/*
This program is a simple program that trains a neural network to recognize handwritten digits, demonstrating that there is no need for a massive library to do machine learning.

It loads the training and test datasets, and repeatedly calculates the network's output for a randomly selected sample. It then calculates the difference between the actual output and the expected output, and slightly tweaks every model parameter to bring the expected and actual outputs slightly closer together.

After enough iterations of this, the model predicts the correct digit more than 95% of the time.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "matrix.h"

#define LEARNING_RATE 0.1f
#define EPOCHS 30

typedef enum {
    INPUT_NODE,
    PARAM_NODE,
    MUL_NODE,
    ADD_NODE,
    NL_NODE,
} node_type;

typedef struct node node;
struct node {
    node_type type;
    node *left;
    node *right;
    mlmatrix matrix;
    mlmatrix deriv;
};

mlmatrix make_matrix(int rows, int cols) {
    float *m = malloc(rows * cols * sizeof(float));
    for(int i = 0; i < rows*cols; i++) {
        m[i] = 0.0f;
    }
    return (mlmatrix) {
        .rows = rows,
        .cols = cols,
        .data = m,
    };
}

mlmatrix random_matrix(int rows, int cols) {
    float *m = malloc(rows * cols * sizeof(float));
    for(int i = 0; i < rows*cols; i++) {
        m[i] = (rand()/(float)RAND_MAX) - 0.5f;
    }
    return (mlmatrix) {
        .rows = rows,
        .cols = cols,
        .data = m,
    };

}

void free_matrix(mlmatrix m) {
    free(m.data);
}

void onehot_encode(mlmatrix *m, int n) {
    for(int i = 0; i < m->rows*m->cols; i++) {
        m->data[i] = 0.0f;
    }
    m->data[n] = 1.0f;
}

node make_input_node(int rows, int cols) {
    return (node) {
        .type = INPUT_NODE,
        .left = NULL,
        .right = NULL,
        .matrix = make_matrix(rows, cols),
        .deriv = make_matrix(rows, cols),
    };
}

node make_param_node(int rows, int cols) {
    return (node) {
        .type = PARAM_NODE,
        .left = NULL,
        .right = NULL,
        .matrix = random_matrix(rows, cols),
        .deriv = make_matrix(rows, cols),
    };
}

node make_mul_node(node *left, node *right) {
    return (node) {
        .type = MUL_NODE,
        .left = left,
        .right = right,
        .matrix = make_matrix(left->matrix.rows, right->matrix.cols),
        .deriv = make_matrix(left->matrix.rows, right->matrix.cols),
    };
}

node make_add_node(node *left, node *right) {
    return (node) {
        .type = ADD_NODE,
        .left = left,
        .right = right,
        .matrix = make_matrix(left->matrix.rows, left->matrix.cols),
        .deriv = make_matrix(left->matrix.rows, left->matrix.cols),
    };
}

node make_nl_node(node *x) {
    return (node) {
        .type = NL_NODE,
        .left = x,
        .right = NULL,
        .matrix = make_matrix(x->matrix.rows, x->matrix.cols),
        .deriv = make_matrix(x->matrix.rows, x->matrix.cols),
    };
}

void free_all(node *n) {
    free_matrix(n->matrix);
    free_matrix(n->deriv);
    if(n->left != NULL) {
        free_all(n->left);
    }
    if(n->right != NULL) {
        free_all(n->right);
    }
}

void load_input(node *n, uint8_t *dataset, int idx) {
    for(int i = 0; i < 784; i++) {
        n->matrix.data[i] = dataset[idx*784 + i] / 255.0f;
    }
}

int interpret_result(node *n) {
    int max_idx = 0;
    for(int i = 0; i < n->matrix.rows*n->matrix.cols; i++) {
        if(n->matrix.data[i] > n->matrix.data[max_idx]) {
            max_idx = i;
        }
    }

    return max_idx;
}

void forward_prop(node *n) {
    if(n == NULL) {
        return;
    }

    forward_prop(n->left);
    forward_prop(n->right);

    switch(n->type) {
        case INPUT_NODE:
            return;
        case PARAM_NODE:
            return;
        case MUL_NODE:
            mat_mul(&n->matrix, &n->left->matrix, &n->right->matrix);
            return;
        case ADD_NODE:
            mat_add(&n->matrix, &n->left->matrix, &n->right->matrix);
            return;
        case NL_NODE:
            mat_nl(&n->matrix, &n->left->matrix);
            return;
    }
}

void backprop(node *n) {
    // assume the derivatives of this node have already been calculated
    switch(n->type) {
        case INPUT_NODE:
            return;
        case PARAM_NODE:
            return;
        case MUL_NODE: {
            mlmatrix LT = make_matrix(n->left->matrix.cols, n->left->matrix.rows);
            mlmatrix RT = make_matrix(n->right->matrix.cols, n->right->matrix.rows);
            mat_transpose(&LT, &n->left->matrix);
            mat_transpose(&RT, &n->right->matrix);
            mat_mul(&n->left->deriv, &n->deriv, &RT);
            mat_mul(&n->right->deriv, &LT, &n->deriv);
            free_matrix(LT);
            free_matrix(RT);
            backprop(n->left);
            backprop(n->right);
            }
            return;
        case ADD_NODE:
            mat_copy(&n->left->deriv, &n->deriv);
            mat_copy(&n->right->deriv, &n->deriv);
            backprop(n->left);
            backprop(n->right);
            return;
        case NL_NODE:
            mat_invnl(&n->left->deriv, &n->matrix, &n->deriv);
            backprop(n->left);
            return;
    }
}

void update_weights(node *n) {
    if(n == NULL) {
        return;
    }

    if(n->type == PARAM_NODE) {
        mat_param_update(&n->matrix, &n->deriv, LEARNING_RATE);
    }

    update_weights(n->left);
    update_weights(n->right);
}

int main(int argc, char **argv) {
    srand(time(NULL));

    FILE *trdf = fopen("./train-images-idx3-ubyte", "rb");
    FILE *trlf = fopen("./train-labels-idx1-ubyte", "rb");
    FILE *tdf = fopen("./t10k-images-idx3-ubyte", "rb");
    FILE *tlf = fopen("./t10k-labels-idx1-ubyte", "rb");

    uint8_t *training_data = malloc(60000 * 784);
    uint8_t *training_labels = malloc(60000);
    uint8_t *test_data = malloc(10000 * 784);
    uint8_t *test_labels = malloc(10000);

    fseek(trdf, 16, SEEK_SET);
    fread(training_data, 1, 60000 * 784, trdf);
    fseek(trlf, 8, SEEK_SET);
    fread(training_labels, 1, 60000, trlf);
    fseek(tdf, 16, SEEK_SET);
    fread(test_data, 1, 10000 * 784, tdf);
    fseek(tlf, 8, SEEK_SET);
    fread(test_labels, 1, 10000, tlf);

    // create a computation graph, with the output y as the root
    node X = make_input_node(784, 1);
    node W1 = make_param_node(40, 784);
    node tmp0 = make_mul_node(&W1, &X);
    node b1 = make_param_node(40, 1);
    node tmp1 = make_add_node(&tmp0, &b1);
    node x2 = make_nl_node(&tmp1);
    node W2 = make_param_node(30, 40);
    node tmp2 = make_mul_node(&W2, &x2);
    node b2 = make_param_node(30, 1);
    node tmp3 = make_add_node(&tmp2, &b2);
    node x3 = make_nl_node(&tmp3);
    node W3 = make_param_node(10, 30);
    node tmp4 = make_mul_node(&W3, &x3);
    node b3 = make_param_node(10, 1);
    node tmp5 = make_add_node(&tmp4, &b3);
    node y = make_nl_node(&tmp5);
    mlmatrix expected = make_matrix(10, 1);

    for(int epoch = 1; epoch <= EPOCHS; epoch++) {
        // evaluate test dataset accuracy
        int num_correct = 0;
        for(int i = 0; i < 10000; i++) {
            load_input(&X, test_data, i);
            forward_prop(&y);
            if(interpret_result(&y) == test_labels[i]) {
                num_correct++;
            }
        }
        printf("epoch %d: %d/10000 correct\n", epoch, num_correct);

        // 10k samples per epoch
        for(int i = 0; i < 10000; i++) {
            int sample_idx = 60000;
            while(sample_idx >= 60000) {
                sample_idx = rand() % 65536;
            }
            load_input(&X, training_data, sample_idx);
            forward_prop(&y);
            onehot_encode(&expected, training_labels[sample_idx]); // encode the expected value
            mat_sub(&y.deriv, &y.matrix, &expected); // here, the derivatives in the output node are manually calculated, and all other derivatives are recursively calculated from that using backpropagate()
            backprop(&y);
            update_weights(&y); // slightly modify all parameters to decrease error
        }
    }

    free_all(&y);
    free_matrix(expected);

    free(training_data);
    free(training_labels);
    free(test_data);
    free(test_labels);

    fclose(trdf);
    fclose(trlf);
    fclose(tdf);
    fclose(tlf);

    return 0;
}
