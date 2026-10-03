#ifndef CUDA_NN_H
#define CUDA_NN_H

/*
 * Interface C publique exposée à Go via CGO.
 *
 * Le fichier .cu contient l'implémentation réelle.
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialise CUDA et sélectionne le GPU 0.
 *
 * Retourne :
 *   0  : succès
 *  -1  : erreur
 */
int cuda_nn_init(void);

/*
 * Libère les ressources CUDA.
 */
void cuda_nn_shutdown(void);


/*
 * ============================================================
 * CALCULS FORWARD
 * ============================================================
 */

/*
 * Calcul matriciel :

 *   output[i] = somme(input[j] * weights[i][j])
 *
 * Les poids sont stockés en row-major :
 *
 *   weights[neurone][entrée]
 */
int cuda_linear(
    const float* input,
    const float* weights,
    float* output,
    int input_size,
    int output_size
);


/*
 * Ajout du biais :

 *   values[i] += bias[i]
 */
int cuda_add_bias(
    float* values,
    const float* bias,
    int size
);


/*
 * Fonction sigmoid :

 *   sigmoid(x) = 1 / (1 + exp(-x))
 */
int cuda_sigmoid(
    const float* input,
    float* output,
    int size
);


/*
 * ============================================================
 * CALCULS BACKWARD
 * ============================================================
 */

/*
 * Mise à jour des poids :

 *   W -= learning_rate * gradient * input
 */
int cuda_update_weights(
    float* weights,
    const float* gradients,
    const float* input,
    float learning_rate,
    int rows,
    int cols
);


/*
 * Mise à jour des biais :

 *   B -= learning_rate * gradient
 */
int cuda_update_bias(
    float* bias,
    const float* gradients,
    float learning_rate,
    int size
);


/*
 * Calcule le gradient de la couche cachée :

 * delta_hidden[i] =
 *
 *   sigmoid'(hidden[i])
 *   *
 *   somme(
 *       weights_output[j][i] *
 *       delta_output[j]
 *   )
 */
int cuda_hidden_gradient(
    const float* output_weights,
    const float* delta_output,
    const float* hidden_activation,
    float* delta_hidden,
    int hidden_size,
    int output_size
);


#ifdef __cplusplus
}
#endif

#endif
