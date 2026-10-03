#include "cuda_nn.h"

#include <cuda_runtime.h>

#include <cstdio>


/*
 * ============================================================
 * OUTILS DE DIAGNOSTIC
 * ============================================================
 */

/*
 * Convertit une erreur CUDA en code d'erreur simple.
 */
static int check_cuda(
    cudaError_t status
)
{
    if (status == cudaSuccess) {
        return 0;
    }

    fprintf(
        stderr,
        "[CUDA] %s\n",
        cudaGetErrorString(status)
    );

    return -1;
}


/*
 * Vérifie le résultat d'un kernel.
 */
static int check_kernel(void)
{
    if (check_cuda(cudaGetLastError()) != 0) {
        return -1;
    }

    if (check_cuda(cudaDeviceSynchronize()) != 0) {
        return -1;
    }

    return 0;
}


/*
 * ============================================================
 * KERNEL : LINEAR
 * ============================================================
 *
 * Pour chaque neurone :
 *
 *   output[i] =
 *       somme(input[j] * weights[i][j])
 *
 * Chaque thread CUDA calcule ici un neurone.
 */
__global__
void linear_kernel(
    const float* input,
    const float* weights,
    float* output,
    int input_size,
    int output_size
)
{
    int neuron =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (neuron >= output_size) {
        return;
    }

    float somme = 0.0f;

    for (int j = 0; j < input_size; ++j) {

        somme +=
            input[j] *
            weights[
                neuron * input_size + j
            ];
    }

    output[neuron] = somme;
}


/*
 * ============================================================
 * KERNEL : BIAS
 * ============================================================
 */
__global__
void add_bias_kernel(
    float* values,
    const float* bias,
    int size
)
{
    int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i >= size) {
        return;
    }

    values[i] += bias[i];
}


/*
 * ============================================================
 * KERNEL : SIGMOID
 * ============================================================
 */
__global__
void sigmoid_kernel(
    const float* input,
    float* output,
    int size
)
{
    int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i >= size) {
        return;
    }

    float x = input[i];

    output[i] =
        1.0f /
        (1.0f + expf(-x));
}


/*
 * ============================================================
 * KERNEL : UPDATE WEIGHTS
 * ============================================================
 *
 * Pour chaque poids :
 *
 *   W[i][j] -=
 *       learning_rate *
 *       gradient[i] *
 *       input[j]
 */
__global__
void update_weights_kernel(
    float* weights,
    const float* gradients,
    const float* input,
    float learning_rate,
    int rows,
    int cols
)
{
    int index =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int total =
        rows * cols;

    if (index >= total) {
        return;
    }

    int row =
        index / cols;

    int col =
        index % cols;

    weights[index] -=
        learning_rate *
        gradients[row] *
        input[col];
}


/*
 * ============================================================
 * KERNEL : UPDATE BIAS
 * ============================================================
 */
__global__
void update_bias_kernel(
    float* bias,
    const float* gradients,
    float learning_rate,
    int size
)
{
    int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i >= size) {
        return;
    }

    bias[i] -=
        learning_rate *
        gradients[i];
}


/*
 * ============================================================
 * KERNEL : GRADIENT COUCHE CACHÉE
 * ============================================================
 *
 * Pour chaque neurone caché :
 *
 *   delta_hidden =
 *
 *       sigmoid'(hidden)
 *
 *       *
 *
 *       somme(
 *           W_output *
 *           delta_output
 *       )
 */
__global__
void hidden_gradient_kernel(
    const float* output_weights,
    const float* delta_output,
    const float* hidden_activation,
    float* delta_hidden,
    int hidden_size,
    int output_size
)
{
    int hidden =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (hidden >= hidden_size) {
        return;
    }

    float propagation = 0.0f;

    for (int output = 0;
         output < output_size;
         ++output)
    {
        propagation +=
            output_weights[
                output * hidden_size + hidden
            ] *
            delta_output[output];
    }

    float activation =
        hidden_activation[hidden];

    float derivee =
        activation *
        (1.0f - activation);

    delta_hidden[hidden] =
        propagation *
        derivee;
}


/*
 * ============================================================
 * INITIALISATION CUDA
 * ============================================================
 */
int cuda_nn_init(void)
{
    int device_count = 0;

    cudaError_t status =
        cudaGetDeviceCount(
            &device_count
        );

    if (status != cudaSuccess) {

        fprintf(
            stderr,
            "[CUDA] Impossible de détecter les GPU : %s\n",
            cudaGetErrorString(status)
        );

        return -1;
    }

    if (device_count == 0) {

        fprintf(
            stderr,
            "[CUDA] Aucun GPU CUDA détecté.\n"
        );

        return -1;
    }

    status =
        cudaSetDevice(0);

    if (status != cudaSuccess) {
        return check_cuda(status);
    }

    cudaDeviceProp properties;

    status =
        cudaGetDeviceProperties(
            &properties,
            0
        );

    if (status != cudaSuccess) {
        return check_cuda(status);
    }

    printf(
        "[CUDA] GPU : %s\n",
        properties.name
    );

    printf(
        "[CUDA] Compute Capability : %d.%d\n",
        properties.major,
        properties.minor
    );

    return 0;
}


/*
 * ============================================================
 * ARRÊT CUDA
 * ============================================================
 */
void cuda_nn_shutdown(void)
{
    cudaDeviceSynchronize();
    cudaDeviceReset();
}


/*
 * ============================================================
 * CUDA LINEAR
 * ============================================================
 */
int cuda_linear(
    const float* input,
    const float* weights,
    float* output,
    int input_size,
    int output_size
)
{
    float* d_input = nullptr;
    float* d_weights = nullptr;
    float* d_output = nullptr;

    size_t input_bytes =
        sizeof(float) *
        input_size;

    size_t weight_bytes =
        sizeof(float) *
        input_size *
        output_size;

    size_t output_bytes =
        sizeof(float) *
        output_size;


    /*
     * Allocation GPU.
     */
    if (check_cuda(
        cudaMalloc(
            (void**)&d_input,
            input_bytes
        )
    ) != 0) {

        return -1;
    }


    if (check_cuda(
        cudaMalloc(
            (void**)&d_weights,
            weight_bytes
        )
    ) != 0) {

        cudaFree(d_input);

        return -1;
    }


    if (check_cuda(
        cudaMalloc(
            (void**)&d_output,
            output_bytes
        )
    ) != 0) {

        cudaFree(d_input);
        cudaFree(d_weights);

        return -1;
    }


    /*
     * CPU -> GPU
     */
    if (check_cuda(
        cudaMemcpy(
            d_input,
            input,
            input_bytes,
            cudaMemcpyHostToDevice
        )
    ) != 0) {

        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_output);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            d_weights,
            weights,
            weight_bytes,
            cudaMemcpyHostToDevice
        )
    ) != 0) {

        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_output);

        return -1;
    }


    /*
     * Lancement du kernel.
     */
    const int threads = 256;

    const int blocks =
        (output_size + threads - 1) /
        threads;

    linear_kernel<<<blocks, threads>>>(
        d_input,
        d_weights,
        d_output,
        input_size,
        output_size
    );


    if (check_kernel() != 0) {

        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_output);

        return -1;
    }


    /*
     * GPU -> CPU
     */
    if (check_cuda(
        cudaMemcpy(
            output,
            d_output,
            output_bytes,
            cudaMemcpyDeviceToHost
        )
    ) != 0) {

        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_output);

        return -1;
    }


    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);

    return 0;
}


/*
 * ============================================================
 * CUDA BIAS
 * ============================================================
 */
int cuda_add_bias(
    float* values,
    const float* bias,
    int size
)
{
    float* d_values = nullptr;
    float* d_bias = nullptr;

    size_t bytes =
        sizeof(float) * size;


    if (check_cuda(
        cudaMalloc(
            (void**)&d_values,
            bytes
        )
    ) != 0) {

        return -1;
    }


    if (check_cuda(
        cudaMalloc(
            (void**)&d_bias,
            bytes
        )
    ) != 0) {

        cudaFree(d_values);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            d_values,
            values,
            bytes,
            cudaMemcpyHostToDevice
        )
    ) != 0) {

        cudaFree(d_values);
        cudaFree(d_bias);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            d_bias,
            bias,
            bytes,
            cudaMemcpyHostToDevice
        )
    ) != 0) {

        cudaFree(d_values);
        cudaFree(d_bias);

        return -1;
    }


    const int threads = 256;

    const int blocks =
        (size + threads - 1) /
        threads;


    add_bias_kernel<<<blocks, threads>>>(
        d_values,
        d_bias,
        size
    );


    if (check_kernel() != 0) {

        cudaFree(d_values);
        cudaFree(d_bias);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            values,
            d_values,
            bytes,
            cudaMemcpyDeviceToHost
        )
    ) != 0) {

        cudaFree(d_values);
        cudaFree(d_bias);

        return -1;
    }


    cudaFree(d_values);
    cudaFree(d_bias);

    return 0;
}


/*
 * ============================================================
 * CUDA SIGMOID
 * ============================================================
 */
int cuda_sigmoid(
    const float* input,
    float* output,
    int size
)
{
    float* d_input = nullptr;
    float* d_output = nullptr;

    size_t bytes =
        sizeof(float) * size;


    if (check_cuda(
        cudaMalloc(
            (void**)&d_input,
            bytes
        )
    ) != 0) {

        return -1;
    }


    if (check_cuda(
        cudaMalloc(
            (void**)&d_output,
            bytes
        )
    ) != 0) {

        cudaFree(d_input);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            d_input,
            input,
            bytes,
            cudaMemcpyHostToDevice
        )
    ) != 0) {

        cudaFree(d_input);
        cudaFree(d_output);

        return -1;
    }


    const int threads = 256;

    const int blocks =
        (size + threads - 1) /
        threads;


    sigmoid_kernel<<<blocks, threads>>>(
        d_input,
        d_output,
        size
    );


    if (check_kernel() != 0) {

        cudaFree(d_input);
        cudaFree(d_output);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            output,
            d_output,
            bytes,
            cudaMemcpyDeviceToHost
        )
    ) != 0) {

        cudaFree(d_input);
        cudaFree(d_output);

        return -1;
    }


    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}


/*
 * ============================================================
 * CUDA UPDATE WEIGHTS
 * ============================================================
 */
int cuda_update_weights(
    float* weights,
    const float* gradients,
    const float* input,
    float learning_rate,
    int rows,
    int cols
)
{
    float* d_weights = nullptr;
    float* d_gradients = nullptr;
    float* d_input = nullptr;

    size_t weight_bytes =
        sizeof(float) *
        rows *
        cols;

    size_t gradient_bytes =
        sizeof(float) *
        rows;

    size_t input_bytes =
        sizeof(float) *
        cols;


    if (check_cuda(
        cudaMalloc(
            (void**)&d_weights,
            weight_bytes
        )
    ) != 0) {

        return -1;
    }


    if (check_cuda(
        cudaMalloc(
            (void**)&d_gradients,
            gradient_bytes
        )
    ) != 0) {

        cudaFree(d_weights);

        return -1;
    }


    if (check_cuda(
        cudaMalloc(
            (void**)&d_input,
            input_bytes
        )
    ) != 0) {

        cudaFree(d_weights);
        cudaFree(d_gradients);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            d_weights,
            weights,
            weight_bytes,
            cudaMemcpyHostToDevice
        )
    ) != 0) {

        cudaFree(d_weights);
        cudaFree(d_gradients);
        cudaFree(d_input);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            d_gradients,
            gradients,
            gradient_bytes,
            cudaMemcpyHostToDevice
        )
    ) != 0) {

        cudaFree(d_weights);
        cudaFree(d_gradients);
        cudaFree(d_input);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            d_input,
            input,
            input_bytes,
            cudaMemcpyHostToDevice
        )
    ) != 0) {

        cudaFree(d_weights);
        cudaFree(d_gradients);
        cudaFree(d_input);

        return -1;
    }


    const int threads = 256;

    const int total =
        rows * cols;

    const int blocks =
        (total + threads - 1) /
        threads;


    update_weights_kernel<<<blocks, threads>>>(
        d_weights,
        d_gradients,
        d_input,
        learning_rate,
        rows,
        cols
    );


    if (check_kernel() != 0) {

        cudaFree(d_weights);
        cudaFree(d_gradients);
        cudaFree(d_input);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            weights,
            d_weights,
            weight_bytes,
            cudaMemcpyDeviceToHost
        )
    ) != 0) {

        cudaFree(d_weights);
        cudaFree(d_gradients);
        cudaFree(d_input);

        return -1;
    }


    cudaFree(d_weights);
    cudaFree(d_gradients);
    cudaFree(d_input);

    return 0;
}


/*
 * ============================================================
 * CUDA UPDATE BIAS
 * ============================================================
 */
int cuda_update_bias(
    float* bias,
    const float* gradients,
    float learning_rate,
    int size
)
{
    float* d_bias = nullptr;
    float* d_gradients = nullptr;

    size_t bytes =
        sizeof(float) * size;


    if (check_cuda(
        cudaMalloc(
            (void**)&d_bias,
            bytes
        )
    ) != 0) {

        return -1;
    }


    if (check_cuda(
        cudaMalloc(
            (void**)&d_gradients,
            bytes
        )
    ) != 0) {

        cudaFree(d_bias);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            d_bias,
            bias,
            bytes,
            cudaMemcpyHostToDevice
        )
    ) != 0) {

        cudaFree(d_bias);
        cudaFree(d_gradients);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            d_gradients,
            gradients,
            bytes,
            cudaMemcpyHostToDevice
        )
    ) != 0) {

        cudaFree(d_bias);
        cudaFree(d_gradients);

        return -1;
    }


    const int threads = 256;

    const int blocks =
        (size + threads - 1) /
        threads;


    update_bias_kernel<<<blocks, threads>>>(
        d_bias,
        d_gradients,
        learning_rate,
        size
    );


    if (check_kernel() != 0) {

        cudaFree(d_bias);
        cudaFree(d_gradients);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            bias,
            d_bias,
            bytes,
            cudaMemcpyDeviceToHost
        )
    ) != 0) {

        cudaFree(d_bias);
        cudaFree(d_gradients);

        return -1;
    }


    cudaFree(d_bias);
    cudaFree(d_gradients);

    return 0;
}


/*
 * ============================================================
 * CUDA GRADIENT HIDDEN
 * ============================================================
 */
int cuda_hidden_gradient(
    const float* output_weights,
    const float* delta_output,
    const float* hidden_activation,
    float* delta_hidden,
    int hidden_size,
    int output_size
)
{
    float* d_output_weights = nullptr;
    float* d_delta_output = nullptr;
    float* d_hidden_activation = nullptr;
    float* d_delta_hidden = nullptr;

    size_t weights_bytes =
        sizeof(float) *
        hidden_size *
        output_size;

    size_t output_bytes =
        sizeof(float) *
        output_size;

    size_t hidden_bytes =
        sizeof(float) *
        hidden_size;


    if (check_cuda(
        cudaMalloc(
            (void**)&d_output_weights,
            weights_bytes
        )
    ) != 0) {

        return -1;
    }


    if (check_cuda(
        cudaMalloc(
            (void**)&d_delta_output,
            output_bytes
        )
    ) != 0) {

        cudaFree(d_output_weights);

        return -1;
    }


    if (check_cuda(
        cudaMalloc(
            (void**)&d_hidden_activation,
            hidden_bytes
        )
    ) != 0) {

        cudaFree(d_output_weights);
        cudaFree(d_delta_output);

        return -1;
    }


    if (check_cuda(
        cudaMalloc(
            (void**)&d_delta_hidden,
            hidden_bytes
        )
    ) != 0) {

        cudaFree(d_output_weights);
        cudaFree(d_delta_output);
        cudaFree(d_hidden_activation);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            d_output_weights,
            output_weights,
            weights_bytes,
            cudaMemcpyHostToDevice
        )
    ) != 0) {

        cudaFree(d_output_weights);
        cudaFree(d_delta_output);
        cudaFree(d_hidden_activation);
        cudaFree(d_delta_hidden);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            d_delta_output,
            delta_output,
            output_bytes,
            cudaMemcpyHostToDevice
        )
    ) != 0) {

        cudaFree(d_output_weights);
        cudaFree(d_delta_output);
        cudaFree(d_hidden_activation);
        cudaFree(d_delta_hidden);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            d_hidden_activation,
            hidden_activation,
            hidden_bytes,
            cudaMemcpyHostToDevice
        )
    ) != 0) {

        cudaFree(d_output_weights);
        cudaFree(d_delta_output);
        cudaFree(d_hidden_activation);
        cudaFree(d_delta_hidden);

        return -1;
    }


    const int threads = 256;

    const int blocks =
        (hidden_size + threads - 1) /
        threads;


    hidden_gradient_kernel<<<blocks, threads>>>(
        d_output_weights,
        d_delta_output,
        d_hidden_activation,
        d_delta_hidden,
        hidden_size,
        output_size
    );


    if (check_kernel() != 0) {

        cudaFree(d_output_weights);
        cudaFree(d_delta_output);
        cudaFree(d_hidden_activation);
        cudaFree(d_delta_hidden);

        return -1;
    }


    if (check_cuda(
        cudaMemcpy(
            delta_hidden,
            d_delta_hidden,
            hidden_bytes,
            cudaMemcpyDeviceToHost
        )
    ) != 0) {

        cudaFree(d_output_weights);
        cudaFree(d_delta_output);
        cudaFree(d_hidden_activation);
        cudaFree(d_delta_hidden);

        return -1;
    }


    cudaFree(d_output_weights);
    cudaFree(d_delta_output);
    cudaFree(d_hidden_activation);
    cudaFree(d_delta_hidden);

    return 0;
}
