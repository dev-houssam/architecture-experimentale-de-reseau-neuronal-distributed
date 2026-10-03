# Compilation

## Vérifier CUDA :

`nvcc --version`

## Puis vérifier le GPU :

`nvidia-smi`

## Compiler :

`make`

## Exécuter :

`make run`

## Pour une RTX A6000 :

`make CUDA_ARCH=sm_86`
