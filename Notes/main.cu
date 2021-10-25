#include <cuda.h>
#include <cuda_runtime.h>


Type de kernel : 
	__global__ : kernel executé sur le GPU, mais appelé par le CPU
	__device__ : kernel executé et appelé par le GPU
	__host__ : mode par défaut : executé et appelé par le CPU

Appel de kernel : 
	kernel <<< nBlocks, threadsParBloc >>> 	(arguments);
	nBlocks : nombre de subdivisions appliquées à la grille à calculer, type dim3
	threadsParBLoc : nombre de threads à executer simultanement sur chaque
					 bloc, de type dim3.

Chaque kernel dispose de variables implicites, en lecture seule
	blockIdx : index du bloc dans la grille
	threadIdx : index du thread dans le bloc
	blockDim : nombre de threads par bloc

int* A;
int size = n*n* sizeof(int);
cudaMalloc( (void**) &A, size);
cudaFree(A);

cudaMemcpy(A_GPU, A_CPU, size, cudaMemcpyHostToDevice);



