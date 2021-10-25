#include <stdio.h>
#include <stdlib.h>
#include "../common/check.h"


#define N	256

__constant__ int const_a[N];
__constant__ int const_b[N];


/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void add_vector(int *dev_c)
{
	int tx = threadIdx.x;
	dev_c[tx] = const_a[tx] + const_b[tx];
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void)
{
	int *host_a, *host_b, *host_c;
	int *dev_c;

	//	Déclaration vecteurs sur HOST
	host_a = (int*)malloc(N * sizeof(int));
	host_b = (int*)malloc(N * sizeof(int));
	host_c = (int*)malloc(N * sizeof(int));

	//	Initialisation des vecteurs
	for (int i=0; i<N; i++)
	{
		host_a[i] = i;
		host_b[i] = i*i;
	}

	//	Allocation des vecteurs sur DEVICE
	cudaMalloc( (void**)&dev_c, N*sizeof(int));

	//	Copie des données vers GPU
	cudaMemcpyToSymbol(const_a, host_a, N*sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(const_b, host_b, N*sizeof(int), 0, cudaMemcpyHostToDevice);

	// appel kernel
	dim3 blockPerGrid	(1, 1, 1);
	dim3 ThreadPerBlock	(N, 1, 1);
	add_vector<<<blockPerGrid, ThreadPerBlock>>>(dev_c);

	//	Copie des données depuis le GPU
	cudaMemcpy(host_c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

	//	Affichage résultats
	for (int i=0; i<N; i++)
	{
		if (i%32 == 0) printf("\n");
		printf("%5d ", host_c[i]);
	}

	//	Libération mémoire CPU
	free(host_a);
	free(host_b);
	free(host_c);

	//	Libération mémoire GPU
	cudaFree(dev_c);

	//	 Pas de free des mémoires constantes !!

	return 0;
}
