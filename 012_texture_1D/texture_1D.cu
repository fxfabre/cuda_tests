#include <stdio.h>
#include <stdlib.h>
#include "../common/check.h"

#define N	256

texture<int> texture_a;
texture<int> texture_b;

__global__ void add_vector(int *dev_c)
{
	int tx = threadIdx.x;
	dev_c[tx] = 23;
	dev_c[tx] = tex1Dfetch(texture_a, tx) + tex1Dfetch(texture_b, tx);
}

int main()
{
	int *host_a, *host_b, *host_c;
	int *dev_a , *dev_b , *dev_c;

	//	Déclaration vecteurs sur HOST
	host_a = (int*)malloc(N * sizeof(int));
	host_b = (int*)malloc(N * sizeof(int));
	host_c = (int*)malloc(N * sizeof(int));

	//	Initialisation des vecteurs
	for (int i=0; i<N; i++)
	{
		host_a[i] = 2*i;
		host_b[i] = i*i;
	}

	//	Allocation des vecteurs sur DEVICE
	cudaMalloc( (void**)&dev_a, N*sizeof(int));
	cudaMalloc( (void**)&dev_b, N*sizeof(int));
	cudaMalloc( (void**)&dev_c, N*sizeof(int));

	//	Association mémoire globale - mémoire texture
	cudaBindTexture(NULL, texture_a, dev_a, N*sizeof(int));
	cudaBindTexture(NULL, texture_b, dev_b, N*sizeof(int));

	//	Copie des données vers GPU
	cudaMemcpy(dev_a, host_a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, N*sizeof(int), cudaMemcpyHostToDevice);

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

	//	Libération mémoire textures
	cudaUnbindTexture( texture_a );
	cudaUnbindTexture( texture_b );

	//	Libération mémoire GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
