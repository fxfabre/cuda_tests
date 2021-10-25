#include <stdio.h>
#include <stdlib.h>
#include "../common/check.h"

#define N	64
#define P	4

/*
 * Somme de matrices N x P
 */

texture<int, 2> texture_a;
texture<int, 2> texture_b;

__global__ void add_vector(int *dev_a, int *dev_c)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	dev_c[tx + ty*N] = tex2D(texture_a, tx, ty) + tex2D(texture_b, tx, ty);
}

int main()
{
	int *host_a, *host_b, *host_c;
	int *dev_a , *dev_b , *dev_c;

	//	Déclaration vecteurs sur HOST
	host_a = (int*)malloc(N * P * sizeof(int));
	host_b = (int*)malloc(N * P * sizeof(int));
	host_c = (int*)malloc(N * P * sizeof(int));

	//	Initialisation des vecteurs
	for (int i=0; i<N*P; i++)
	{
		host_a[i] = i;
		host_b[i] = 2*i;
	}

	//	Allocation des vecteurs sur DEVICE
	cudaMalloc( (void**)&dev_a, N*P*sizeof(int));
	cudaMalloc( (void**)&dev_b, N*P*sizeof(int));
	cudaMalloc( (void**)&dev_c, N*P*sizeof(int));

	//	Génération du descripteur
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();

	//	Association mémoire globale - mémoire texture
	cudaBindTexture2D(NULL, texture_a, dev_a, N, P, N*sizeof(int));
	cudaBindTexture2D(NULL, texture_b, dev_b, N, P, N*sizeof(int));

	//	Copie des données vers GPU
	cudaMemcpy(dev_a, host_a, N*P*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, N*P*sizeof(int), cudaMemcpyHostToDevice);

	// appel kernel
	dim3 blockPerGrid	(1, 1, 1);
	dim3 ThreadPerBlock	(N, P, 1);
	add_vector<<<blockPerGrid, ThreadPerBlock>>>(dev_a, dev_c);

	//	Copie des données depuis le GPU
	cudaMemcpy(host_c, dev_c, N*P*sizeof(int), cudaMemcpyDeviceToHost);

	//	Affichage résultats
	for (int i=0; i<N*P; i++)
	{
		if (i%32 == 0) printf("\n");
		if (i%N  == 0) printf("\n");
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
