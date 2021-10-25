#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CLOCKS_PAR_SEC  1000000l

#define N	8192

texture<float> texture_a;
texture<float> texture_b;

/************************************************************************/
/* Example                                                              */
/************************************************************************/
__global__ void add_matrix(float *c)
{
	int tx = threadIdx.x + blockIdx.x * blockDim.x;

	c[tx] = tex1Dfetch(texture_a, tx) + tex1Dfetch(texture_b, tx);
}

/************************************************************************/
/*		 Main                                                           */
/************************************************************************/
int main(int argc, char* argv[])
{
	float *host_a, *host_b, *host_c;
	float *dev_a, *dev_b, *dev_c;

	const int size = N*sizeof(float);

	host_a = (float*)malloc( N * sizeof(float));
	host_b = (float*)malloc( N * sizeof(float));
	host_c = (float*)malloc( N * sizeof(float));

	cudaMalloc( (void**)&dev_a, size );
	cudaMalloc( (void**)&dev_b, size );
	cudaMalloc( (void**)&dev_c, size );

	//	Association mémoire globale - mémoire texture
	cudaBindTexture(NULL, texture_a, dev_a, N*sizeof(float));
	cudaBindTexture(NULL, texture_b, dev_b, N*sizeof(float));

	for ( int i = 0; i < N; ++i )
	{
		host_a[i] = 3.0;
		host_b[i] = 2.0;
	}

	/* mesure du temps d'execution */
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	/*	Copie des données vers le GPU	*/
	cudaMemcpy(dev_a, host_a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, N*sizeof(int), cudaMemcpyHostToDevice);

	/* execution de l'opération sur GPU */
	dim3 dimBlock	( N/128, 1 );
	dim3 dimGrid	( 128, 1 );
	add_matrix<<<dimGrid, dimBlock>>>(dev_c);

	cudaMemcpy( host_c, dev_c, size, cudaMemcpyDeviceToHost );

	/* Fin de la mesure du temps d'execution du programme */
	cudaEventRecord(stop, 0);
	cudaEventSynchronize( stop );
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/* vérification des résultats */
	for (int i=0; i<N; i++)
	{
		if (host_c[i] != 5)
		{
			printf("erreur à l'adresse %d \n", i);
			printf("c[%d] = %f \n", i, host_c[i] );
			getchar();
			return 0;
		}
	}

	/* affichage du temps d'execution */
	printf("temps écoule sur GPU : %f ms \n", time);


	/**********************************************
		execution de la même opération sur CPU
	 **********************************************/
	int j=0;
	clock_t t1, t2;
	double tempsCPU;
	t1 = clock();

	/* execution de l'opération sur CPU */
	for (j=0; j<1000; j++)
	{
		for (int i=0; i<N; i++)
		{
			host_c[i] = host_a[i] + host_b[i];
		}
	}

	t2 = clock();
	tempsCPU = (double)difftime(t2, t1)/(double)CLOCKS_PAR_SEC;

	/* affichage du temps d'execution */
	printf("temps écoule sur CPU: %f ms \n", tempsCPU * 1000.0 / j);

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

	return EXIT_SUCCESS;
}
