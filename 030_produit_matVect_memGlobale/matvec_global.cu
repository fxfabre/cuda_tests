#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CLOCKS_PAR_SEC  1000000l

#define N	256


/************************************************************************/
/* Example                                                              */
/************************************************************************/
__global__ void matVec(float *a, float *b, float *c)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int index = x * N;
	float tmp = 0;

	for (int i=0; i<N; i++)
	{
		tmp += a[index + i] * b[i];
	}
	c[x] = tmp;
}

/************************************************************************/
/*		 Main                                                           */
/************************************************************************/
int main(int argc, char* argv[])
{
	float *host_a, *host_b, *host_c;
	float *dev_a, *dev_b, *dev_c;

	const int size = N * sizeof(float);

	host_a = (float*)malloc( size * N);
	host_b = (float*)malloc( size);
	host_c = (float*)malloc( size);

//	cudaHostAlloc(&dev_a, size*N, cudaHostAllocDefault);
	cudaMalloc( (void**)&dev_a, size * N);
	cudaMalloc( (void**)&dev_b, size );
	cudaMalloc( (void**)&dev_c, size );

	for (int i = 0; i < N*N; ++i)
	{
		host_a[i] = 3.0;
	}
	for (int i = 0; i < N; ++i )
	{
		host_b[i] = 2.0;
	}

	/* mesure du temps d'execution */
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	/*	Copie des données vers le GPU	*/
	cudaMemcpy(dev_a, host_a, size * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, size, 	cudaMemcpyHostToDevice);

	/* execution de l'opération sur GPU */
	dim3 ThreadPerBlock	( 128	, 1 );
	dim3 BlockPerGrid	( N/128	, 1 );
	matVec<<<BlockPerGrid, ThreadPerBlock>>>(dev_a, dev_b, dev_c);

	cudaMemcpy( host_c, dev_c, size, cudaMemcpyDeviceToHost );

	/* Fin de la mesure du temps d'execution du programme */
	cudaEventRecord(stop, 0);
	cudaEventSynchronize( stop );
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree( dev_c );

	/* vérification des résultats */
	for (int i=0; i<N; i++)
	{
		if (host_c[i] != 6*N)
		{
//			printf("erreur à l'adresse %d \n", i);
			printf("c[%3d] = %5.1f \n", i, host_c[i] );
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
			host_c[i] = host_a[i] + host_b[i];
	}

	t2 = clock();
	tempsCPU = (double)difftime(t2, t1)/(double)CLOCKS_PAR_SEC;

	/* affichage du temps d'execution */
	printf("temps écoule sur CPU: %f ms \n", tempsCPU * 1000.0 / j);

	free(host_a);
	free(host_b);
	free(host_c);

	return EXIT_SUCCESS;
}
