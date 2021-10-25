#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define CLOCKS_PAR_SEC  1000000l


typedef int vector_t;

#define N	8192

/************************************************************************/
/* Example                                                              */
/************************************************************************/
__global__ void add_matrix(vector_t *a, vector_t *b, vector_t *c)
{
	int tx = threadIdx.x + blockIdx.x * blockDim.x;

	c[tx] = a[tx] + b[tx];
}

/************************************************************************/
/*		 Main                                                           */
/************************************************************************/
int main(int argc, char* argv[])
{
	vector_t *a = new vector_t[N];
	vector_t *b = new vector_t[N];
	vector_t *c = new vector_t[N];

	for ( int i = 0; i < N; ++i )
	{
		a[i] = 3;
		b[i] = 2;
	}

	vector_t *ad, *bd, *cd;
	const int size = N*sizeof(vector_t);

	cudaMalloc( (vector_t**)&ad, size );
	cudaMalloc( (vector_t**)&bd, size );
	cudaMalloc( (vector_t**)&cd, size );

	/* mesure du temps d'execution */
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	/*	Copie des données vers le GPU	*/
	cudaMemcpy( ad, a, size, cudaMemcpyHostToDevice );
	cudaMemcpy( bd, b, size, cudaMemcpyHostToDevice );

	dim3 dimBlock	( N/512, 1 );
	dim3 dimGrid	( 512, 1 );

	/* execution de l'opération sur GPU */
	add_matrix<<<dimGrid, dimBlock>>>( ad, bd, cd);

	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost );

	/* Fin de la mesure du temps d'execution du programme */
	cudaEventRecord(stop, 0);
	cudaEventSynchronize( stop );
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree( ad );
	cudaFree( bd );
	cudaFree( cd );

	/* vérification des résultats */
	for (int i=0; i<N; i++)
	{
		if (c[i] != 5)
		{
			printf("erreur à l'adresse %d \n", i);
			printf("c[%d] = %d \n", i, c[i] );
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
			c[i] = a[i] + b[i];
	}

	t2 = clock();
	tempsCPU = (double)difftime(t2, t1)/(double)CLOCKS_PAR_SEC;

	/* affichage du temps d'execution */
	printf("temps écoule sur CPU: %f ms \n", tempsCPU * 1000.0 / j);

	getchar();
	delete[] a;
	delete[] b;
	delete[] c;

	return EXIT_SUCCESS;
}
