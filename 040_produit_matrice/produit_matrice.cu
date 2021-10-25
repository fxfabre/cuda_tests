#include <stdio.h>
#include <stdlib.h>

#define N	128
#define blocksizeX	512
#define blocksizeY	1

#define CLOCKS_PAR_SEC  1000000l

/************************************************************************/
/* Example                                                              */
/************************************************************************/
__global__ void add_matrix(int *a, int *b, int *c)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int valeur = 0;

	__shared__ int sa[N*N];
	__shared__ int sb[N*N];

	sa[x*N + y] = a[x*N + y];
	sb[x*N + y] = b[x*N + y];

	__syncthreads();

	for (int p=0; p<N; p++)
	{
		valeur += sa[x*N + p] * sb[p*N + y];
	}
	c[x*N + y] = valeur;

}

/************************************************************************/
/* HelloCUDA                                                            */
/************************************************************************/
int main(int argc, char* argv[])
{
	int *a = new int[N*N];
	int *b = new int[N*N];
	int *c = new int[N*N];

	for ( int i = 0; i < N*N; ++i )
	{
		a[i] = 1;
		b[i] = 3;
		c[i] = 0;
	}

	int *ad, *bd, *cd;
	const int size = N*N*sizeof(int);

	cudaMalloc( (void**)&ad, size );
	cudaMalloc( (void**)&bd, size );
	cudaMalloc( (void**)&cd, size );

	cudaMemcpy( ad, a, size, cudaMemcpyHostToDevice );
	cudaMemcpy( bd, b, size, cudaMemcpyHostToDevice );

	dim3 dimBlock( blocksizeX, blocksizeY );
	dim3 dimGrid( N/dimBlock.x, N/dimBlock.y );

	/* mesure du temps d'execution */
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	/* execution de l'opération sur GPU */
	add_matrix<<<dimGrid, dimBlock>>>( ad, bd, cd);
	cudaThreadSynchronize();

	/* Fin de la mesure du temps d'execution du programme */
	cudaEventRecord(stop, 0);
	cudaEventSynchronize( stop );
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost );

	cudaFree( ad );
	cudaFree( bd );
	cudaFree( cd );

	/* vérification des résultats */
	for (int i=0; i < N*N; i++)
	{
		if (c[i] == 884)
		{
			printf("erreur à l'adresse %d \n", i);
			printf("c[%d] = %f \n", i, c[i] );
			getchar();
			return 0;
		}
	}

	/* affichage du temps d'execution */
	printf("temps d'execution sur GPU : %f ms \n", time);


	/**********************************************
		execution de la même opération sur CPU
	 **********************************************/
	clock_t t1, t2;
	double tempsCPU;
	t1 = clock();

	for ( int i = 0; i < N*N; ++i )
	{
		a[i] = 1;
		b[i] = 3;
		c[i] = 0;
	}

	/* execution de l'opération sur CPU */
	for (int x=0; x<N; x++)
	{
		for (int y=0; y<N; y++)
		{
			for (int p=0; p<N; p++)
			{
				c[x*N + y] += a[x*N + p] * b[p*N + y];
			}
		}
	}

	t2 = clock();
	tempsCPU = (double)difftime(t2, t1)/(double)CLOCKS_PAR_SEC;

	/* affichage du temps d'execution */
	printf("temps écoule sur CPU: %f ms \n", tempsCPU * 1000.0);

	getchar();
	delete[] a;
	delete[] b;
	delete[] c;


	return 0;
}
