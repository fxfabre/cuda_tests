#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <driver_types.h>
#include "../common/book.h"

#define CLOCKS_PAR_SEC  1000000l

#define	NB_THREAD	192
#define NB_ITER		10

#define	MATRIX_WIDTH	1024
#define MATRIX_HEIGHT	(NB_THREAD * NB_ITER)
#define	VECTOR_LENGTH	MATRIX_WIDTH

__constant__ float const_x[VECTOR_LENGTH];

//	Version un peu plus rapide
__global__ void matVec_V1(float *A, float *b)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	float tmp = 0;

	for (int i=x; i<MATRIX_WIDTH * MATRIX_HEIGHT; i+=MATRIX_WIDTH)
	{
		tmp += A[i] * const_x[i/MATRIX_HEIGHT];
	}
	b[x] = tmp;
}
//	Version un peu plus lente
__global__ void matVec_V2(float *A, float *b)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	float tmp = 0;
	int offset = x * MATRIX_WIDTH;

	for (int i=0; i<MATRIX_WIDTH; i++)
	{
		tmp += A[i+offset] * const_x[i];
	}
	b[x] = tmp;
}
//	Version avec flux slow
__global__ void matVec(float *A, float *b)
{
	register int x = threadIdx.x + blockIdx.x * blockDim.x;
	register float tmp = 0;
//	register int offset = x*MATRIX_WIDTH;

#pragma unroll
	for (int i=0; i<MATRIX_WIDTH; i++)
	{
		tmp += 2.0 * const_x[i];	// A[i+offset]
	}
	b[x] = tmp;
}
//	Initialisation d'un vecteur à zero
__global__ void init_0(float *x)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	x[idx] = 0;
}


/************************************************************************/
/*		 Main                                                           */
/************************************************************************/
int main(int argc, char* argv[])
{
	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	float *host_A, *host_x, *host_b;
	float *dev_A0, *dev_b;
	float *dev_A1;

	// allocate the memory on the Host (CPU)
	HANDLE_ERROR( cudaHostAlloc((void**)&host_A,
								MATRIX_WIDTH * MATRIX_HEIGHT * sizeof(float),
								cudaHostAllocDefault ));
	HANDLE_ERROR( cudaHostAlloc((void**)&host_x,
								VECTOR_LENGTH * sizeof(float),
								cudaHostAllocDefault ));
	HANDLE_ERROR( cudaHostAlloc((void**)&host_b,
								VECTOR_LENGTH * sizeof(float),
								cudaHostAllocDefault ));

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc(	(void**)&dev_A0,
    							MATRIX_WIDTH * NB_THREAD * sizeof(float) ));
    HANDLE_ERROR( cudaMalloc(	(void**)&dev_b,
    							VECTOR_LENGTH * sizeof(float) ));
    HANDLE_ERROR( cudaMalloc(	(void**)&dev_A1,
    							MATRIX_WIDTH * NB_THREAD * sizeof(float) ));

    //	init data
	for (int i = 0; i < MATRIX_WIDTH * MATRIX_HEIGHT; ++i)
	{
		host_A[i] = (float)(i/MATRIX_WIDTH);
	}
	for (int i = 0; i < VECTOR_LENGTH; ++i )
	{
		host_x[i] = 2.0;
	}

	/* mesure du temps d'execution */
	cudaEvent_t start, stop;
	float tempsGPU;
	HANDLE_ERROR( cudaEventCreate(&start)			);
	HANDLE_ERROR( cudaEventCreate(&stop)			);
	HANDLE_ERROR( cudaEventRecord(start, stream0)	);

	/*	Copie des données vers le GPU	*/
	HANDLE_ERROR( cudaMemcpyToSymbol(	const_x, host_x, VECTOR_LENGTH * sizeof(float), 0,
										cudaMemcpyHostToDevice) );

	/*	Initialisation des données sur le GPU	*/
	init_0<<<1, VECTOR_LENGTH>>>(dev_b);

	dim3 ThreadPerBlock	( NB_THREAD	, 1 );
	dim3 BlockPerGrid	( 1			, 1 );
	int offset = NB_THREAD * MATRIX_WIDTH;
	for (int i=0 ; i < NB_ITER ; i+=2)
	{
		HANDLE_ERROR( cudaMemcpyAsync(dev_A0, host_A + i*offset,
				MATRIX_WIDTH * NB_THREAD * sizeof(float), cudaMemcpyHostToDevice, stream0) );
		HANDLE_ERROR( cudaMemcpyAsync(dev_A1, host_A + (i+1)*offset,
				MATRIX_WIDTH * NB_THREAD * sizeof(float), cudaMemcpyHostToDevice, stream1) );


		matVec<<<BlockPerGrid, ThreadPerBlock, 0, stream0>>>(dev_A0, dev_b + i*NB_THREAD);
		matVec<<<BlockPerGrid, ThreadPerBlock, 0, stream1>>>(dev_A1, dev_b + (i+1)*NB_THREAD);
	}
	HANDLE_ERROR( cudaStreamSynchronize(stream0));
	HANDLE_ERROR( cudaStreamSynchronize(stream1));
	HANDLE_ERROR( cudaMemcpy( host_b, dev_b, VECTOR_LENGTH * sizeof(float), cudaMemcpyDeviceToHost));

	/* Fin de la mesure du temps d'execution du programme */
	cudaEventRecord(stop, stream0);
	cudaEventSynchronize( stop );
	cudaEventElapsedTime(&tempsGPU, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree( dev_A0 );
	cudaFree( dev_A1 );
	cudaFree( dev_b );
	cudaStreamDestroy(stream0);

	/* vérification des résultats */
	printf("Resultats calcul GPU : \n");
	for (int i=0; i < VECTOR_LENGTH/NB_THREAD; i++)
	{
		printf("b[%3d] = %5.1f \n", i*NB_THREAD, host_b[i*NB_THREAD] );
	}

	/* affichage du temps d'execution */
	printf("temps écoule sur GPU : %f ms \n\n", tempsGPU);

	/**********************************************
		execution de la même opération sur CPU
	 **********************************************/
	int k=1;
	clock_t t1, t2;
	double tempsCPU;
	t1 = clock();

	/* execution de l'opération sur CPU	*/
	for (k=0; k<50; k++)
	{
		for (int i=0; i<MATRIX_HEIGHT; i++)
		{
			host_b[i] = 0;
			for (int j=0; j<MATRIX_WIDTH; j++)
			{
				host_b[i] += host_A[i*MATRIX_WIDTH + j] * host_x[j];
			}
		}
	}
	for (int i=0; i < VECTOR_LENGTH/NB_THREAD; i++)
	{
		printf("b[%3d] = %5.1f \n", i*NB_THREAD, host_b[i*NB_THREAD] );
	}

	t2 = clock();
	tempsCPU = (double)difftime(t2, t1)/(double)CLOCKS_PAR_SEC;

	/* affichage du temps d'execution */
	tempsCPU = tempsCPU * 1000.0 / k;
	printf("temps écoule sur CPU: %f ms \n\n", tempsCPU);
	printf("Speedup : %3.2fx \n", tempsCPU / tempsGPU);

	cudaFreeHost(host_A);
	cudaFreeHost(host_b);

	return EXIT_SUCCESS;
}
