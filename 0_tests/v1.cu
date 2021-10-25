/**
 *	Tests sur l'utitilisation des variables threadIdx, blockIdx, gridDim
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../common/check.h"

#define N	80

void tmp();


__global__ void test(int *data)
{
	unsigned int index = threadIdx.x;
//	unsigned int value = threadIdx.x;
	unsigned int value = gridDim.x;

	if (index < N)
	{
		data[index] = 3;
	}
}

/**
 * Host function
 */
int main(void)
{
	tmp();
	return 0;

	int i;

	int* h_vector;
	int* d_vector = NULL;

	//	Allocation mémoire host
	h_vector = (int *)malloc(N * sizeof(int));

	// Initialisation mémoire host
	for (i=0; i<N; i++)
		h_vector[i] = -1;

	//	Allocation mémoire device
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_vector, sizeof(int) * N));

	//	Copie des données vers device
	CUDA_CHECK_RETURN(cudaMemcpy(d_vector, h_vector,
		sizeof(int) * N, cudaMemcpyHostToDevice));

	//	Appel kernel
	dim3 blockPerGrid	(1	,1	,1);
	dim3 ThreadPerBlock (N	,1	,1);
	test<<<blockPerGrid, ThreadPerBlock>>>(d_vector);

	//	Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaThreadSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(h_vector, d_vector,
		sizeof(int) * N, cudaMemcpyDeviceToHost));

	for (i = 0; i < N; i++)
	{
		if (i % 8 == 0)	printf("\n");
		printf("%3d ", h_vector[i]);
	}

	CUDA_CHECK_RETURN(cudaFree((void*) d_vector));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}

/*
 *


Xvss=0.5:4
b=0.03:0.2
 */

void tmp()
{
	FILE* outfile = NULL;
	outfile = fopen("data_out", "w+");

	double Io = 900.0;
	double ac = 0.049;

	for (double Xvss=0.5; Xvss < 4.1 ; Xvss +=0.25)
	{
		for (double b=0.03; b<0.2001 ; b+=0.02)
		{
			double Iav = 2*Io/(b*ac*Xvss)*(1-exp(-ac*Xvss*b));
			fprintf(outfile, "%f\t%f\t%f \n", Xvss, b, Iav);
		}
	}
	fclose(outfile);



}
