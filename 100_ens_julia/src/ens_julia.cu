#include <stdio.h>
#include <stdlib.h>
#include "common.h"

#define SIZE_X	32
#define SIZE_Y	32
#define DELTA_X	0.01
#define DELTA_y	0.01


__global__ void ensJulia(int **d_img)
{
	unsigned int idxTx = threadIdx.x;
	unsigned int idxTy = threadIdx.y;

	if ( (idxTx < 2*SIZE_X) && (idxTy < 2*SIZE_Y) )
	{
		d_img[idxTx][idxTy] = idxTx + idxTy * 2 *SIZE_Y;
	}
}

int main(void)
{
	int i,j;

	int h_img[2*SIZE_X][2*SIZE_Y];
	int **d_img = NULL;


	CUDA_CHECK_RETURN(	cudaMalloc((void**) &d_img,
		sizeof(int) * 4 * SIZE_X * SIZE_Y));
//	CUDA_CHECK_RETURN(	cudaMemcpy(d, idata,
//		sizeof(int) * 4 * SIZEX * SIZE_Y, cudaMemcpyHostToDevice));

	dim3 threadSize(SIZE_X*2, SIZE_Y*2);
	ensJulia<<<threadSize, 1>>>(d_img);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(h_img, d_img,
		sizeof(int) * 4 * SIZE_X * SIZE_Y, cudaMemcpyDeviceToHost));

	//	Affichage
	for (i=0; i<2*SIZE_X; i++)
	{
		for (j=0; j<2*SIZE_Y; j++)
		{
			printf("%3d ", h_img[i][j]);
		}
		printf("\n");
	}

	CUDA_CHECK_RETURN(cudaFree((void*) d_img));
	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}
