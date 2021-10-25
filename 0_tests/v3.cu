#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <driver_types.h>
#include "../common/book.h"

#define CLOCKS_PAR_SEC  1000000l

#define	N	64

//	Initialisation d'un vecteur à zero
__global__ void init_0(float *x)
{
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	x[idx] = 63.21;
}


/************************************************************************/
/*		 Main                                                           */
/************************************************************************/
int main(int argc, char* argv[])
{
	float *host_b;
	float *dev_b;


	host_b = (float *)malloc(N * sizeof(float));
	cudaMalloc((void**) &dev_b, sizeof(float) * N);

	init_0<<<1, N>>>(dev_b);

	HANDLE_ERROR( cudaMemcpy( host_b, dev_b, N * sizeof(float), cudaMemcpyDeviceToHost));


	/* vérification des résultats */
	for (int i=0; i < N; i++)
	{
		printf("b[%3d] = %5.1f \n", i, host_b[i] );
	}


	cudaFree( dev_b );
	cudaFreeHost(host_b);
	return EXIT_SUCCESS;
}
