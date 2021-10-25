#include <stdio.h>
#include <stdlib.h>


int main(void)
{
	cudaDeviceProp prop;
	int whichDevice;

	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);

	if (! prop.deviceOverlap)
	{
		printf("Le GPU ne gère pas les recouvrement !\n");
		printf("Pas d'accélération possible avec les flux...\n");
	}
	else
	{
		printf("Le GPU gère les recouvrement :)\n");
		printf("Utilise les Flux, et que ca saute !\n");
	}


	return 0;
}
