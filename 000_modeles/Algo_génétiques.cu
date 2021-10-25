#include <stdio.h>
#include <stdlib.h>

#define CLOCKS_PER_SEC  1000000l

#define blocksizeX	512
#define blocksizeY	1
#define Block_Chromosome	5120	//	N * blockSizeX * blockSizeY

#define gridsizeX	1024
#define gridsizeY	1

#define N	10					// taille d'une solution = nb de dépot
#define NB_CHROMOSOME	1024	//	gridsizeX * gridsizeY
#define NN (N*NB_CHROMOSOME)

#define A	3
#define	B	6

#define	AFFICHAGE	1

/************************************************************************/
/*		Code GPU                                                        */
/************************************************************************/
__global__ static void croisement(int* D_p, int* D_e)
{
	char k = B;
	char i;

	short Tx = threadIdx.x;
	short idxPa = (blockIdx.x * blockDim.x + Tx) * N;

	short xa = Tx * N;
	short xb;

	__shared__ char used[Block_Chromosome];
	__shared__ char Parent[Block_Chromosome];

	//	copie des parents de mémoire globale à mémoire partagée
	for (i=0; i<N; i++)
	{
		used[xa + i] = 0;
		Parent[xa+i] = D_p[idxPa + i];
	}
	__syncthreads();

	if (Tx %2 == 0)
		xb = xa +N;
	else
		xb = xa -N;

	//	copie de la section identique au centre
	for (i=A; i<B; i++)
	{
		D_e[idxPa + i] = Parent[xb + i];
		used[ xa + Parent[xb + i] ] = 1;
	}

	__syncthreads();

	for (i=0; i<N; i++)
	{
		if (used[ xa + Parent[xa + i] ] == 0)
		{
			D_e[idxPa + k] = Parent[xa + i];
			k = (k + 1) % N;
		}
	}
}


/************************************************************************/
/*		 Main                                                           */
/*	H_var	: variable sur Host (CPU)									*/
/*	D_var	: variable sur Device (GPU)									*/
/************************************************************************/
int main(int argc, char* argv[])
{
	int pa[N] = {8, 7, 3, 4, 5, 6, 0, 2, 1, 9};
	int pb[N] = {7, 6, 0, 1, 2, 9, 8, 4, 3, 5};

	int *H_p = new int[NN];				//	matrice contenant les parents, à copier vers GPU
	int *H_e = new int[NN];				//	matrice contenant les enfants
	for ( int i = 0; i < NN; ++i )	H_e[i] = 0;

	/* affichage des parents */
	if (AFFICHAGE)
	{
		for (int i=0; i<N; i++) printf("%c ", pa[i] + 'A');
		printf("\n");
		for (int i=0; i<N; i++) printf("%c ", pb[i] + 'A');
		printf("\n\n");
	}

	/* initialisation de la matrice */
	for (int i=0; i<NB_CHROMOSOME; i++)
	{
		for (int j=0; j<N; j++)
		{
			H_p[i*N + j] = pa[j];
		}
		i++;
		for (int j=0; j<N; j++)
		{
			H_p[i*N + j] = pb[j];
		}
	}

	int *D_p, *D_e;
	const int sizeNN = NN * sizeof(int);

	cudaMalloc( (void**)&D_p, sizeNN );
	cudaMalloc( (void**)&D_e, sizeNN );

	cudaMemcpy( D_p, H_p, sizeNN, cudaMemcpyHostToDevice );

	dim3 dimBlock( blocksizeX, blocksizeY );
	dim3 dimGrid( gridsizeX/dimBlock.x, gridsizeY/dimBlock.y );

	/* mesure du temps d'execution */
	cudaEvent_t start, stop;
	float time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	/* execution de l'opération sur GPU */
	croisement<<<dimGrid, dimBlock>>>( D_p, D_e);
	cudaThreadSynchronize();

	/* Fin de la mesure du temps d'execution du programme */
	cudaEventRecord(stop, 0);
	cudaEventSynchronize( stop );
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy( H_e, D_e, sizeNN, cudaMemcpyDeviceToHost );

	cudaFree( D_p );
	cudaFree( D_e );

	/* affichage du temps d'execution */
	printf("temps d'execution sur GPU : %f ms \n", time);

	/* affichage des résultats */
	if (AFFICHAGE)
	{
		printf("matrice des enfants obtenue sur GPU : \n");
		for (int i=0; i<4; i++)
		{
			for (int j=0; j<N; j++)
			{
				printf("%c ", H_e[i*N + j] + 'A');
			}
			printf("\n");
		}
		printf("\n\n");
	}


	/**********************************************
		execution de la même opération sur CPU
	 **********************************************/
	int used[N] = {0};
	int idxPa, idxPb;

	/* mesure du temps d'execution */
	clock_t t1, t2;
	double tempsCPU;
	t1 = clock();

	for (int p=0; p<10000; p++)
	{
		for (idxPa=0; idxPa<NB_CHROMOSOME; idxPa++)
		{
			for (int i=0; i<N; i++) used[i] = 0;

			if (idxPa %2 == 0)
			{
				idxPb = idxPa + 1;
			}
			else
			{
				idxPb = idxPa - 1;
			}

			for (int i=A; i<B; i++)
			{
				H_e[idxPa*N + i] = H_p[idxPb*N + i];
				used[ H_p[idxPb*N + i] ] = 1;
			}

			int i = B;
			for (int j=0; j<N; j++)
			{
				if (used[ H_p[idxPa*N + j] ] == 0)
				{
					H_e[idxPa*N + i] = H_p[idxPa*N + j];
					i = (i+1) % N;
				}

			}
		}
	}

	/* Fin de la mesure du temps d'execution du programme */
	t2 = clock();
	tempsCPU = (double)difftime(t2, t1)/(double)CLOCKS_PER_SEC;

	/* affichage du temps d'execution */
	printf("temps écoule sur CPU: %f ms \n", tempsCPU * 1000.0);

	/* affichage des résultats */
	if (AFFICHAGE)
	{
		printf("matrice des enfants obtenue sur CPU : \n");
		for (int i=0; i<2; i++)
		{
			for (int j=0; j<N; j++)
			{
				printf("%c ", H_p[i*N + j] + 'A');
			}
			printf("\n");
		}
	}

	getchar();
	delete[] H_p;
	delete[] H_e;

	return 0;
}
