dim3 blockPerGrid	(1, 1, 1);
dim3 ThreadPerBlock	(N, 1, 1);
add<<<BlockPerGrid, ThreadPerBlock>>>();
add<<<N, 1>>>();		//	N blocs, 1 thread
	->	0 < blockIdx.x < N
	->	threadIdx.x = 0, blockDim.x = 1, GridDim.x = N
add<<<1, N>>>();		//	1 bloc , N threads
	->	0 < threadIdx.x < N
	->	blockIdx.x = 0, blockDim.x = N, GridDim.x = 1

#pragma unroll 32

Mémoire __constant__ efficace si 16 threads successifs lisent la même donnée.
+ Lecture fréquente des mêmes variables.

Performances maxi pour nb de blocks = 2 x le nombre de multiprocesseur (2 pour moi)

