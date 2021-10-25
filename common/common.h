#ifndef COMMON_H_
#define COMMON_H_

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


struct cuComplex
{
	float r;
	float i;

	cuComplex(float a, float b) : r(a), i(b) {}

	float distance(void)
	{
		return r*r + i*i;
	}
	cuComplex operator*(const cuComplex &a)
	{
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	cuComplex operator+(const cuComplex &a)
	{
		return cuComplex(r+a.r , i+a.i);
	}
};


#endif /* COMMON_H_ */
