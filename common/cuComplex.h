#ifndef _CUCOMPLEX_H_
#define _CUCOMPLEX_H_


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


#endif /* _CUCOMPLEX_H_ */
