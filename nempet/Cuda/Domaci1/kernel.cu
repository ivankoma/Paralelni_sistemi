#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#define size 5

using namespace std;

__global__ void callOperation(int *a, int *b, int x, int *res, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < n) {
		res[tid] = ((a[tid] * x) + b[tid]);
	}
}

__global__ void callOperationSharedStatic(int *a, int *b, int x, int *res, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n)
	{
		return;
	}

	__shared__ int s_a[size], s_b[size], s_res[size];
	__shared__ int s_x;

	s_x = x;
	s_a[tid] = a[tid];
	s_b[tid] = b[tid];

	s_res[tid] = ((s_a[tid] * s_x) + s_b[tid]);
	res[tid] = s_res[tid];
}

__global__ void callOperationSharedDynamic(int *a, int *b, int x, int *res, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n)
	{
		return;
	}

	extern __shared__ int arrays[];
	__shared__ int s_x;

	int *s_a = arrays;
	int *s_b = &s_a[n];
	int *s_res = &s_b[n];

	s_x = x;
	s_a[tid] = a[tid];
	s_b[tid] = b[tid];

	s_res[tid] = ((s_a[tid] * x) + s_b[tid]);
	res[tid] = s_res[tid];
}

int main()
{
	//A*x + B

	int *a, *b, *res;
	int x = 10;
	int *d_a, *d_b, *d_res;

	a = (int*)malloc(size * sizeof(int));
	b = (int*)malloc(size * sizeof(int));
	res = (int*)malloc(size * sizeof(int));

	for (int i = 0; i < size; i++) {
		a[i] = size * i;
		b[i] = -i;
	}

	cout << "\n\nX je " << x;
	cout << "\n\nA niz" << endl;
	for (int i = 0; i < size; i++) {
		cout << a[i] << "\t";
	}
	cout << "\n\nB niz" << endl;
	for (int i = 0; i < size; i++) {
		cout << b[i] << "\t";
	}


	//cuda deo

	cudaMalloc(&d_a, size * sizeof(int));
	cudaMalloc(&d_b, size * sizeof(int));
	cudaMalloc(&d_res, size * sizeof(int));

	cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

	//callOperation << <size / 256 + 1, 256 >> > (d_a, d_b, x, d_res, size);
	//callOperationSharedStatic << <size / 256 + 1, 256 >> > (d_a, d_b, x, d_res, size);
	callOperationSharedDynamic << < size / 256 + 1, 256, size * sizeof(int) + size * sizeof(int)  + size * sizeof(int) >> > (d_a, d_b, x, d_res, size);

	cudaMemcpy(res, d_res, size * sizeof(int), cudaMemcpyDeviceToHost);

	cout << "\n\nRes niz" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << res[i] << "\t";
	}

	free(a);
	free(b);
	free(res);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);
	cudaDeviceReset();

	cout << endl;
	system("PAUSE");
	return 0;
}