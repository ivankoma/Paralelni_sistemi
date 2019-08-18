
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#define size 15

using namespace std;


__global__ void callOperation(int *a, int *b, int *res, int x, int n) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n) {
		res[tid] = a[tid] - (b[tid] * x);
	}
}

__global__ void callOperationSharedStatic(int *a, int *b, int *res, int x, int n)
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

	s_res[tid] = s_a[tid] - (s_b[tid] * s_x);
	res[tid] = s_res[tid];
}

__global__ void callOperationSharedDynamic(int *a, int *b, int *res, int x, int n)
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

	s_res[tid] = s_a[tid] - (s_b[tid] * s_x);
	res[tid] = s_res[tid];
}

int main()
{
	int *a, *b, *res;
	int x = 10;
	int *d_a, *d_b, *d_res;

	a = (int*)malloc(size * sizeof(int));
	b = (int*)malloc(size * sizeof(int));
	res = (int*)malloc(size * sizeof(int));

	for (int i = 0; i < size; i++) {
		a[i] = i*size;
		b[i] = -i;
	}

	cout << "\n\nNiz A:" << endl;
	for (int i = 0; i < size; i++) {
		cout << a[i] << "\t";
	}

	cout << "\n\nNiz B:" << endl;
	for (int i = 0; i < size; i++) {
		cout << b[i] << "\t";
	}

	cout << "\n\nSkalar je:" << x << endl;

	cudaMalloc(&d_a, size * sizeof(int));
	cudaMalloc(&d_b, size * sizeof(int));
	cudaMalloc(&d_res, size * sizeof(int));

	cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

	//callOperation << <size / 256 + 1, 256 >> > (d_a, d_b, d_res, x, size);
	//callOperationSharedStatic << <size / 256 + 1, 256 >> > (d_a, d_b, d_res, x, size);
	callOperationSharedDynamic << <size / 256 + 1, 256, size * sizeof(int) + size * sizeof(int) + size * sizeof(int) >> > (d_a, d_b, d_res, x, size);

	cudaMemcpy(res, d_res, size * sizeof(int), cudaMemcpyDeviceToHost);

	cout << "\n\nNiz Rez:" << endl;
	for (int i = 0; i < size; i++) {
		cout << res[i] << "\t";
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);
	free(a);
	free(b);
	free(res);

	cudaDeviceReset();


	cout << endl;
	system("PAUSE");
	return 0;
}







