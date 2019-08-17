
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define size 25
#define threads 32

using namespace std;

__global__ void callOperation(int *a, int *b, int *c, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n) {
		return;
	}

	if (a[tid] <= b[tid])
	{
		c[tid] = a[tid];
	}
	else
	{
		c[tid] = b[tid];
	}
}

__global__ void callOperationSharedStatic(int *a, int *b, int *c, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n) {
		return;
	}

	__shared__ int s_a[size], s_b[size], s_c[size];

	s_a[tid] = a[tid];
	s_b[tid] = b[tid];

	if (s_a[tid] <= s_b[tid])
	{
		s_c[tid] = s_a[tid];
	}
	else
	{
		s_c[tid] = s_b[tid];
	}
	c[tid] = s_c[tid];
}

__global__ void callOperationSharedDynamic(int *a, int *b, int *c, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n) {
		return;
	}

	extern __shared__ int data[];

	int *s_a = data;
	int *s_b = &s_a[size];
	int *s_c = &s_b[size];

	s_a[tid] = a[tid];
	s_b[tid] = b[tid];

	if (s_a[tid] <= s_b[tid])
	{
		s_c[tid] = s_a[tid];
	}
	else
	{
		s_c[tid] = s_b[tid];
	}
	c[tid] = s_c[tid];
}


int main()
{
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;

	a = (int*)malloc(size * sizeof(int));
	b = (int*)malloc(size * sizeof(int));
	c = (int*)malloc(size * sizeof(int));

	for (int i = 0; i < size; i++) {
		a[i] = -i;
		b[i] = -i + size / 11;
	}

	cout << "\n\nNiz A:" << endl;
	for (int i = 0; i < size; i++) {
		cout << a[i] << "\t";
	}

	cout << "\n\nNiz B:" << endl;
	for (int i = 0; i < size; i++) {
		cout << b[i] << "\t";
	}

	cudaMalloc((void**)&d_a, size * sizeof(int));
	cudaMalloc((void**)&d_b, size * sizeof(int));
	cudaMalloc((void**)&d_c, size * sizeof(int));

	cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 numberOfBlocks(size / threads + 1, 1, 1);
	dim3 numberOfThreads(threads, 1, 1);

	//callOperation << <numberOfBlocks, numberOfThreads >> > (d_a, d_b, d_c, size);
	//	callOperationSharedStatic << <numberOfBlocks, numberOfThreads >> > (d_a, d_b, d_c, size);
	callOperationSharedDynamic << <numberOfBlocks, numberOfThreads, size * sizeof(int) + size * sizeof(int) + size * sizeof(int) >> > (d_a, d_b, d_c, size);

	cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

	cout << "\n\nNiz C:" << endl;
	for (int i = 0; i < size; i++) {
		cout << c[i] << "\t";
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(a);
	free(b);
	free(c);
	cudaDeviceReset();

	system("PAUSE");
	return 0;
}