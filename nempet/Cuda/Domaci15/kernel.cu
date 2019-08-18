
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define size 25
#define threads 32

using namespace std;

__global__ void callOperation(int * a, int *b, int *res, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n)
	{
		return;
	}

	res[tid] = a[tid] - b[tid];
	if (res[tid] < 0)
	{
		res[tid] = 0;
	}
}


__global__ void callOperationSharedStatic(int * a, int *b, int *res, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n)
	{
		return;
	}

	__shared__ int s_a[size], s_b[size], s_res[size];

	s_a[tid] = a[tid];
	s_b[tid] = b[tid];

	s_res[tid] = s_a[tid] - s_b[tid];
	if (s_res[tid] < 0)
	{
		s_res[tid] = 0;
	}
	res[tid] = s_res[tid];
}

__global__ void callOperationSharedDynamic(int * a, int *b, int *res, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n)
	{
		return;
	}

	extern __shared__ int data[];

	int *s_a = data;
	int *s_b = &s_a[size];
	int *s_res = &s_b[size];

	s_a[tid] = a[tid];
	s_b[tid] = b[tid];

	s_res[tid] = s_a[tid] - s_b[tid];
	if (s_res[tid] < 0)
	{
		s_res[tid] = 0;
	}
	res[tid] = s_res[tid];
}

int main()
{
	int *a, *b, *res;
	int *d_a, *d_b, *d_res;

	a = (int*)malloc(size * sizeof(int));
	b = (int*)malloc(size * sizeof(int));
	res = (int*)malloc(size * sizeof(int));

	for (int i = 0; i < size; i++)
	{
		a[i] = -i;
		b[i] = i;
	}

	cout << "\nNiz A:" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << a[i] << "\t";
	}

	cout << "\nNiz B:" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << b[i] << "\t";
	}

	cudaMalloc((void**)&d_a, size * sizeof(int));
	cudaMalloc((void**)&d_b, size * sizeof(int));
	cudaMalloc((void**)&d_res, size * sizeof(int));

	cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 numberOfBlocks(size / threads + 1, 1, 1);
	dim3 numberOfThreads(threads, 1, 1);

	//callOperation << < numberOfBlocks, numberOfThreads >> > (d_a, d_b, d_res, size);
	//callOperationSharedStatic << < numberOfBlocks, numberOfThreads >> > (d_a, d_b, d_res, size);
	callOperationSharedDynamic << < numberOfBlocks, numberOfThreads, size * sizeof(int)  + size * sizeof(int)  + size * sizeof(int) >> > (d_a, d_b, d_res, size);

	cudaMemcpy(res, d_res, size * sizeof(int), cudaMemcpyDeviceToHost);

	cout << "\nNiz RES:" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << res[i] << "\t";
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);
	free(a);
	free(b);
	free(res);
	cudaDeviceReset();


	system("PAUSE");
	return 0;
}