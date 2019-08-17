
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define size 5
#define threads 32

using namespace std;

__global__ void callOperation(int * a, int *b, int *res, int n)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx >= n || tidy >= n)
	{
		return;
	}
	int tid = tidx * n + tidy;

	res[tid] = a[tid] - b[tid];
	if (res[tid] < 0)
	{
		res[tid] = 0;
	}
}

__global__ void callOperationSharedStatic(int * a, int *b, int *res, int n)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx >= n || tidy >= n)
	{
		return;
	}
	int tid = tidx * n + tidy;

	__shared__ int s_a[size * size], s_b[size * size], s_res[size * size];

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
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx >= n || tidy >= n)
	{
		return;
	}
	int tid = tidx * n + tidy;

	extern __shared__ int data[];

	int *s_a = data;
	int *s_b = &s_a[size * size];
	int *s_res = &s_b[size * size];

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

	a = (int*)malloc(size * size * sizeof(int));
	b = (int*)malloc(size * size * sizeof(int));
	res = (int*)malloc(size * size * sizeof(int));

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			a[i * size + j] = -i;
			b[i * size + j] = j;
		}
	}

	cout << "\nMat A:" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << "\n";
		for (int j = 0; j < size; j++)
		{
			cout << a[i * size + j] << "\t";
		}
	}

	cout << "\nMat B:" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << "\n";
		for (int j = 0; j < size; j++)
		{
			cout << b[i * size + j] << "\t";
		}
	}

	cudaMalloc((void**)&d_a, size * size * sizeof(int));
	cudaMalloc((void**)&d_b, size * size * sizeof(int));
	cudaMalloc((void**)&d_res, size * size * sizeof(int));

	cudaMemcpy(d_a, a, size * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 numberOfBlocks(size / threads + 1, size / threads + 1, 1);
	dim3 numberOfThreads(threads, threads, 1);

	//callOperation << < numberOfBlocks, numberOfThreads >> > (d_a, d_b, d_res, size);
	//callOperationSharedStatic << < numberOfBlocks, numberOfThreads >> > (d_a, d_b, d_res, size);
	callOperationSharedDynamic << < numberOfBlocks, numberOfThreads, size * size * sizeof(int) + size * size * sizeof(int)+ size * size * sizeof(int) >> > (d_a, d_b, d_res, size);

	cudaMemcpy(res, d_res, size * size * sizeof(int), cudaMemcpyDeviceToHost);

	cout << "\nMat Res:" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << "\n";
		for (int j = 0; j < size; j++)
		{
			cout << res[i * size + j] << "\t";
		}
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