#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define size 5
#define threads 32

using namespace std;

__global__ void callOperation(int *a, int *res, int x, int n)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx >= n || tidy >= n) {
		return;
	}

	int tid = tidx * n + tidy;

	res[tid] = a[tid] * x;
}

__global__ void callOperationSharedStatic(int *a, int *res, int x, int n)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx >= n || tidy >= n) {
		return;
	}

	int tid = tidx * n + tidy;

	__shared__ int s_a[size * size], s_res[size * size], s_x;

	s_x = x;
	s_a[tid] = a[tid];

	s_res[tid] = s_a[tid] * s_x;

	res[tid] = s_res[tid];
}

__global__ void callOperationSharedDynamic(int *a, int *res, int x, int n)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx >= n || tidy >= n) {
		return;
	}

	int tid = tidx * n + tidy;

	extern __shared__ int data[];

	int *s_a = data;
	int *s_res = &s_a[size * size];

	__shared__ int s_x;

	s_x = x;
	s_a[tid] = a[tid];

	s_res[tid] = s_a[tid] * s_x;

	res[tid] = s_res[tid];
}

int main()
{
	int *a, *res;
	int *d_a, *d_res;
	int x = 10;

	a = (int*)malloc(size * size * sizeof(int));
	res = (int*)malloc(size* size * sizeof(int));

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			a[i * size + j] = size * i - j;
		}
	}

	cout << "Skalar x: " << x << endl;

	cout << "\n\nMatrica A:" << endl;
	for (int i = 0; i < size; i++) {
		cout << endl;
		for (int j = 0; j < size; j++) {
			cout<<a[i * size + j]<<"\t";
		}
	}

	cudaMalloc((void**)&d_a, size * size * sizeof(int));
	cudaMalloc((void**)&d_res, size * size * sizeof(int));

	cudaMemcpy(d_a, a, size * size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 numberOfBlocks(size / threads + 1, size / threads + 1, 1);
	dim3 numberOfThreads(threads, threads, 1);

	//callOperation << <numberOfBlocks, numberOfThreads >> > (d_a, d_res, x, size);
	//callOperationSharedStatic << <numberOfBlocks, numberOfThreads >> > (d_a, d_res, x, size);
	callOperationSharedDynamic << <numberOfBlocks, numberOfThreads , size * size * sizeof(int) + size * size * sizeof(int) >> > (d_a, d_res, x, size);

	cudaMemcpy(res, d_res, size *size * sizeof(int), cudaMemcpyDeviceToHost);


	cout << "\n\nMatrica RES:" << endl;
	for (int i = 0; i < size; i++) {
		cout << endl;
		for (int j = 0; j < size; j++) {
			cout << res[i * size + j] << "\t";
		}
	}


	cudaFree(d_a);
	cudaFree(d_res);
	free(res);
	free(a);

	system("PAUSE");
	return 0;
}
