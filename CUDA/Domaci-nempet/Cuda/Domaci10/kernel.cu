#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define threads 32
#define size 5

using namespace std;

__global__ void callOperation(int *a, int *result, int k, int n)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx >= n || tidy >= n)
	{
		return;
	}

	int tid = tidx * n + tidy;

	if (a[tid] == k)
	{
		atomicAdd(result, 1);
	}
}

int main()
{
	int k = 5;
	int result = 0;
	int *a;
	int *d_a, *d_result;

	a = (int*)malloc(size * size * sizeof(int));

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			a[i * size + j] = size;
		}
	}

	cout << "Vrednost K: " << k<<  endl;
	cout << "\n\nMatrica A:" << endl;
	for (int i = 0; i < size; i++) {
		cout << "\n";
		for (int j = 0; j < size; j++) {
			cout << a[i * size + j] << "\t";
		}
	}

	cudaMalloc((void**)&d_a, size * size * sizeof(int));
	cudaMalloc((void**)&d_result, sizeof(int));

	cudaMemcpy(d_a, a, size * size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 numberOfBlocks(size / threads + 1, size / threads + 1, 1);
	dim3 numberOfThreads(threads, threads, 1);
	callOperation << <numberOfBlocks, numberOfThreads >> > (d_a, d_result, k, size);

	cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

	cout << "Broj ponavljana K: " << result<< endl;

	cudaFree(d_a);
	cudaFree(d_result);
	free(a);
	cudaDeviceReset();

	system("PAUSE");
	return 0;
}