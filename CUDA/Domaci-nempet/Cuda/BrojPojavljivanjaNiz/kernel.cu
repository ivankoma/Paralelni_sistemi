#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define size 128
#define threads 32

using namespace std;

__global__ void callOperation(int *niz, int *res, int k, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n) {
		return;
	}

	if (niz[tid] == k) {
		atomicAdd(res, 1);
	}
}

int main()
{
	int k = 128;
	int res = 0;
	int *niz;
	int *d_res, *d_niz;

	niz = (int*)malloc(size * sizeof(int));

	for (int i = 0; i < size; i++) {
		niz[i] = size;
	}

	cudaMalloc((void**)&d_niz, size * sizeof(int));
	cudaMalloc((void**)&d_res, sizeof(int));

	cudaMemcpy(d_niz, niz, size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 numberOfBlocks(size / threads + 1, 1, 1);
	dim3 numberOfThreads(threads, 1, 1);

	callOperation << <numberOfBlocks, numberOfThreads >> > (d_niz, d_res, k, size);

	cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);

	cout << "Rezultat je: " << res << endl;

	cudaFree(d_res);
	cudaFree(d_niz);
	free(niz);

	system("PAUSE");
	return 0;
}