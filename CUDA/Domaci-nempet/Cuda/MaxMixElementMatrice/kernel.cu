
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define size 5
#define threads 32

using namespace std;


__global__ void callOperationMin(int *mat)
{
	int tid = threadIdx.x;
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (tid < offset)
		{
			if (mat[tid] > mat[tid + offset])
			{
				mat[tid] = mat[tid + offset];
			}

			__syncthreads();
		}
	}

	if (tid == 0)
	{
		mat[blockIdx.x] = mat[tid];
	}
}

__global__ void callOperationMax(int *mat)
{
	int tid = threadIdx.x;
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (tid < offset)
		{
			if (mat[tid] < mat[tid + offset])
			{
				mat[tid] = mat[tid + offset];
			}

			__syncthreads();
		}
	}

	if (tid == 0)
	{
		mat[blockIdx.x] = mat[tid];
	}
}

int main()
{
	int *mat;
	int *d_mat;

	mat = (int*)malloc(size * size * sizeof(int));

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			mat[i * size + j] = i * j;
		}
	}

	cout << "\nMatrica:" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << endl;
		for (int j = 0; j < size; j++)
		{
			cout << mat[i * size + j] << "\t";
		}
	}

	cudaMalloc((void**)&d_mat, size * size * sizeof(int));
	cudaMemcpy(d_mat, mat, size * size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 numberOfBlocks(size / threads + 1, size / threads + 1, 1);
	dim3 numberOfThreads(threads, threads, 1);

	int min, max;

	callOperationMin << <numberOfBlocks, numberOfThreads >> > (d_mat);
	cudaMemcpy(&min, d_mat, sizeof(int), cudaMemcpyDeviceToHost);
	cout << "\nMin el: " << min << endl;

	callOperationMax<< <numberOfBlocks, numberOfThreads >> > (d_mat);
	cudaMemcpy(&max, d_mat, sizeof(int), cudaMemcpyDeviceToHost);
	cout << "\nMax el: " << max << endl;

	cudaFree(d_mat);
	free(mat);
	cudaDeviceReset();

	cout << endl;
	system("PAUSE");
    return 0;
}