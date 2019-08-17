
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define size 16
#define threads 32

using namespace std;


__global__ void callOperationMin(int * niz)
{
	int tid = threadIdx.x;
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (tid < offset)
		{
			if (niz[tid] > niz[tid + offset])
			{
				niz[tid] = niz[tid + offset];
			}
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		niz[blockIdx.x] = niz[tid];
	}
}

__global__ void callOperationMax(int * niz)
{
	int tid = threadIdx.x;
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if(tid < offset)
		{
			if (niz[tid] < niz[tid + offset])
			{
				niz[tid] = niz[tid + offset];
			}

			__syncthreads();
		}
	}

	if (tid == 0)
	{
		niz[blockIdx.x] = niz[tid];
	}
}

int main()
{
	int *niz;
	int *d_niz;

	niz = (int*)malloc(size * sizeof(int));
	int * niz2 = (int*)malloc(size * sizeof(int));

	for (int i = 0; i < size; i++) {
		niz[i] = i;
	}

	cout << "Niz: " << endl;
	for (int i = 0; i < size; i++)
	{
		cout << niz[i] << "\t";
	}

	cudaMalloc((void**)&d_niz, size * sizeof(int));
	cudaMemcpy(d_niz, niz, size * sizeof(int), cudaMemcpyHostToDevice);

	callOperationMax << <size / threads + 1, threads >> > (d_niz);
	int max;
	cudaMemcpy(&max, d_niz, sizeof(int), cudaMemcpyDeviceToHost);

	/*
	callOperationMin << <size / threads + 1, threads >> > (d_niz);
	int min;
	cudaMemcpy(&min, d_niz, sizeof(int), cudaMemcpyDeviceToHost);
	*/

	cudaMemcpy(niz2, d_niz, size * sizeof(int), cudaMemcpyDeviceToHost);

	cout << "\n\nMax je: " << max;
	//cout << "\n\nMin je: " << min;

	cout << "\n\nNiz: " << endl;
	for (int i = 0; i < size; i++)
	{
		cout << niz2[i] << "\t";
	}

	cudaFree(d_niz);
	free(niz);
	cudaDeviceReset();

	cout << endl;
	system("PAUSE");
    return 0;
}
