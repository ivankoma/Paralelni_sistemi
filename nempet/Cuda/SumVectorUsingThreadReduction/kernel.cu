
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>

#define size 64
#define threads 128

using namespace std;

/*
Jedan primer ThreadReduction-a
__global__ void callOperation(int *v)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int step = 1;
	int numOfThreads = blockDim.x;

	while (numOfThreads > 0)
	{
		if (tid < size)
		{
			int first = tid * step * 2;
			int second = first + step;
			v[first] += v[second];
		}

		step <<= 1;
		numOfThreads >>= 1;
	}
}
*/

__global__ void callOperationReduction(int *v)
{
	int tid = threadIdx.x;
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (tid < offset)
		{
			v[tid] += v[tid + offset];
		}

		__syncthreads();
	}

	if (tid == 0)
	{
		v[blockIdx.x] = v[0];
	}
}


__global__ void callOperationReductionSharedStatic(int *v)
{
	__shared__ int s_v[size];

	int tid = threadIdx.x;
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < size)
	{
		s_v[tid] = v[tidx];
	}
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (tid < offset)
		{
			s_v[tid] += s_v[tid + offset];
		}

		__syncthreads();
	}

	if (tid == 0)
	{
		v[blockIdx.x] = s_v[0];
	}
}

__global__ void callOperationReductionSharedDynamic(int *v)
{
	extern __shared__ int s_v[];

	int tid = threadIdx.x;
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < size)
	{
		s_v[tid] = v[tidx];
	}
	__syncthreads();

	for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (tid < offset)
		{
			s_v[tid] += s_v[tid + offset];
		}

		__syncthreads();
	}

	if (tid == 0)
	{
		v[blockIdx.x] = s_v[0];
	}
}

int main()
{
	int *v,*sum2, sum;
	int *d_v, *d_res;

	v = (int*)malloc(size * sizeof(int));
	sum2 = (int*)malloc(size * sizeof(int));

	for (int i = 0; i < size; i++)
	{
		v[i] = i;
	}

	cout << "Niz: " << endl;
	for (int i = 0; i < size; i++)
	{
		cout << v[i] << "\t";
	}

	cudaMalloc((void**)&d_v, size * sizeof(int));
	cudaMemcpy(d_v, v, size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 numberOfBlocks(size / threads + 1, 1, 1);
	dim3 numberOfThreads(threads, 1, 1);

	//callOperationReduction << <numberOfBlocks, numberOfThreads >> > (d_v);
	callOperationReductionSharedStatic << <numberOfBlocks, numberOfThreads >> > (d_v);
	//callOperationReductionSharedDynamic << <numberOfBlocks, numberOfThreads, size * sizeof(int) >> > (d_v);

	cudaMemcpy(sum2, d_v, size * sizeof(int), cudaMemcpyDeviceToHost);


	cout << "\n\nNiz: " << endl;
	for (int i = 0; i < size; i++)
	{
		cout <<sum2[i] << "\t";
	}
	cout << "\n\nSum is: " << sum2[0] << endl;

	cudaFree(d_v);
	free(v);
	free(sum2);
	cudaDeviceReset();

	cout << endl;
	system("PAUSE");
    return 0;
}