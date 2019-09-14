#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#define SIZE 5
#define threads 32

using namespace std;

__global__ void addTwoArrays(int *v1, int *v2, int *r, int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= n) {
		return;
	}

	r[tid] = v1[tid] + v2[tid];
}

__global__ void addTwoArraysSharedStatic(int *v1, int *v2, int *r, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n)
	{
		return;
	}

	__shared__ int s_v1[SIZE], s_v2[SIZE], s_r[SIZE];

	s_v1[tid] = v1[tid];

	s_v2[tid] = v2[tid];

	s_r[tid] = s_v1[tid] + s_v2[tid];
	r[tid] = s_r[tid];
}


__global__ void addTwoArraysSharedDynamic(int *v1, int *v2, int *r, int n)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n)
	{
		return;
	}

	extern __shared__ int arrays[];
	int *s_v1 = arrays;
	int *s_v2 = &s_v1[n];
	int *s_r = &s_v2[n];

	s_v1[tid] = v1[tid];

	s_v2[tid] = v2[tid];

	s_r[tid] = s_v1[tid] + s_v2[tid];
	r[tid] = s_r[tid];
}


int main()
{
	int *vector1, *vector2, *resultVector;
	int *d_v1, *d_v2, *d_result;

	vector1 = (int*)malloc(SIZE * sizeof(int));
	vector2 = (int*)malloc(SIZE * sizeof(int));
	resultVector = (int*)malloc(SIZE * sizeof(int));

	cout << "Popunjavanje vektora" << endl;
	for (int i = 0; i < SIZE; i++)
	{
		vector1[i] = SIZE *i;
		vector2[i] = -i;
	}

	cout << "\nVektor 1 stampa" << endl;
	for (int i = 0; i < SIZE; i++)
	{
		cout << vector1[i] << "\t";
	}

	cout << "\n\nVektor 2 stampa" << endl;
	for (int i = 0; i < SIZE; i++)
	{
		cout << vector2[i] << "\t";
	}

	cudaMalloc(&d_v1, SIZE * sizeof(int));
	cudaMalloc(&d_v2, SIZE * sizeof(int));
	cudaMalloc(&d_result, SIZE * sizeof(int));

	cudaMemcpy(d_v1, vector1, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v2, vector2, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	//addTwoArrays <<< SIZE / threads + 1, threads >> > (d_v1, d_v2, d_result, SIZE);
	//addTwoArraysSharedStatic << < SIZE / threads + 1, threads >> > (d_v1, d_v2, d_result, SIZE);
	addTwoArraysSharedDynamic<< < SIZE / threads + 1, threads, SIZE * sizeof(int) + SIZE * sizeof(int) + SIZE * sizeof(int) >> > (d_v1, d_v2, d_result, SIZE);

	cudaMemcpy(resultVector, d_result, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	cout << "\n\nStampa rezultujuceg vektora" << endl;
	for (int i = 0; i < SIZE; i++)
	{
		cout << resultVector[i] << "\t";
	}

	cudaFree(d_v1);
	cudaFree(d_v2);
	free(vector1);
	free(vector2);
	free(resultVector);
	cudaDeviceReset();

	cout << endl;
	system("PAUSE");
	return 0;
}