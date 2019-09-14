#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#define size 5
#define numThreads 32

using namespace std;

__global__ void addMatrix(int *a, int *b, int *res, int n)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx >= n || tidy >= n) {
		return;
	}

	int tid = tidx * n + tidy;
	res[tid] = a[tid] + b[tid];
}

__global__ void addMatrixSharedStatic(int *a, int *b, int *res, int n)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx >= n || tidy >= n)
	{
		return;
	}

	int tid = tidx * n + tidy;

	__shared__ int s_a[size*size], s_b[size*size], s_res[size*size];

	s_a[tid] = a[tid];
	s_b[tid] = b[tid];

	s_res[tid] = s_a[tid] + s_b[tid];
	res[tid] = s_res[tid];
}

__global__ void addMatrixSharedDynamic(int *a, int *b, int *res, int n)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	if (tidx >= n || tidy >= n)
	{
		return;
	}

	int tid = tidx * n + tidy;

	extern __shared__ int arrays[];

	int *s_a = arrays;
	int *s_b = &arrays[size * size];
	int *s_res = &s_b[size*size];

	s_a[tid] = a[tid];
	s_b[tid] = b[tid];

	s_res[tid] = s_a[tid] + s_b[tid];
	res[tid] = s_res[tid];
}

int main()
{
	int *a, *b, *res;
	int *d_a, *d_b, *d_res;

	//alokacija
	a = (int*)malloc(size * size * sizeof(int));
	b = (int*)malloc(size * size * sizeof(int));
	res = (int*)malloc(size * size * sizeof(int));

	//popunim random podacima
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			a[i * size + j] = i * size;
			b[i * size + j] = -i;
		}
	}

	//stampa
	cout << "\n\nVelicina matrice:" << size << endl;

	cout << "\n\nMatrica A:" << endl;
	for (int i = 0; i < size; i++) {
		cout << "\n";
		for (int j = 0; j < size; j++) {
			cout << a[i * size + j] << "\t";
		}
	}

	cout << "\n\nMatrica B:" << endl;
	for (int i = 0; i < size; i++) {
		cout << "\n";
		for (int j = 0; j < size; j++) {
			cout << b[i * size + j] << "\t";
		}
	}


	//cuda deo


	//alokacija
	cudaMalloc((void**)&d_a, size * size * sizeof(int));
	cudaMalloc((void**)&d_b, size * size * sizeof(int));
	cudaMalloc((void**)&d_res, size * size * sizeof(int));

	//slanje podataka na device
	cudaMemcpy(d_a, a, size * size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * size * sizeof(int), cudaMemcpyHostToDevice);

	//poziv kernela
	dim3 numberOfBlocks(size / numThreads + 1, size / numThreads + 1);
	dim3 numberOfThreads(numThreads, numThreads);

	//addMatrix << <numberOfBlocks, numberOfThreads >> > (d_a, d_b, d_res, size);
	//addMatrixSharedStatic << <numberOfBlocks, numberOfThreads >> > (d_a, d_b, d_res, size);
	addMatrixSharedDynamic << <numberOfBlocks, numberOfThreads, size * size * sizeof(int) + size * size * sizeof(int)  + size * size * sizeof(int) >> > (d_a, d_b, d_res, size);


	//vracanje rezultata nazad
	cudaMemcpy(res, d_res, size * size * sizeof(int), cudaMemcpyDeviceToHost);

	//stampa
	cout << "\n\nA+B = :" << endl;
	for (int i = 0; i < size; i++) {
		cout << "\n";
		for (int j = 0; j < size; j++) {
			cout << res[i * size + j] << "\t";
		}
	}

	//dealokacija
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);
	free(a);
	free(b);

	cudaDeviceReset();

	cout << endl;
	system("PAUSE");
	return 0;
}