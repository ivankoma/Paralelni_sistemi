
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#define threads 32
#define size 128

using namespace std;


__global__ void callOperation(int *a, int *b, int *res, int k, int p, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n) {
		return;
	}

	res[tid] = a[tid] - b[tid];
	if (res[tid] < k) {
		res[tid] = p;
	}
}

__global__ void callOperationSharedStatic(int *a, int *b, int *res, int k, int p, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n) {
		return;
	}

	__shared__ int s_a[size], s_b[size], s_res[size], s_k, s_p;

	s_k = k;
	s_p = p;
	s_a[tid] = a[tid];
	s_b[tid] = b[tid];

	s_res[tid] = s_a[tid] - s_b[tid];
	if (s_res[tid] < s_k) {
		s_res[tid] = s_p;
	}

	res[tid] = s_res[tid];
}


__global__ void callOperationSharedDynamic(int *a, int *b, int *res, int k, int p, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= n) {
		return;
	}

	extern __shared__ int data[];

	int *s_a = data;
	int *s_b = &s_a[n];
	int *s_res = &s_b[n];

	__shared__ int s_k, s_p;

	s_k = k;
	s_p = p;
	s_a[tid] = a[tid];
	s_b[tid] = b[tid];

	s_res[tid] = s_a[tid] - s_b[tid];
	if (s_res[tid] < s_k) {
		s_res[tid] = s_p;
	}

	res[tid] = s_res[tid];
}

int main()
{
	int *a, *b, *res;
	int *d_a, *d_b, *d_res;
	int k, p;

	cout << "Unesi k:" << endl;
	cin >> k;
	cout << "Unesi p:" << endl;
	cin >> p;

	a = (int*)malloc(size * sizeof(int));
	b = (int*)malloc(size * sizeof(int));
	res = (int*)malloc(size * sizeof(int));

	for (int i = 0; i < size; i++) {
		a[i] = size * i;
		b[i] = -i;
	}

	cout << "\n\nNiz a:" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << a[i] << "\t";
	}

	cout << "\n\nNiz b:" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << b[i] << "\t";
	}

	cudaMalloc((void**)&d_a, size * sizeof(int));
	cudaMalloc((void**)&d_b, size * sizeof(int));
	cudaMalloc((void**)&d_res, size * sizeof(int));

	cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

	//callOperation << <size / threads + 1, threads >> > (d_a, d_b, d_res, k, p, size);
	//callOperationSharedStatic << <size / threads + 1, threads >> > (d_a, d_b, d_res, k, p, size);
	callOperationSharedDynamic << <size / threads + 1, threads, size * sizeof(int) + size * sizeof(int) + size * sizeof(int) >> > (d_a, d_b, d_res, k, p, size);

	cudaMemcpy(res, d_res, size * sizeof(int), cudaMemcpyDeviceToHost);

	cout << "\n\nNiz res:" << endl;
	for (int i = 0; i < size; i++)
	{
		cout << res[i] << "\t";
	}

	cudaFree(d_res);
	cudaFree(d_a);
	cudaFree(d_b);
	free(a);
	free(b);
	free(res);

	cudaDeviceReset();

	system("PAUSE");
	return 0;
}