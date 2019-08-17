#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

using namespace std;

__global__ void add(int *a, int *b, int *sum)
{
	*sum = *a + *b;
}

__global__ void add2(int a, int b, int *sum)
{
	*sum = *sum + a + b;
}

int main()
{
	int a = 100;
	int b = 99;
	int sum;
	int *d_sum;
	int *d_a, *d_b;

	// ------ sa pointerima ----------

	cudaMalloc(&d_sum, sizeof(int));
	cudaMalloc(&d_a, sizeof(int));
	cudaMalloc(&d_b, sizeof(int));

	cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

	add<<<1,1>>> (d_a, d_b, d_sum);

	cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

	cout << sum << endl;

	cudaFree(d_sum);
	cudaFree(d_a);
	cudaFree(d_b);

	// -------------------------

	// ------- po vrednosti ---------------

	cudaMalloc(&d_sum, sizeof(int));

	cudaMemcpy(d_sum, &sum, sizeof(int), cudaMemcpyHostToDevice);

	add2 << <1, 1 >> > (a, b, d_sum);

	cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

	cout << sum << endl;

	cudaFree(d_sum);

	// ----------------------------

	cudaDeviceReset();

	cout << endl;
	system("pause");
    return 0;
}