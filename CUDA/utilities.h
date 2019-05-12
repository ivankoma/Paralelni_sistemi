#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
using namespace std;

void print_array(int * a, int n) {
	for (int i = 0; i < n; i++) {
		printf("%d ", a[i]);
	}
	printf("\n");
}

void print_matrix(int * a, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			printf("%d\t", a[i*m + j]);
		}
		printf("\n");
	}
}

void fill_array(int * a, int n, string how, int range=0) {
	/// random, i, i*i, 2*i 
	srand(time(NULL));
	if (how == "random") {
		for (int i = 0; i < n; i++) {
			a[i] = -range + rand() % (2*range);
		}
		return;
	}
	if (how == "i") {
		for (int i = 0; i < n; i++) {
			a[i] = i;
		}
		return;
	}
	if (how == "i*i") {
		for (int i = 0; i < n; i++) {
			a[i] = i*i;
		}
		return;
	}
	if (how == "2*i") {
		for (int i = 0; i < n; i++) {
			a[i] = 2 * i;
		}
		return;
	}
}

void compare_arrays(int *a, int *b, int size) {
	for (int i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			printf("Arrays are different!\n");
			printf("%d!=%d", a[i], b[i]);
			return;
		}
	}
	printf("Congratulations, arrays are the same.\n");
}

void sum_two_arrays_cpu(int * a, int * b, int * c, int size) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
}

int cpu_find_array(int * a, int size, string what) {
	int value = a[0];
	if (what == "min") {
		for (int i = 0; i < size; i++) {
			if (a[i] < value) {
				value = a[i];
			}
		}
	}
	if (what == "max") {
		for (int i = 0; i < size; i++) {
			if (a[i] > value) {
				value = a[i];
			}
		}
	}
	return value;
}