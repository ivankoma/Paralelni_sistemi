# MPI

```c
#include "stdafx.h"
#include "mpi.h"
#include <stdio.h>

using namespace std; 
int main()
{
  int num_tasks, rank;

  int rc = MPI_Init(NULL, NULL);
  if (rc != MPI_SUCCESS) {
    printf("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("Greetings from Process: %d", rank);
  MPI_Finalize();
    return 0;
}
```

## Docs

### MPI_Address

Gets the address of a location in memory 

```
int MPI_Address(const void *input_location, MPI_Aint *output_address)
```
> This is depricated function. To be able to use it, add `_CRT_SECURE_NO_WARNINGS` and `MSMPI_NO_DEPRECATE_20` to preprocessor definitions [in this way](https://stackoverflow.com/questions/16883037/remove-secure-warnings-crt-secure-no-warnings-from-projects-by-default-in-vis)
### MPI_Send

```
int MPI_Send(const void *buf, //initial address of send buffer (choice) 
             int count, 
             MPI_Datatype datatype, 
             int dest,        // rank of destination (integer) 
             int tag,         // message tag (integer) 
             MPI_Comm comm)   // communicator (handle) 
```

### MPI_Recv
```
int MPI_Recv(void *buf, 
            int count, 
            MPI_Datatype datatype, 
            int source, 
            int tag, 
            MPI_Comm comm, 
            MPI_Status *status)
```

### MPI_Bcast

Broadcasts a message from the process with rank "root" to all other processes of the communicator. 

[How to use](https://stackoverflow.com/questions/7864075/using-mpi-bcast-for-mpi-communication)

```
int MPI_Bcast(void *buffer,
              int count, 
              MPI_Datatype datatype, 
              int root, 
              MPI_Comm comm)
```

### MPI_type_struct

```
int MPI_Type_struct(int count,
                   const int *array_of_blocklengths,
                   const MPI_Aint *array_of_displacements,
                   const MPI_Datatype *array_of_types,
                   MPI_Datatype *newtype)
```


- `count` number of blocks (integer) in datatype, also number of entries in arrays array_of_types, array_of_displacements and array_of_blocklengths 
- `array_of_blocklengths` number of elements in each block (array). *i-ti* član ovog niza je broj elemenata tipa `array_of_types[i]` u i-tom bloku.
- `array_of_displacements` byte displacement of each block (array). Niz pomeraja svakog bloka u odnosu na početnu adresu strukture, ali izražen u bajtovima. Dobija se uz pomoć funkcije `MPI_Address`
- `array_of_types` type of elements in each block (array of handles to datatype objects) 


Example:

```
struct{char a, int b, double c} val;

MPI_Address(&val.a, &baseadr);  // array_of_types[0] is MPI_CHAR
MPI_Address(&val.b, &adr_b);    // array_of_types[1] is MPI_INT
MPI_Address(&val.c, &adr_c);    // array_of_types[2] is MPI_DOUBLE

array_of_displacements[0]=0 // pomeraj prvog bloka u odnosu na početak strukture
array_of_displacements[1]=adr_b-baseadr
array_of_displacements[2]=adr_c-baseadr
```

### MPI_Typevector

Omogucava da formiramo izveden tip podatka gde su blokovi koji čine izveden tip iste veličine. Poceči blokova se nalaze na *jednakim rastojanjima*.

```
int MPI_Typevector(
  int count, 
  int block_len,
  int stride, //predstavlja razmak izmedju pocetaka blokova izrazen u elementima starog tipa (stari tip je sledeci argument funkcije),
  MPI_Type oldtype);
```

- `count` number of blocks (nonnegative integer) 
- `blocklength` number of elements in each block (nonnegative integer) 
- `stride` number of elements between start of each block (integer). Predstavlja razmak izmedju početaka blokova izražen u elementima starog tipa. Stari tip je sledeći tip funkcije.
- `oldtype` old datatype (handle) 

Example:

```
// There are two blocks, each made of 3 blocks of old data type
count = 2, blocklen=3, stride=5, oldtype=MPI_Int;
```

|1|||||2||||||
|-|-|-|-|-|-|-|-|-|-|-|
|x|x|x|_|_|x|x|x|_|_|_|

Example: Send a column

```
count = n, blocklen=1, stride=n, MPI_Int;

[x] x x
[x] x x
[x] x x n*n
```

### MPI_Type_indexed

```
MPI_Type_indexed(
  int count, 
  int *array_of_blocklens,
  int *array_of_displacements, 
  MPI_Datatype oldtype,
  MPI_Datatype *newtype
);
```

- `count` number of blocks, also number of entries in *array_of_displacements* and *array_of_blocklengths*
- `array_of_blocklengths` number of elements in each block (array of nonnegative integers). Niz koji pamti veličinu svakog bloka, koja može biti različita
- `array_of_displacements` displacement of each block in multiples of oldtype (array of integers). Niz koji pamti pomeraje za svaki blok ali pomeraji su izrazeni u elementima starog tipa (u MPI_Typevector je izrazen u bajtovima)
- `oldtype` old datatype (handle) 

Example:

```
count=3, 
blocklens[0]=2
blocklens[1]=1
blocklens[2]=4
displacements[0]=0
displacements[1]=3
displacements[2]=5
```

|0|1|2|3|4|5|6|7|8|
|-|-|-|-|-|-|-|-|-|
|A|A|_|B|_|C|C|C|C|


## 1. Zadatak

> Napisati MPI program kojim se izvedenim tipovima podataka vrši slanje kolone sa indeksom 1 procesa 0 matrice A, u vrstu sa indeksom 2 procesa 1. matrice A. Matrica A je reda *n*.

```
Matrix in process 0:
0       1       2       3
4       5       6       7
8       9       10      11
12      13      14      15
Matrix in process 1:
0       0       0       0
0       0       0       0
1       5       9       13
0       0       0       0
```
```c
#include "stdafx.h"
#include "mpi.h"
#include <stdio.h>
#include "C:\Users\i\Desktop\GitHub\CUDA\utilities.h"
#define n 4

using namespace std; 
void main(int argc, char *argv[]){
  int A[n][n], rank, i, j;
  MPI_Status st;
  MPI_Datatype kolona;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Type_vector(n, 1, n, MPI_INT, &kolona); // n blokova, velicine 1, udaljeni n blokova, tip podataka u matrici, novi tip podataka
  MPI_Type_commit(&kolona);

  if (rank == 0) {
    fill_array(&A[0][0], n*n, "2*i");
    printf("Matrix in process 0:\n");
    print_matrix(&A[0][0], n, n);
    // buf, count, MPI_Datatype, dest, tag, MPI_Comm
    MPI_Send(&(A[0][1]), 1, kolona, 1, 0, MPI_COMM_WORLD);
  }
  else {
    fill_array(&A[0][0], n*n, "0");
    // buf, count, MPI_Datatype, source, tag, MPI_Comm, MPI_Status)
    MPI_Recv(&A[2][0], n, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
    printf("Matrix in process 1:\n");
    print_matrix(&A[0][0], n, n);
  }
  MPI_Finalize();
}
```

## 2. zadatak

> Napisati MPI program kojim se elementi gornje trougaone matrice procesa 0 salju i primaju u donju trougaonu matricu procesa 1.

```
Process 0:
0       1       2       3
4       5       6       7
8       9       10      11
12      13      14      15
Process 1:
0       0       0       0
1       2       0       0
3       5       6       0
7       10      11      15
```

Gornji trougao:

```
array_of_blocklens[0]=n     // n-1
array_of_blocklens[1]=n-1
array_of_blocklens[2]=n-2
array_of_blocklens[3]=n-3

array_of_displacements[0]=0 // (n+1)*i
array_of_displacements[1]=5
array_of_displacements[2]=10
array_of_displacements[3]=15
```

Donji trougao:

```
array_of_blocklens[0]=1     // i+1
array_of_blocklens[1]=2
array_of_blocklens[2]=3
array_of_blocklens[3]=4

array_of_displacements[0]=0 // n*i
array_of_displacements[1]=4
array_of_displacements[2]=8
array_of_displacements[3]=12
```

```
void main(int argc, char *argv[]){
  int A[n][n], rank, i, j;
  int array_of_blocklens[n], array_of_displacements[n];
  MPI_Status st;
  MPI_Datatype gornji_t, donji_t;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // Za gornji trougao
  for (int i = 0; i<n; i++) {
    array_of_blocklens[i] = n - i;
    array_of_displacements[i] = (n + 1)*i;
  }
  
  // count, *array_of_blocklens, *array_of_displacements, *old_data_type, *new_data_type
  MPI_Type_indexed(n, array_of_blocklens, array_of_displacements, MPI_INT, &gornji_t);
  MPI_Type_commit(&gornji_t); // Sada mozemo da koristimo gornji_t
  
  // Za donji trougao
  for (int i = 0; i< n; i++) {
    array_of_blocklens[i] = i + 1;
    array_of_displacements[i] = n*i;
  }
  
  MPI_Type_indexed(n, array_of_blocklens, array_of_displacements, MPI_INT, &donji_t);
  MPI_Type_commit(&donji_t);
  
  if (rank == 0) {
    fill_array(&A[0][0], n*n, "i");
    // buf, count, MPI_Datatype, dest, tag, MPI_Comm
    MPI_Send(&A[0][0], 1, gornji_t, 1, 0, MPI_COMM_WORLD);
    printf("Process 0:\n");
    print_matrix(&A[0][0], n, n);
  }
  else {
    fill_array(&A[0][0], n*n, "0");
    // buf, count, MPI_Datatype, source, tag, MPI_Comm, MPI_Status)
    MPI_Recv(&A[0][0], 1, donji_t, 0, 0, MPI_COMM_WORLD, &st);
    printf("Process 1:\n");
    print_matrix(&A[0][0], n, n);
  }
  MPI_Finalize();
}
```
 
Slanje prvog izvedenog tipa: čim se posalje zadnji bajt, odmah ide slanje sledećeg.

## 3. Zadatak

> Napisati MPI program koji cita jedan podatak tipa `int` i jedan podatak tipa `double` sa standardnog ulaza u procesu `0`, a nakon toga koriscenjem izvedenih tipova podataka salje oba podatka istovremeno svim procesima. 


|||||||||||||||||
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|A|A|A|A|_|_|_|_|B|B|B|B|B|B|B|B|

```
using namespace std; 
void main(int argc, char *argv[]) {
  int rank;
  struct { int a=0; double b=0; } val;
  MPI_Datatype struktura, oldtypes[2];
  int blocklens[2];
  MPI_Aint dsp[2];
  MPI_Aint base_adr, adr1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  blocklens[0] = 1;
  oldtypes[0] = MPI_INT;
  blocklens[1] = 1;
  oldtypes[1] = MPI_DOUBLE;

  MPI_Address(&val.a, &base_adr);
  MPI_Address(&val.b, &adr1);
  dsp[1] = adr1 - base_adr;
  dsp[0] = 0;
  // count, *array_of_blocklengths, *array_of_displacements, *array_of_types, *newtype
  MPI_Type_struct(2, blocklens, dsp, oldtypes, &struktura);
  MPI_Type_commit(&struktura);

  if (rank == 0) {
    scanf("%d %d", &val.a, &val.b);
  }

  printf("[%d]: Before Bcast, val.a=%d val.b=%d\n", rank, val.a, val.b);
  // everyone calls Bcast- data is taken from root and ends up in everyone's buf
  MPI_Bcast(&val, 1, struktura, 0, MPI_COMM_WORLD);
  printf("[%d]: After Bcast, val.a=%d val.b=%d\n\n", rank, val.a, val.b);
  MPI_Finalize();
}

```