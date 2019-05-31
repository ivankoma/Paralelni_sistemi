# MPI

[MPI Tutorial](<https://mpitutorial.com>)

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

```c
int MPI_Address(const void *input_location, MPI_Aint *output_address)
```
> This is depricated function. To be able to use it, add `_CRT_SECURE_NO_WARNINGS` and `MSMPI_NO_DEPRECATE_20` to preprocessor definitions [in this way](https://stackoverflow.com/questions/16883037/remove-secure-warnings-crt-secure-no-warnings-from-projects-by-default-in-vis)
### MPI_Send

```c
int MPI_Send(
      const void *buf, //initial address of send buffer (choice) 
      int count, 
      MPI_Datatype datatype, 
      int dest,        // rank of destination (integer) 
      int tag,         // message tag (integer) 
      MPI_Comm comm)   // communicator (handle) 
```

### MPI_Recv

```c
int MPI_Recv(
      void *buf, 
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

```c
int MPI_Bcast(
      void *buffer,
      int count, 
      MPI_Datatype datatype, 
      int root, 
      MPI_Comm comm)
```

### MPI_Scatter

[Tutorial](https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/)

![Difference](https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/broadcastvsscatter.png)

```c
int MPI_Scatter(
      const void *sendbuf, 
      int sendcount, 
      MPI_Datatype sendtype,
      void *recvbuf, 
      int recvcount, 
      MPI_Datatype recvtype, 
      int root,
      MPI_Comm comm)
```

- `sendbuf` address of send buffer (choice, significant only at root) 
- `sendcount` number of elements sent to each process (integer, significant only at root) 
- `sendtype` data type of send buffer elements (significant only at root) (handle) 
- `recvcount` number of elements in receive buffer (integer) 
- `recvtype` data type of receive buffer elements
- `root` rank of sending process (integer)


### MPI_Gather

![Tutorial](https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/gather.png)

```c
int MPI_Gather(
      const void *sendbuf,
      int sendcount,
      MPI_Datatype sendtype,
      void *recvbuf,
      int recvcount,
      MPI_Datatype recvtype,
      int root, 
      MPI_Comm comm)
```

- `sendbuf` starting address of send buffer (choice) 
- `sendcount` number of elements in send buffer (integer) 
- `sendtype` data type of send buffer elements (handle) 
- `recvcount` number of elements for any single receive (integer, significant only at root) 
- `recvtype` data type of recv buffer elements (significant only at root) (handle) 
- `root` rank of receiving process (integer) 
- `comm` communicator (handle) 

### MPI_type_struct

```c
int MPI_Type_struct(
      int count,
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

```c
struct{char a, int b, double c} val;

MPI_Address(&val.a, &baseadr);  // array_of_types[0] is MPI_CHAR
MPI_Address(&val.b, &adr_b);    // array_of_types[1] is MPI_INT
MPI_Address(&val.c, &adr_c);    // array_of_types[2] is MPI_DOUBLE

array_of_displacements[0]=0 // pomeraj prvog bloka u odnosu na početak strukture
array_of_displacements[1]=adr_b-baseadr
array_of_displacements[2]=adr_c-baseadr
```

### MPI_Type_vector

Omogucava da formiramo izveden tip podatka gde su blokovi koji čine izveden tip iste veličine. Poceči blokova se nalaze na *jednakim rastojanjima*.

```c
int MPI_Type_vector(
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

```c
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

Modification:

```
Matrix in process 0
0       1       2       3
4       5       6       7
8       9       10      11
12      13      14      15
Matrix in process 1
0       4       0       0
0       5       0       0
0       6       0       0
0       7       0       0
Press any key to continue . . .
...
MPI_Type_vector(n, 1, n, MPI_INT, &kolona);
MPI_Type_commit(&kolona);

MPI_Type_vector(n, 1, 1, MPI_INT, &red);
MPI_Type_commit(&red);
...
MPI_Send(&A[1][0], 1, red, 1, 0, MPI_COMM_WORLD);
MPI_Recv(&A[0][1], 1, kolona, 0, 0, MPI_COMM_WORLD, &st);
```

## 2. zadatak

> Napisati MPI program kojim se elementi gornje trougaone matrice procesa 0 salju i primaju u donju trougaonu matricu procesa 1.

```c
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

```c
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

```c
array_of_blocklens[0]=1     // i+1
array_of_blocklens[1]=2
array_of_blocklens[2]=3
array_of_blocklens[3]=4

array_of_displacements[0]=0 // n*i
array_of_displacements[1]=4
array_of_displacements[2]=8
array_of_displacements[3]=12
```

```c
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

## 3. Zadatak

> Napisati MPI program koji cita jedan podatak tipa `int` i jedan podatak tipa `double` sa standardnog ulaza u procesu `0`, a nakon toga koriscenjem izvedenih tipova podataka salje oba podatka istovremeno svim procesima. 


|||||||||||||||||
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|A|A|A|A|_|_|_|_|B|B|B|B|B|B|B|B|

```c
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
  dsp[0] = 0;
  dsp[1] = adr1 - base_adr;
  
  // count, *array_of_blocklengths, *array_of_displacements, *array_of_types, *newtype
  MPI_Type_struct(2, blocklens, dsp, oldtypes, &struktura); // 2 because of int and double
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

Modification:

```c
using namespace std; 
void main(int argc, char *argv[]) {
  struct {
    int a = -1;
    double b = -1;
    float c[2] = { -1.5, -1.5 };  // array
  } val;
  const int DIFFERENT_DATATYPES = 3;

  int rank;
  MPI_Datatype nova_struktura, old_types[DIFFERENT_DATATYPES];
  MPI_Aint dsp[DIFFERENT_DATATYPES];
  int block_lens[DIFFERENT_DATATYPES];
  ...
  block_lens[0] = 1;
  block_lens[1] = 1;
  block_lens[2] = 2;  // 2 because of the array

  old_types[0] = MPI_INT;
  old_types[1] = MPI_DOUBLE;
  old_types[2] = MPI_FLOAT;

  MPI_Address(&val.a, &base_adr);
  MPI_Address(&val.b, &adr_b);
  MPI_Address(&val.c, &adr_c0);
  dsp[0] = 0;
  dsp[1] = adr_b - base_adr;
  dsp[2] = adr_c0 - base_adr;

  MPI_Type_struct(DIFFERENT_DATATYPES, block_lens, dsp, old_types, &nova_struktura);
  MPI_Type_commit(&nova_struktura);
  ...
```

---

## MPI 2. nedelja

*Slanje prvog izvedenog tipa: čim se pošalje zadnji bajt, odmah ide slanje sledećeg (počevši od jedan bajt posle kraja posledneg izvedenog tipa).*

Primer izvedenog tipa podatka:

```c
MPI_Type_vector(3,2,4,MPI_INT,&izvtip);
```

|1||||||||||||
|-|-|-|-|-|-|-|-|-|-|-|-|
|x|x|_|_|x|x|_|_|x|x|_|_|

Bez praznih mesta (bajtova):

|1||||||||||2||||||||||3||||||||||
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|x|x|_|_|x|x|_|_|x|x|x|x|_|_|x|x|_|_|x|x|x|x|_|_|x|x|_|_|x|x|_|_|x|x|

Ako hocemo sa praznim:

|1||||||||||||2||||||||||||3||||||||||||
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|x|x|_|_|x|x|_|_|x|x|_|_|x|x|_|_|x|x|_|_|x|x|_|_|x|x|_|_|x|x|_|_|x|x|_|_|

```c
MPI_Type_create_resized(MPI_Datatype it, 
  MPI_AInt lb,
  MPI_AInt extent, 
  MPI_Datatype *newtype
  )
```

- `lb` Nova donja granica tipa koja odgovara najmanjem displacement-u podatka u starom tipu. Obično je 0.
- `extent` (rastojanje) New extent of datatype (address integer). Utiče na to odakle će krenuti slanje svake sledeće jedinice novog tipa. Izrazen u bajtovima. Jednaka najmanjem displacement-u u tipu.
- `newtype` Output. Novi izvedeni Tip

`lb + extend = gornja granica`

```c
MPI_Type_vector(...);
MPI_Type_create_resized(izvtip, 0, 12 * sizeof(int));
MPI_Send(buff, 10 , izvtip, 1, 0, MPI_COMM_WORLD);
```

### Send a column to each process.

|   |   |   |   |
|---|---|---|---|
|**a00**|a01|a02|a03|
|**a10**|a11|a12|a13|
|**a20**|a21|a22|a23|
|**a30**|a31|a32|a33|

```c
MPI_Type_vector(n,1,n,MPI_INT,&kolona);
MPI_Type_create_resized(kolona, 0, sizeof(int), &nkolona)
MPI_Scatter(&a[0][0], 1, kolona, ...);
```

### Send to each process n/p columns

`n=6, p=3`

|p0|p0|p1|p1|p2|p2|
|---|---|---|---|---|---|
|x|x|x|x|x|x|
|x|x|x|x|x|x|
|x|x|x|x|x|x|
|x|x|x|x|x|x|
|x|x|x|x|x|x|
|x|x|x|x|x|x|

```c
MPI_Type_vector(n, n/p, n, MPI_INT, &kolone)
MPI_Type_create_resized(kolone, 0, (n/p)*sizeof(int), &nkolona);
MPI_Scatter(&A[0][0], 1, nkolona...);
```

## Zadatak 1

> Napisati MPI program koji realizuje množenje matrice *A*, n*n, i vektora *Bn*, čime se dobija rezultujući vektor *Cn*. Matrica *A* i vektor *Bn* se inicijalizuju u master procesu. Matrica *A* je podeljena u blokove po vrstama i to tako da proces *Pi* dobija vrste sa indeksima *L*, gde je `L mod p=i (0<=i<=p-1)` tj. vrste sa indeksima `i, i+p, i+2p,..., i+n-p`. Master proces distribuira blokove matrice *A* i ceo vektor *B* svim procesima. Slanje svakog bloka matrice *A* se obavlja odjednom. Svaki proces učestvuje u izračunavanju rezultata koji se prikazuje u master procesu.

```
A:
0       1       2       3       4       5
6       7       8       9       10      11
12      13      14      15      16      17
18      19      20      21      22      23
24      25      26      27      28      29
30      31      32      33      34      35

B:
0 2 4 6 8 10

local_C: 110 290 470 650 830 1010
```

![multiplication](img/multiplication.png)

*broj procesa p=3, n=6:*

*P=0 dobija L=0 i L=3 vrstu* `0 mod 3 = 0; 3 mod 3 = 0` tj. `0, 0+3`

*P=1 dobija L=1 i L=4 vrstu* `1 mod 3 = 1; 4 mod 3 = 1` tj. `1, 1+3`

*P=2 dobija L=2 i L=5 vrstu* `2 mod 3 = 2; 5 mod 3 = 2` tj. `2, 2+3`


*An\*n*

|i|P|||||||
|---|---|---|---|---|---|---|---|
|0|**P0**|x|x|x|x|x|x|
|1|P1|x|x|x|x|x|x|
|2|*P2*|x|x|x|x|x|x|
|3|**P0**|x|x|x|x|x|x|
|4|P1|x|x|x|x|x|x|
|5|*P2*|x|x|x|x|x|x|

`*`

|vektor *Bn*|
|---|
|x|
|x|
|x|
|x|
|x|

`=`

|vektor *Cn*|
|---|
|x|
|x|
|x|
|x|
|x|

```c
MPI_Type_vector(n/p, n, p*n, MPI_INT, &vrste)  // 1 new datatype has two whole rows
MPI_Type_create_resize(&vrste, 0, n*sizeof(int), &nvrste)
```

```c
void main(int argc, char * argv[]) {
  int A[n][n], B[n], C[n], rank, i, j;
  MPI_Datatype rows, separated_rows, new_type, fixed_new_type;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //MPI_Comm_size(MPI_COMM_WORLD, &p);
  int * local_A = (int*)malloc((n / p) * n * sizeof(int));
  int * local_C = (int*)malloc((n / p) * sizeof(int));

  MPI_Type_vector(n / p, n, p*n, MPI_INT, &rows);
  MPI_Type_create_resized(rows, 0, n * sizeof(int), &separated_rows);
  MPI_Type_commit(&separated_rows);

  if (rank == 0) {
    printf("A:\n");
    fill_array(&A[0][0], n*n, "i", rank);
    print_matrix(&A[0][0], n, n);

    printf("\nB:\n");
    fill_array(&B[0], n, "2*i");
    print_array(&B[0], n);
    printf("\n");
  }

  // MPI_Scatter(send_buffer, send_count, send_type, recv_buffer, recv_count, recv_type, rank, comm ) 
  MPI_Scatter(&A[0][0], 1, separated_rows, local_A, (n / p) * n, MPI_INT, 0, MPI_COMM_WORLD);
  // MPI_Bcast(buffer, count, datatype, rank, comm)
  MPI_Bcast(B, n, MPI_INT, 0, MPI_COMM_WORLD);

  for (int i = 0; i < n / p; i++) { // 0 and 1, for n=6 and p=3
    local_C[i] = 0;
    for (int j = 0; j < n; j++) { // 0..5
      local_C[i] += local_A[i*n + j] * B[j];
      // if (rank == 0) { printf("rank=%d i=%d %d*%d\n", rank, i, local_A[i*n + j], B[j]);}
      /*      p0  p1  p2        p0  p1  p2
      local_c[0]  x   x   x  local_c[1]  x   x   x
      */
    }
  }
  
  // printf(">> rank=%d:\n\tlocal_A[0]=%d local_A[1]=%d\n\tlocal_A[n]=%d local_A[n+1]=%d\n\tlocal_C[0]=%d local_C[1]=%d\n", rank, local_A[0], local_A[1], local_A[n], local_A[n+1], local_C[0], local_C[1]);

  // Error! This way it would be 110 650 290 830 470 1010
  // MPI_Gather(send_buffer, send_count, send_type, recv_buffer, recv_count, recv_type, rank, comm)
  // MPI_Gather(local_C, n / p, MPI_INT, C, 2, MPI_INT, 0, MPI_COMM_WORLD); [12:30]

  // MPI_Type_vector(count, blocklength, stride, oldtype, newtype)
  MPI_Type_vector(n/p, 1, p, MPI_INT, &new_type); // [14:50]  [1][ ][ ][1]
  // MPI_Type_create_resized(oldtype, lb, extent, newtype)  
  MPI_Type_create_resized(new_type, 0, sizeof(int), &fixed_new_type);  // [15:50] [1][2][3][1][2][3]
  MPI_Type_commit(&fixed_new_type);

  // MPI_Gather(send_buffer, send_count, send_type, recv_buffer, recv_count, recv_type, rank, comm)
  MPI_Gather(local_C, n/p, MPI_INT, C, 1, fixed_new_type, 0, MPI_COMM_WORLD);
  
  if (rank == 0) {
    printf("local_C: ");
    print_array(&C[0], n);
  }
  MPI_Finalize();
}
```

## Zadatak 2

> Napisati MPI program koji realizuje množenje matrica *An\*n* i *Bn\*n* čime se dobija rezultujuca matrica *Cn\*n*. Množenje se obavlja tako sto master proces inicijalizuje matrice *A* i *B* i šalje svakom procesu po jednu kolonu matrice *A* i jednu vrstu *B*. Svi procesi učestvuju u izračunavanju a rezultat se nalazi i prikazuje u procesu sa rankom 0.

```
A:
1       2       3
4       5       6
7       8       9

B:
1       3       5
7       9       11
13      15      17

C:
54      66      78
117     147     177
180     228     276

vvvvvvvvvvvvvvvv
rank=0
        [1]
        [4]
        [7]

*       [1 3 5]

=       [1 3 5]
        [4 12 20]
        [7 21 35]
^^^^^^^^^^^^^^^^
vvvvvvvvvvvvvvvv
rank=1
        [2]
        [5]
        [8]

*       [7 9 11]

=       [14 18 22]
        [35 45 55]
        [56 72 88]
^^^^^^^^^^^^^^^^
vvvvvvvvvvvvvvvv
rank=2
        [3]
        [6]
        [9]

*       [13 15 17]

=       [39  45   d51]
        [78  90  102]
        [117 135 153]
^^^^^^^^^^^^^^^^
```
|||
|-|-|
|p0|p1|
|**a00**|a01|
|**a10**|a11|

`*`

||||
|-|-|-|
|p0|**b00**|**b01**|
|p1|b10|b11|

`=`

|||
|-|-|
|c00|c01|
|c10|c11|

```
a00*b00 + a01*b10   a00*b01 + a01*b11
a10*b00 + a11*b10   a10*b01 + a11*b11
```

```c
void main(int argc, char * argv[]) {
  int A[n][n], B[n][n], C[n][n], tmp[n][n], one_row[n], one_col[n];
  int rank, i, j;
  MPI_Datatype col, succesive_col;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    fill_array(&A[0][0], n*n, "i", 0, 1);
    fill_array(&B[0][0], n*n, "2*i", 0, 1);
    printf("A:\n");
    print_matrix(&A[0][0], n, n);
    printf("\nB:\n");
    print_matrix(&B[0][0], n, n);
  }

  MPI_Type_vector(n, 1, n, MPI_INT, &col);
  MPI_Type_create_resized(col, 0, sizeof(int), &succesive_col);
  MPI_Type_commit(&succesive_col);
  MPI_Scatter(&A[0][0], 1, succesive_col, one_col, n, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatter(&B[0][0], n, MPI_INT, one_row, n, MPI_INT, 0, MPI_COMM_WORLD);

  for (i = 0; i<n; i++) {
    for (j = 0; j<n; j++) {
      tmp[i][j] = one_col[i] * one_row[j];
    }
  }
  // MPI_Reduce (send_buffer, recv_buffer, count, datatype, operation, rank, comm)
  MPI_Reduce(&tmp[0][0], &C[0][0], n*n, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("\nC:\n");
    print_matrix(&C[0][0], n, n);
  }
  MPI_Finalize();
}
```

---