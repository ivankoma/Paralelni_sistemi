# Laboratorijske vežbe - MPI

## Lab 1

### Zadatak 1 & 2

> Napisati MPI program koji pronalazi minimalnu vrednost u delu matrice reda n (n-parno) koga čine kolone matrice sa parnim indeksom (j=0,2,4,...). Matrica je inicijalizovana u master procesu (P0). Svaki proces treba da dobije elemente kolona sa parnim indeksom iz odgovarajućih n/p vrsta (p-broj procesa, n deljivo sa p) i nađe lokalni minimum. Na taj način, P0 dobija elemente kolona sa parnim indeksom iz prvih n/p vrsta i nalazi lokalni minimum, P1 dobija elemente kolona sa parnim indeksom iz sledećih n/p vrsta i nalazi lokalni minimum itd. Nakon toga, u master procesu se izračunava i na ekranu prikazuje globalni minimum u traženom delu matrice. Zadatak realizovati korišćenjem isključivo grupnih operacija i izvedenih tipova podataka.

```
n=6
p=3
n/p=2, po 2 vrste dobija proces
```

|0|1|2|3|4|5|
|---|---|---|---|---|---|
|**P1**|x|**P1**|x|**P1**|x|
|**P1**|x|**P1**|x|**P1**|x|
|P2|x|P2|x|P2|x|
|P2|x|P2|x|P2|x|
|*P3*|x|*P3*|x|*P3*|x|
|*P3*|x|*P3*|x|*P3*|x|

```
n=6
p=2
n/p=3, po 2 vrste dobija proces
```

|||||||
|---|---|---|---|---|---|
|**P1**|x|**P1**|x|**P1**|x|
|**P1**|x|**P1**|x|**P1**|x|
|**P1**|x|**P1**|x|**P1**|x|
|P2|x|P2|x|P2|x|
|P2|x|P2|x|P2|x|
|P2|x|P2|x|P2|x|

```
-10     0       -7      0       -9      0
-19     0       -16     0       12      0
-16     0       -16     0       0       0
2       0       -1      0       6       0
12      0       -9      0       2       0
-18     0       -7      0       5       0

rank=0
        -10     -7      -9
        -19     -16     12
        local_min=-19

rank=1
        -16     -16     0
        2       -1      6
        local_min=-16
        
rank=2
        12      -9      2
        -18     -7      5
        local_min=-18
        
>>>global_min=-19<<<
```

### 2

> Napisati MPI program koji pronalazi maksimalnu vrednost u delu matrice reda n (n-parno) koga čine vrste matrice sa parnim indeksom i=0,2,4,...). Matrica je inicijalizovana u master procesu (P0). Svaki proces treba da dobije elemente vrsta sa parnim indeksom iz odgovarajućih n/p kolona (p-broj procesa, n deljivo sa p) i nađe lokalni maksimum. Na taj način, P0 dobija elemente vrsta sa parnim indeksom iz prvih n/p kolona i nalazi lokalni maksimum, P1 dobija elemente kolona sa parnim indeksom iz sledećih n/p kolona i nalazi lokalni maksimum itd. Nakon toga, u master procesu se izračunava i na ekranu prikazuje globalni maksimum u traženom delu matrice. Zadatak realizovati korišćenjem isključivo grupnih  operacija i izvedenih tipova podataka.

||||||||
|---|---|---|---|---|---|---|
|**0**|**P0**|**P0**|P1|P1|*P2*|*P2*|
|**1**|x|x|x|x|x|x|
|**2**|**P0**|**P0**|P1|P1|*P2*|*P2*|
|**3**|x|x|x|x|x|x|
|**4**|**P0**|**P0**|P1|P1|*P2*|*P2*|
|**5**|x|x|x|x|x|x|

```
-6      6       -16     13      9       10
0       0       0       0       0       0
13      -6      -1      -17     -4      11
0       0       0       0       0       0
-11     12      -4      16      -1      3
0       0       0       0       0       0

rank=0
        -6      6
        13      -6
        -11     12
        local_min=-11

rank=1
        -16     13
        -1      -17
        -4      16
        local_min=-17

rank=2
        9       10
        -4      11
        -1      3
        local_min=-4
        
>>>global_min=-17<<<
```

```
void main(int argc, char * argv[]) {
  const int lab = 2; // or 2
  int A[n][n], local_A[n / p * n / 2], global_min, rank;
  MPI_Datatype dt, succesive_dt;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    fill_array(&A[0][0], n*n, "random", 20);
    // optional
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (lab == 1) { if (j % 2 == 1) A[i][j] = 0; }  // because these values are not important
        else if (i % 2 == 1) A[i][j] = 0;  // because these values are not important
      }
    }
    print_matrix(&A[0][0], n, n);
  }

  if (lab == 1) {
    MPI_Type_vector((n / p) * n / 2, 1, 2, MPI_INT, &dt);
    MPI_Type_create_resized(dt, 0, (n / p) * n * sizeof(int), &succesive_dt);
  }else {
    MPI_Type_vector(0.5 * (n / p) * n / 2, 2, 2 * n, MPI_INT, &dt);
    MPI_Type_create_resized(dt, 0, (n / p) * sizeof(int), &succesive_dt);
  }
  MPI_Type_commit(&succesive_dt);

  MPI_Scatter(&A[0][0], 1, succesive_dt, local_A, (n / p) * n / 2, MPI_INT, 0, MPI_COMM_WORLD);

  int local_min = local_A[0];
  for (int i = 1; i < (n / p) * n / 2; i++) {
    if (local_A[i] < local_min) {
      local_min = local_A[i];
    }
  }
  if (lab == 1) {
    printf("\nrank=%d\n\t%d\t%d\t%d\n\t%d\t%d\t%d\n\tlocal_min=%d\n", rank, local_A[0], local_A[1], local_A[2], local_A[3], local_A[4], local_A[5], local_min);
  }
  else {
    printf("\nrank=%d\n\t%d\t%d\n\t%d\t%d\n\t%d\t%d\n\tlocal_min=%d\n", rank, local_A[0], local_A[1], local_A[2], local_A[3], local_A[4], local_A[5], local_min);
  }

  MPI_Reduce(&local_min, &global_min, p, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("\n>>>global_min=%d<<<\n", global_min);
  }

  MPI_Finalize();
}
```

### Zadatak 3 & 4

> Proces 0 kreira matricu reda n i šalje i-om procesu po dve kvazidijagonale matrice, obe na udaljenosti i od glavne dijagonale. Proces i kreira svoju matricu tako što smešta primljene dijagonale u prvu i drugu kolonu matrice a ostala mesta popunjava nulama. Napisati MPI program koji realizuje opisanu komunikaciju korišćenjem izvedenih tipova podataka i prikazuje vrednosti odgovarajućih kolona.

*Valjda se ovo traži:*

`n=6`

`p=4`

|||||||
|-|-|-|-|-|-|
|**d**|1a|**2a**|3|4||
|1A|**d**|1b|**2b**|3|4|
|**2A**|1B|**d**|1c|**2c**|3|
|3|**2B**|1C|**d**|1d|**2d**|
|4|3|**2C**|1D|**d**|1e|
||4|3|**2D**|1E|**d**|

P1:

|||||||
|-|-|-|-|-|-|
|1A|1a|0|0|0|0|
|1B|1b|0|0|0|0|
|1C|1c|0|0|0|0|
|1D|1d|0|0|0|0|
|1E|1e|0|0|0|0|
|0|0|0|0|0|0|

P2:

|||||||
|-|-|-|-|-|-|
|2A|2a|0|0|0|0|
|2B|2b|0|0|0|0|
|2C|2c|0|0|0|0|
|2D|2d|0|0|0|0|
|0|0|0|0|0|0|
|0|0|0|0|0|0|

```
A:
0       1       2       3       4       5
6       7       8       9       10      11
12      13      14      15      16      17
18      19      20      21      22      23
24      25      26      27      28      29
30      31      32      33      34      35
rank=1
1       6       0       0       0       0
8       13      0       0       0       0
15      20      0       0       0       0
22      27      0       0       0       0
29      34      0       0       0       0
0       0       0       0       0       0
rank=2
2       12      0       0       0       0
9       19      0       0       0       0
16      26      0       0       0       0
23      33      0       0       0       0
0       0       0       0       0       0
0       0       0       0       0       0
rank=3
3       18      0       0       0       0
10      25      0       0       0       0
17      32      0       0       0       0
0       0       0       0       0       0
0       0       0       0       0       0
0       0       0       0       0       0
```

> Proces 0 kreira matricu reda n i šalje i-om procesu po dve kvazidijagonale matrice, obe na udaljenosti i od sporedne dijagonale. Proces i kreira svoju matricu tako što smešta primljene dijagonale u prvu i drugu vrstu matrice a ostala mesta popunjava nulama. Napisati MPI program koji realizuje opisanu komunikaciju korišćenjem izvedenih tipova podataka i prikazuje vrednosti odgovarajućih kolona.

|||||||
|-|-|-|-|-|-|
||4A|3A|**2A**|1A|**sd**|
|4B|3B|**2B**|1B|**sd**|1a|
|3C|**2C**|1C|**sd**|1b|**2a**|
|**2D**|1D|**sd**|1c|**2b**|3a|
|1E|**sd**|1d|**2c**|3b|4a|
|**sd**|1ee|**2d**|3c|4b||

P1:

|||||||
|-|-|-|-|-|-|
|1A|1B|1C|1D|1E|0|
|1a|1b|1c|1d|1e|0|
|0|0|0|0|0|0|
|0|0|0|0|0|0|
|0|0|0|0|0|0|
|0|0|0|0|0|0|

```c
#define n 6
#define p 4 // number of processes
using namespace std; 

void main(int argc, char * argv[]) {
  const int zad = 3;  // or 4
  int A[n][n], local_A[n][n], rank;
  MPI_Datatype diagonal, succesive_diagonal;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int * above_diag = (int*)malloc((n - rank) * sizeof(int));
  int * under_diag = (int*)malloc((n - rank) * sizeof(int));

  if (rank == 0) {
    fill_array(&A[0][0], n*n, "i");
    printf("A:\n");
    print_matrix(&A[0][0], n, n);
  }

  if (zad == 3) {
    MPI_Type_vector(n, 1, n + 1, MPI_INT, &diagonal);
  }
  else {
    MPI_Type_vector(n, 1, n - 1, MPI_INT, &diagonal);
  }
  MPI_Type_commit(&diagonal);

  if (rank == 0) {
    if (zad == 3) {
      for (int i = 1; i < p; i++) {
        MPI_Send(&A[0][i], 1, diagonal, i, 0, MPI_COMM_WORLD);
        MPI_Send(&A[i][0], 1, diagonal, i, 0, MPI_COMM_WORLD);
      }
    }
    else {
      for (int i = 1; i < p; i++) {
        MPI_Send(&A[0][n - 1 - i], 1, diagonal, i, 0, MPI_COMM_WORLD);
        MPI_Send(&A[i][n - 1], 1, diagonal, i, 0, MPI_COMM_WORLD);
      }
    }

  }
  else {
    MPI_Recv(above_diag, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
    MPI_Recv(under_diag, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

    fill_array(&local_A[0][0], n*n, "0");
    printf("rank=%d\n", rank);
    if (zad == 3) {
      for (int i = 0; i < n - rank; i++) {
        local_A[i][0] = above_diag[i];
        local_A[i][1] = under_diag[i];
      }
    }
    else {
      for (int i = 0; i < n - rank; i++) {
        local_A[0][i] = above_diag[i];
        local_A[1][i] = under_diag[i];
      }
    }

    print_matrix(&local_A[0][0], n, n);
  }

  MPI_Finalize();
}
```

### Zadatak 5


> Napisati MPI program koji realizuje množenje matrice *Am\*n* i matrice *Bn\*k*, čime se dobija rezultujuća matrica *Cm\*k*. Množenje se obavlja tako što master proces šalje svakom procesu celu matricu A i po *k/p* kolona matrice *B* (*p* broj procesa, *k* je deljivo sa *p*). Svi procesi učestvuju u izračunavanju. Konačni rezultat množenja se nalazi u master procesu koji ga i prikazuje. Predvideti da se slanje *k/p* kolona matrice *B* svakom procesu obavlja odjednom i to direktno iz matrice *B*. Zadatak rešiti korišćenjem grupnih operacija i izvedenih tipova podataka.

p=2
A m=3 n=5

||||||
|-|-|-|-|-|
||||||
||||||
||||||

B n=5 k=4

|||||
|-|-|-|-|
|p1|p1|p2|p2|
|p1|p1|p2|p2|
|p1|p1|p2|p2|
|p1|p1|p2|p2|
|p1|p1|p2|p2|

C m=3 k=4

|||||
|-|-|-|-|
|||||
|||||
|||||

```
A:
0       1       2       3       4
5       6       7       8       9
10      11      12      13      14

B:
0       1       2       3
4       5       6       7
8       9       10      11
12      13      14      15
16      17      18      19

rank=0 C:
120     130     0       0
320     355     0       0
520     580     0       0

rank=1 C:
0       0       140     150
0       0       390     425
0       0       640     700

C:
120     130     140     150
320     355     390     425
520     580     640     700
```

```c
#define p 2 // number of processes

void main(int argc, char * argv[]) {
  const int m = 3, n = 5, k = 4;
  int A[m][n], B[n][k], C[m][k], local_C[m][k], local_B[n*(k / p)];
  int rank;
  MPI_Datatype custom_col, succesive_custom_col;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    fill_array(&A[0][0], m*n, "i");
    fill_array(&B[0][0], n*k, "i");
    printf("\nA:\n");
    print_matrix(&A[0][0], m, n);
    printf("\nB:\n");
    print_matrix(&B[0][0], n, k);
  }

  MPI_Bcast(&A[0][0], m*n, MPI_INT, 0, MPI_COMM_WORLD);
  
  MPI_Type_vector(n, k/p, k, MPI_INT, &custom_col);
  MPI_Type_create_resized(custom_col, 0, (k / p) * sizeof(int), &succesive_custom_col);
  MPI_Type_commit(&succesive_custom_col);

  MPI_Scatter(&B[0][0], 1, succesive_custom_col, &local_B, n * (k / p), MPI_INT, 0, MPI_COMM_WORLD);

  fill_array(&local_C[0][0], m*k, "0");
  
  for (int y = 0; y < m; y++) {
    int tmp = 0;
    for (int x = 0; x < k/p; x++) {
      tmp = 0;
      for (int i = 0; i < n; i++) {
        tmp += A[y][i] * local_B[2 * i + x];
      }
      local_C[y][x + rank*(k/p)] = tmp;
    }
  }
  printf("\nrank=%d C:\n", rank);
  print_matrix(&local_C[0][0], m, k);
  
  MPI_Reduce(&local_C[0][0], &C[0][0], m*k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("\nC:\n");
    print_matrix(&C[0][0], m, k);
  }

  MPI_Finalize();
}
```

### Zadatak 6

> Napisati MPI program koji realizuje množenje matrica A i B reda n, čime se dobija rezultujuća matrica C. Nakon toga, u matrici C pronaći maksimalnu vrednost elemenata svake kolone. Množenje se obavlja tako što master proces šalje svakom procesu radniku po jednu vrstu prve matrice i celu drugu matricu. Svaki proces računa po jednu vrstu
rezultujuće matrice i šalje master procesu. Svi procesi učestvuju u izračunavanju. Štampati dobijenu matricu kao i maksimalne vrednosti elemenata svake kolone. Zadatak rešiti korišćenjem grupnih operacija.

```
A:
0       1       2       3
4       5       6       7
8       9       10      11
12      13      14      15

B:
0       2       4       6
8       10      12      14
16      18      20      22
24      26      28      30

C:
112     124     136     148
304     348     392     436
496     572     648     724
688     796     904     1012
```

```c
void main(int argc, char * argv[]) {
  const int n = p;
  int A[n][n], B[n][n], C[n][n], local_C[n], local_A_row[n], rank;
  MPI_Datatype row;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    fill_array(&A[0][0], n*n, "i");
    fill_array(&B[0][0], n*n, "i");
    printf("\nA:\n");
    print_matrix(&A[0][0], n, n);
    printf("\nB:\n");
    print_matrix(&B[0][0], n, n);
  }
  MPI_Type_vector(n, 1, 1, MPI_INT, &row);
  MPI_Type_commit(&row);
  MPI_Bcast(&B[0][0], n*n, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatter(&A[0][0], 1, row, local_A_row, n, MPI_INT, 0, MPI_COMM_WORLD);

  fill_array(&local_C[0], n,"0");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      local_C[i] += local_A_row[j] * B[j][i];
    }
  }

  MPI_Gather(local_C, n, MPI_INT, &C[0][0], n, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("\nC:\n");
    print_matrix(&C[0][0], n, n);
  }

  MPI_Finalize();
}
```