# OpenMP

Notes for Intel's YouTube course called [Introduction to OpenMP - Tim Mattson](https://www.youtube.com/watch?v=nE-xN4Bf8XI&list=PLLX-Q6B8xqZ8n8bwjGdzBJ25X2utwnoEG)

```c
#include <stdio.h>
#include <omp.h>
int main()
{
#pragma omp parallel
    {
    int ID = omp_get_thread_num();
    printf("hello(%d) ", ID);
    printf("world(%d) \n", ID);
    }
    return 0;
}
```

### Calculate Pi

Serial:

```c
#include <stdio.h>
#include <omp.h>
static long num_steps = 100000000;
double step;
#define NUM_THREADS 2
int main()
{
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)num_steps;
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    printf("pi=%f", pi);
    return 0;
}
```

OpenMP:

```c
#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;
double step;
#define NUM_THREADS 2
int main()
{
    double start_time = omp_get_wtime();
    int i, nthreads;
    double x, pi, sum[NUM_THREADS];
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel
    {
        int i, actual_num_threads, id = omp_get_thread_num();
        double x;
        actual_num_threads = omp_get_num_threads();
        nthreads = omp_get_num_threads();
        if (id == 0) nthreads = actual_num_threads;
        sum[id] = 0.0;
        for (int i = 0; i < num_steps; i=i+nthreads) {
            x = (i + 0.5) * step;
            sum[id] = sum[id] + 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < nthreads; i++) {
        pi += sum[i] * step;
    }
    printf("pi=%f\n", pi);
    printf("time=%f", omp_get_wtime() - start_time);
    return 0;
}
```

## Synchronization

### Barrier

`barrier` will enforce every thread to wait at the barrier until all threads have reached the barrier.

```c
#pragma omp parallel
{
    int i = omp_get_thread_num();
    A[id] = big_calc1(id);
#pragma omp barrier             // Every thread has to finish before continuing to the next statement at the same time
    B[id] = big_calc2(id);
}
```

### Critical

Mutual exclusion. Use this because it is more general.
All threads execute the code, but only one at a time, one after another.

```c
float res;
#pragma omp parallel
{
    float B;
    int i, id, nthrds;
    id = omp_get_thread_num();
    nthrds = omp_get_num_threads();
    for(i=id;i<niters;i+=nthrds){
        B = big_job(i);
#pragma omp critical
//{
    res += consume(B);
//}        
 }
}
```

Better example. 

Sum of `1 + 2 + 3 + 4 + 5 + 6 = 21`:

```c
#define NUM_THREADS 3
int main()
{
    int i, sum=0;
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel private(i)
{
    int local_sum = 0;
#pragma omp for
    for (i = 1; i < 7; i++) {
        printf("Thread %d: local_sum(%d)+=%d\n", omp_get_thread_num(), local_sum,i);
        local_sum += i;
    }
#pragma omp critical (optional_name)

    printf("Thread %d: sum(%d)+=local_sum(%d)\n", omp_get_thread_num(), sum, local_sum);
    sum += local_sum; // This can also be calculated using `atomic`

}
printf("After parallel region: sum=%d", sum);
    return 0;
}
```

Output:

```c
Thread 0: local_sum(0)+=1
Thread 2: local_sum(0)+=5
Thread 2: local_sum(5)+=6
Thread 1: local_sum(0)+=3
Thread 1: local_sum(3)+=4
Thread 0: local_sum(1)+=2
Thread 1: sum(0)+=local_sum(7)
Thread 0: sum(7)+=local_sum(3)
Thread 2: sum(10)+=local_sum(11)
After parallel region: sum=21
```

## Atomic

Shortcut to Critical. Only used for the update of a memory location. Can only execute one operation.

`x++``++x``x--``--x`

```c
#pragma omp parallel
{
    double tmp, B;
    B = do_it();
    tmp = big_ugly(B);
#pragma omp atomic
    x += tmp;
}
```

### The OpenMP 3.1 Atomics

Atomic was expanded to cover the full range of common scenarios where you need to protect a memory operation so it occurs atomically.

```c
#pragma omp atomic[read|write|update|capture]
```

Atomic can protect loads

```c
#pragma omp atomic read
v = x; // Either will update v with whole value of x or won't update at all 
```

## Calculate Pi with optimization

```c
#include <stdio.h>
#include <omp.h>

static long num_steps = 100000000;
double step;
#define NUM_THREADS 2
int main()
{
    double start_time = omp_get_wtime();
    double pi = 0.0;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
{
        int i, id, nthrds, nthreads;
        double x, sum;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        for (i = id, sum = 0.0; i < num_steps; i = i + nthrds){
            x = (i + 0.5) * step;
            // NOT HERE! #pragma omp critical
            sum += 4.0 / (1.0 + x * x);
        }
#pragma omp critical
//#pragma omp atomic    // same in this case
        pi += sum * step;
}
    printf("pi=%f\n", pi);
    printf("time=%f", omp_get_wtime() - start_time);
    return 0;
}
```

## Worksharing

### Loop Construct

```c
#pragma omp parallel
{
#pragma omp for
    for (i=0;i<n;i++){
        do();
    }    
}
```

is the same as:

```c
#pragma omp parallel omp for
    for (i=0;i<n;i++){
        do();
    }
```

### Schedule

- `schedule(static [,chunk])` Iterations are pre-determined and predictable by the Programmer. Least work at runtime: scheduling done at compile time.

- `schedule(dynamic[,chunk])` Iterations are unpredictable, highly variable work per iteration. Most work at runtime: complex scheduling logic used at runtime.

- `schedule(guided[,chunk])`

- `schedule(runtime)`

- `schedule(auto)`

```c
omp_set_schedule();
omp_get_schedule();
```

### Working With Loops

This program is unparallelizable:

```c
int i, j, A[MAX];
j = 5;
for(i=0;i<MAX;i++){
    j += 2;
    A[i] = big(j);
}
```

There is no **loop-carry dependancy** here:

```c
int i, j, A[MAX];
#pragma omp parallel for
for(i=0;i<MAX;i++){
    int j = 5 + 2*(i+1);
    A[i] = big(j);
}
```

#### Reduction

```c
double avg=0.0, A[MAX];
int i;
for(i=0;i<MAX;i++){
    avg += A[i];
}
avg = avg / MAX;
```

`reduction (op : list)`

Local copy of each list variable is made (locally in each thread) and initialized depending on the `op`. 
Updates occur on the local copy. When done, local copies are reduced into a single value and combined with the original global value. 
There is no need to explicitly define the variable as `shared`. 
You can't do other operations with the variable except the one defined in `reduction` statement.

```c
double avg=0.0, A[MAX];
int i;
#pragma omp parallel for reduction (+:avg)
{
    // Each thread gets local copy of avg and does local summation.
    for(i=0;i<MAX;i++){
        avg += A[i];
    }
}
avg = avg / MAX;
```

#### Calculate Pi using `for reduction`

```c
    int i;
    double pi, sum = 0.0;
    step = 1.0 / (double)num_steps;
    #pragma omp parallel
    {
        double x;
        #pragma omp for reduction(+:sum)
        for (i = 0; i < num_steps; i++) {
            x = (i + 0.5) * step;
            sum = sum + 4.0 / (1.0 + x * x);
        }
    }
    pi = step * sum;
    printf("pi=%f", pi);
```

## Explicit vs Implied Barriers

```c
#pragma omp parallel shared (A, B, C) private(id)
{
    id = omp_get_thread_num();
    A[id] = big_calc(id);
#pragma omp barrier
#pragma omp for
    for(i=0;i<n;i++){
        C[i] = big_calc3(i,A);
    } // Implicit barrier is here- no threads go beyond end of this loop until all of the threads are finished
#pragma omp for nowait // Since barriers add overhead we can omit it
//nowait means: don't put barrier at the end of this loop 
    for(i=0;i<N;i++){ B[i] = big_calc2(C,i);}
    A[id] = big_calc4(id)
}// Implicit barrier can't be turned off
```

### `master`

There is no implied barrier on entry or exit.

```c
#pragma omp parallel
{
    do_many_things();
#pragma omp master    // Only master thread will run this block of code. No synchronization is implied.
{
    exchange_boundaries();
} // There is no implicit barrier so other threads will not wait for master to finish.
#pragma omp barrier   // We can specify barrier after omp master.
    do_many_other_things();
}
```

### `single`

```c
#pragma omp parallel
{
 do_many_things();
#pragma omp single    // Only one random thread does this block of code. Other threads wait for it to finish. Also `nowait` can be added after `single`.
{
 exchange_boundaries();
} // Implied barrier.
 do_many_other_things();
}
```

### `ordered`

```c
#define NUM_THREADS 2
int main()
{
    int i;
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for private(i) ordered
    for (i = 0; i < 6; i++) {
        printf("? Thread %d in random order  i=%d\n", omp_get_thread_num(),i);
        #pragma omp ordered
        printf(">>Ordered thread %d i=%d\n", omp_get_thread_num(), i);
    }
    return 0;
}
```

Output:

```c
? Thread 0 in random order  i=0
? Thread 1 in random order  i=3
>>Ordered thread 0 i=0
? Thread 0 in random order  i=1
>>Ordered thread 0 i=1
? Thread 0 in random order  i=2
>>Ordered thread 0 i=2
>>Ordered thread 1 i=3
? Thread 1 in random order  i=4
>>Ordered thread 1 i=4
? Thread 1 in random order  i=5
>>Ordered thread 1 i=5
```

### Section and sections

One thread does one section and other does other, but it doesn't matter which thread does what.

```c
#pragma omp parallel
{
    #pragma omp sections
    {
        #pragma omp section
            x_calculation();
        #pragma omp section
            y_calculation();
        #pragma omp section
            z_calculation();
    }
}
```

## Locks

Locks provide greater flexibility over critical sections and atomic updates (possible to implement asynchronous behaviour).
The so-called lock variable, is a special variable and should be manipulated through the API only.

- Simple locks: may not be locked if already in a locked state `omp_init_lock`.
- Nestable locks: may be locked multiple times by the same thread before being unlocked `omp_init_nest_lock`.

```c
omp_init_lock());
omp_set_lock();
omp_unset_lock();
omp_destroy_lock();
omp_test_lock(); 
```

Histogram calculation

```c
#pragma omp parallel for
for(i=0;i<NBUCKETS;i++){
    omp_init_lock(&hist_locks[i]);
    hist[i] = 0;    
}
#pragma omp parallel for
for(i=0;i<NVALS;i++){
    ival = (int) sample(arr[i]);
    omp_set_lock(&hist_locks[ival]);
    hist[ival]++;
    omp_unset_lock(&hist_locks[ival]);
}
for(i=0;i<NBUCKETS;i++)
    omp_destroy_lock(&hist_locks[i]);
```

## OpenMP Runtime Library Routines

The runtime functions take precedence over the corresponding environment variables.

```c
omp_set_num_threads(); // Requests certain number of threads
omp_get_num_threads(); // How many threads do I actually have
omp_get_thread_num();  // Get thread ID
omp_get_max_threads(); // System get give fewer threads than program asked for.
omp_in_parallel();     // Does this function run in parallel or not.
omp_set_dynamic();     // One region can have different number of threads than another mode.
omp_get_dynamic();     // Am I dynamic mode or not?
omp_num_procs();       // Get actual number of cores
```

### Get actual number of threads:

```c
#include <omp.h>
void main()
{
    int num_threads;
    omp_set_dynamic(0);    // Turn off dynamic mode
    omp_set_num_threads(omp_num_procs());    // Request number of threads (give me one thread per processor)
    // omp_get_num_threads(); // This is not in parallel region- it will give you one thread.
    #pragma omp parallel
    {
        int id = omp_get_threads_num();
        #pragma omp single
            num_threads = omp_get_num_threads();
        do_lots_of_stuff(id);
    }
}
```

## Environment Variables in OpenMP

```c
OMP_NUM_THREADS
OMP_STACKSIZE(int_literal)   // Set aside this stack size when you create threads because if you create big variables on the threads you will overflow stack.
OMP_WAIT_POLICY(ACTIVE|PASSIVE) // Active- spin lock (thread spins waiting for something to be available). Passive- put to sleep the thread that is waiting. It costs a lot to put thread to sleep and brig it back back.
OMP_PROC_BIND(TRUE|FALSE) // Once you bind the thread to processor leave it there, do not move them around (cache optimization)
```

## Data Environment

 OpenMP is shared memory programming model meaning that most variables are shared by default (most variables are sitting on the heap and all of the threads can see them). 

Global variables are shared among threads. In C `file scope variables` and `static variables`

Not everything is shared: if it is on the stack it is private to the thread. Stack variables in `functions(C)` called from parallel regions are **private**. Automatic variables within a statement block are private.

**Heap is shared, stack is private.**

Example:

```c
// If we have 3 threads
double A[10]              // Heap
int main(){
    int index[10];        // On heap because it is prior to the parallel region
#pragma omp parallel    // SHARED, PRIVATE, FIRSTPRIVATE, LASTPRIVATE
    // DEFAULT(PRIVATE): Not available in C.
    // DEFAULT(SHARED): Default. 
    // DEFAULT(NONE): I (the compiler) require that you specifically define the datatype of each variable. Useful for debugging.
    work(index);
    printf("%d\n", index[0])
}

extern double A[10]
void work(int *index){
    double temp[10];    // Stack on each the each thread (*3)
    static int count;   // Heap
}
```

### `private(var)`

```c
void wrong(){
    int tmp = 0;
#pragma omp parallel for private(tmp) //Each threads gets its own `tmp` variable.
{
    for (int j=0;j<1000;j++){
        tmp += j;    // There is a problem: private(tmp) creates the variable but doesn't give it initial value.
    }
}
    printf("%d\n",tmp);    // Prints global `tmp=0`
}
```

### `firstprivate(var)`

Creates the private copy but will initialize it with a global value.

```c
incr = 0;
#pragma omp parallel for firstprivate(incr)
{
    for(i=0;i<MAX;i++)
    {
        if( (i % 2) == 0)
        {
            incr++;    // has 0 as the starting value
        };
        A[i] = incr;  
    }

}
```

### `lastprivate(var)`

Whatever value of this variable the last iteration of the loop saw, copy that value to the global variable.

```c
void sq2(int n, double *lastterm)
{
    double x;
    int i;
    #pragma omp parallel for latprivate(x)
    for(i=0;i<n;i++){
        x = a[i]*a[i] + b[i]*b[i];
        b[i] = sqrt(x);
    }
    printf("x=%f",x); // Thread which did the n-1 iteration copies its value to global x
}
```

### Multiple clauses

 Only case when multiple clauses are allowed are for `firstprivate` and `lastprivate`.

```c
int A=1, B=1, C=1;
#pragma omp parallel private(B) firstprivate(C)
{
      // A is shared by all threads
      // B and C are local to each thread
      // B = undefined
      // C = 1
}
// A = 1 OR it has been changed in some thread
// B = 1
// C = 1
```

## Debugging OpenMP Programs

### Elegant solution to calculate Pi

```c
#include <stdio.h>
#include <omp.h>
static long num_steps = 100000000;
double step;
#define NUM_THREADS 2
int main()
{
    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)num_steps;
#pragma omp parallel for private(x) reduction(+:sum)
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    printf("pi=%f", pi);
    return 0;
}
```

## Skills Practice: Linked Lists and OpenMP

```c
p = head
while(p)
{
    process(p);
    p = p->next;
}
```

### Different Ways to Traverse Linked Lists

```c
while( p != NULL){
    p = p->next;
    count++;
}
// Create array big enough to hold it
p = head;
for(i=0;i<count;i++){
    parr[i] = p;
    p = p->next;
}
#pragma omp parallel
{
     #pragma omp for schedule(static,1)
     for(i=0;i<count;i++)
         process_work(parr[i]);
 }
```

## Tasks

### Linked Lists the Easy Way

Tasks are independent units of work. Tasks are composed of: code to execute, data environment, internal control variables (ICV). The runtime system decides when tasks are executed.

```c
#pragma omp parallel
{
    #pragma omp task    // Each thread creates following structured block as a task.
    foo();
    #pragma omp barrier
    #pragma omp single
    {
        #pragma omp task
        bar();
    } // Barrier
}
```

#### Fibonacci number

Private variables of the task are undefined outside the task. That is why we have to add `shared(var)`.

```c
// We have to add this in order to work
int fib(int n)
{
    int x, y;
    if ( n < 2) return n;
    #pragma omp task //shared(x)
        x = fib(n-1);
    #pragma omp task //shared(y)
        y = fib(n-2);
    #pragma omp taskwait
        return x + y;
}
```

Example: 

```c
// We have to add this in order to work
List ml;
Element *e;
#pragma omp parallel
#pragma omp single
{
    for(e=ml->first;e;e=e->next)
    #pragma omp task //firstprivate(e)
        process(e);

}
```

### Understanding Tasks

Linked Lists example:

```c
#pragma omp parallel
{
    #pragma omp single
    {
        node * p = head;    // Private barrier because it is in the scope of single
        while(p){
            #pragma omp task firstprivate(p)
                process(p);
            p = p->next;            
        }
    } 
}
```

![img](img/tasks.png)

## Memory Model, Atomics and Flush (Pairwise Synch)

 Flush is automatically ran:

- After entering or exiting the parallel region.

- After implicit or explicit barrier.

- At entry/exit of critical regions.

- Whenever a lock is set or unset.

Do not use flush with a list. 

## The Pitfalls of Pairwise Synchronization

Use a shared flag variable.

Reader spins waiting for the new flag value.

Use flushes to force updates to and from memory.

This code will work 99.9% of the time:

```c
// Has to have at least 2 threads
int main()
{
    double *A, sum, runtime;
    int num_threads, flag=0;
    A = (double *)malloc(N*sizeof(double));
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            fill_rand(N, A);
            #pragma omp flush 
            flag = 1;
            #pragma omp flush(flag)
        }
        #pragma omp section
        {
            #pragma omp flush(flag)
            while( flag == 0 ){
                #pragma omp flush(flag);
            }
            #pragma omp flush
            sum = sum_array(N, A);
        }
    }
}
```

This code will always work:

```c
// Has to have at least 2 threads
int main()
{
    double *A, sum, runtime;
    int num_threads, flag=0;
    A = (double *)malloc(N*sizeof(double));
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            fill_rand(N, A);
            #pragma omp flush
            #pragma atomic write 
                flag = 1;
            #pragma omp flush(flag)
        }
        #pragma omp section
        {
            while(1){
                #pragma omp flush(flag)
                #pragma omp atomic read
                    flg_tmp = flag;
                if(flg_tmp==1) break;
            }
            #pragma omp flush
            sum = sum_array(N, A);
        }
    }
}
```

## Threadprivate Data

Makes global data private to a thread.
`threadprivate` copies of the designated global variables and common blocks will be made. 
Initial data is undefined, unless `copyin` is used.
The number of threads has to remain the same for all the parallel regions (i.e. no dynamic threads).
Values saved in `threadprivate` variables can span across multiple parallel regions only if the number of theads is constant (`omp_set_dynamic(0)`).

```c
int counter = 0;
#pragma omp threadprivate(counter) // Initialize `counter` to the static value, 0, and it is going to have that value per thread.

int increment_counter()
{
    counter++;
    return(counter);
}
```

Code:

```c
#include <omp.h>
#include <stdio.h>
int th_a, p_b, i;
#pragma omp threadprivate(th_a)
main() {
    omp_set_num_threads(4);
    omp_set_dynamic(0);
#pragma omp parallel private(p_b)
    {
        int tid = omp_get_thread_num();
        th_a = tid;
        p_b = tid;
        printf("Thread %d: th_a=%d p_b=%d\n", tid, th_a, p_b);
    }

    printf("Master thread does serial work here.\np_b is private and that is why its value is lost.\n");
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        printf("Thread %d: th_a=%d p_b=%d\n", tid, th_a, p_b);
    }
}
```

Output:

```c
Thread 0: th_a=0 p_b=0
Thread 1: th_a=1 p_b=1
Thread 2: th_a=2 p_b=2
Thread 3: th_a=3 p_b=3
Master thread does serial work here.
p_b is private and that is why its value is lost.
Thread 0: th_a=0 p_b=0
Thread 1: th_a=1 p_b=0
Thread 2: th_a=2 p_b=0
Thread 3: th_a=3 p_b=0
```

### `copyin`

Code:

```c
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
int global_var = -1;
#pragma omp threadprivate(global_var)

void build()
{
    printf("thread=%d\tglobal_var=%d\n", omp_get_thread_num(), global_var);
}
void without_copyin(int x)
{
    global_var = x;
#pragma omp parallel
    {
        build();
    }
}

void with_copyin(int x)
{
    global_var = x;
#pragma omp parallel copyin(global_var)
    {
        build();
    }
}

int main() {
    omp_set_num_threads(4);
    omp_set_dynamic(0);
    printf("without_copyin(5):\n");
    without_copyin(5);
    printf(">> Non-parallel region: global_var=%d\n", global_var);
    printf("---\nwith_copyin(6):\n");
    with_copyin(6);
    printf(">> Non-parallel region: global_var=%d\n", global_var);
    printf("---\nAnother parallel region:\n");
#pragma omp parallel
    {
        printf("Thread %d: global_var=%d\n", omp_get_thread_num(), global_var);
    }
}
```

Output:

```c
without_copyin(5):
thread=0        global_var=5
thread=1        global_var=-1
thread=2        global_var=-1
thread=3        global_var=-1
>> Non-parallel region: global_var=5
---
with_copyin(6):
thread=2        global_var=6
thread=3        global_var=6
thread=0        global_var=6
thread=1        global_var=6
>> Non-parallel region: global_var=6
---
Another parallel region:
Thread 0: global_var=6
Thread 1: global_var=6
Thread 2: global_var=6
Thread 3: global_var=6
```

Example:

```c
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
int j;
#pragma omp threadprivate(j)

int main() {
    omp_set_num_threads(4);
    omp_set_dynamic(0);
    j = 1;
#pragma omp parallel copyin(j)
    {
#pragma omp master //single
        {
            j = 2;
        }
    }
    printf("j= %d\n", j);
}
```

Output:

```c
j= 2 // for master j=2; for single j=1; 
```
