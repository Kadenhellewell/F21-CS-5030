#include <iostream>

//Parallelize using OpenMP

void count_sort( int a[], int n, int thread_count) {
    int i, j, count;
    int* temp;
    
    for (i = 0; i < n; i++) {
        count = 0;
        for (j = 0; j < n; j++)
            if (a[j] < a[i])
                count++;
            else if (a[j] == a[i] && j < i)
                count++;
        temp[count] = a[i];
    }
    memcpy(a, temp, n*sizeof(int));
    free(temp);
} /* count_sort */

//1. If we try to parallelize the for i loop (the outer loop), which variables should be private
//   and which should be shared? (5 points)
//Answer:
//2. Can we parallelize the call to memcpy? If so, how can we do this? If not, can we modify the code
//   so that this operation will be parallelizable? (5 points)
//Answer:
//3. Write a C/C++ OpenMP program that includes a parallel implementation of count_sort. (25 points)


//Inputs:
//1. Number of threads
//2. Number of elements to sort
int main(int argc, char* argv[])
{
    //Create array of n elements (integers, srand(100))
    //Pass thread_count into count_sort
    //Print array pre and post sorted
    std::cout << "The project can start" << std::endl;
    return 0;
}

