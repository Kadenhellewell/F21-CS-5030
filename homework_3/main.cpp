#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

//Parallelize using OpenMP

void count_sort( int a[], int n, int thread_count) {
    int i, j, count;
    int* temp = new int[n];

#   pragma omp parallel for num_threads(thread_count) private(count, i, j) shared(a, n, temp)
    for (i = 0; i < n; i++) {
        count = 0;
        for (j = 0; j < n; j++)
            if (a[j] < a[i])
                count++;
            else if (a[j] == a[i] && j < i)
                count++;
        //I didn't use reduction here as assignment is not a binary operator
#       pragma omp ciritcal
        temp[count] = a[i];
    }
    //put brand new parellization here?
    memcpy(a, temp, n*sizeof(int));
    delete[] temp;
} /* count_sort */

//1. If we try to parallelize the for i loop (the outer loop), which variables should be private
//   and which should be shared? (5 points)
//Answer: count, temp, and i should be private while a and n should be shared

//2. Can we parallelize the call to memcpy? If so, how can we do this? If not, can we modify the code
//   so that this operation will be parallelizable? (5 points)
//Answer: Not as the code is, but we can with changes to the code. We could have each thread only copy
//        the values it has modified to a. Since memcpy doesn't touch anything around it, it should be
//        safe. However, I don't know how to do that in C++.
//3. Write a C/C++ OpenMP program that includes a parallel implementation of count_sort. (25 points)


//Inputs:
//1. Number of threads
//2. Number of elements to sort
int main(int argc, char* argv[])
{
    //Create array of n elements (integers, srand(100))
    //Pass thread_count into count_sort
    //Print array pre and post sorted
    std::string t_count_string(argv[1]);
    int thread_count = std::stoi(t_count_string);

    std::string el_count_string(argv[2]);
    int num_elements = std::stoi(el_count_string);

    int* array = new int[num_elements];
    srand(100);
    for(int i = 0; i < num_elements; i++)
    {
        array[i] = (rand() % num_elements) + 1;
    }

    std::cout << "Pre-sorted array:" << std::endl;
    for(int i = 0; i < num_elements; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl << std::endl;

    count_sort(array, num_elements, thread_count);

    std::cout << "Sorted array:" << std::endl;
    for(int i = 0; i < num_elements; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

