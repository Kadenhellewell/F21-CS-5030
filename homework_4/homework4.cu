/*
 * Hello world cuda
 *
 * compile: nvcc hello_cuda.cu -o hello
 *
 *  */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>


/***
 * Transposes a (square) rgp image using global memory
 * @param A_d the input array
 * @param B_d the output array
 * @param width size of the arrays
 */
__global__
void cuda_transpose_global(float* A_d, float* B_d, int width){
    // thread id of current block (on x axis)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    //TODO check that row and col are within the array
    B_d[col][row] = A_d[row][col];
}

/***
 * Transposes a (square) rgp image using shared memory
 * @param A_d the input array
 * @param B_d the output array
 * @param width size of the arrays
 */
__global__
void cuda_transpose_shared(float* A_d, float* B_d, int width){
    // thread id of current block (on x axis)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ subTileA;
    __shared__ subTileB;
    //TODO check that row and col are within the array
    B_d[col][row] = A_d[row][col];
}



using namespace std;

/***
 * Transposes a (square) rgp image using the CPU (for validation)
 * @param A_d the input array
 * @param B_d the output array
 * @param width width of the matrix (square)
 */
void serial_transpose(vector<char> const& A, int width, char* B)
{
    vector<int> testing;
    for(int i = 0; i < A.size(); i++)
    {
        int row = i / width; // I expect truncation
        int col = i % width;
        // i = row*width + col
        int b_index = col*width + row;
        B[i] = A[b_index];
        B[++i] = A[++b_index];
        B[++i] = A[++b_index];
    }
}

/***
 * Write image to file
 * @param B the array containing the image
 * @param n the size of the array
 */
void write_image(char* B, int n)
{

    ofstream outFile;
    outFile.open("testing.raw", ios_base::binary);
    if(outFile.is_open())
    {
        for(int i = 0; i < n; i++)
        {
            outFile.put(B[i]);
        }
    }
}



int main()
{
    ifstream inFile;

    inFile.open("gc_1024x1024.raw", ios_base::binary);
    char myChar;
    vector<char> buffer;
    buffer[0];

    if(inFile.is_open())
    {
        cout << "We opened the file" << endl;
        while (inFile >> noskipws >> myChar)
        {
            buffer.push_back(myChar);
        }
        inFile.close();
    }
    else
    {
        cout << "It broke" << endl;
    }

    char *transposed = new char[buffer.size()];
    //Validation function
    serial_transpose(buffer, 1024, transposed);

    write_image(transposed, buffer.size());

    //Does the input get copied to constant memory? I think it does.
    //Preface kernel functions with "__global__". Do I need to preface non-kernel functions?
    //I cannot find where cuda_memcpy and cuda_malloc are used.
    //Use Kernel function that uses global memory
    /***
     * Things to figure out:
     * 1. copy array to GPU global memory
     * 2. Once there, access that memory in the kernel function
     */
    //Use Kernel function that uses shared memory and tiling
    /***
     * Things to figure out:
     * 1. Set up tiling (grid size, block size)
     * 2. copy array to GPU global memory
     * 3. Copy some data (figure out which) into shared memory (from the kernel function)
     * 4. Use shared memory
     */


    return 0;
}

//references:
// https://www.dreamincode.net/forums/topic/170054-understanding-and-reading-binary-files-in-c/

