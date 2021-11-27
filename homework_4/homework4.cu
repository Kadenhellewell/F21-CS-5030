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
#include <cuda.h>


using namespace std;

/***
 * Transposes a (square) rgb image using global memory
 * @param input_image the input array
 * @param output_image the output array
 * @param width width of the image, which is a square
 */
__global__
void cuda_transpose_global(float* input_image, float* transposed_image, int width){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < width*width) // since we're using a square image, there are WIDTH^2 elements.
    {
        int row = index / width; // I expect truncation
        int col = index % width;
        int b_index = col*width + row; //this is the transpose calculation
        transposed_image[index] = input_image[b_index];
        transposed_image[++index] = input_image[++b_index];
        transposed_image[++index] = input_image[++b_index];
    }
}



/***
 * Transposes a (square) rgb image using shared memory
 * @param A_d the input array
 * @param B_d the output array
 * @param width size of the arrays
 */
__global__
void cuda_transpose_shared(float* input_image, float* output_image, int width){
    //I don't know how to actually know this.
    __shared__ char inputTile[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //I'm confused about the difference in accessing the tile's info and the inputs info.
    int inputCol = bx * blockDim.x + tx;
    int inputRow = by * blockDim.y + ty;

    int col = bx * TILE_WIDTH + tx;
    int row = by * TILE_WIDTH + ty;

    if(col < width && row < width)
    {
        //I'm pretty this is wrong as it doesn't appear that the tile is doing anything meaningful
        inputTile[ty][tx] = input_image[inputRow*WIDTH + inputCol];
        output_image[tx][ty] = inputTile[ty][tx];
    }
}

/***
 * Transposes a (square) rgp image using the CPU (for validation)
 * @param A_d the input array
 * @param B_d the output array
 * @param width width of the matrix (square)
 */
 __host__
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
 * @param fileName the name of the file to write to
 */
void write_image(char* B, int n, string fileName)
{
    ofstream outFile;
    outFile.open(fileName, ios_base::binary);
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
    //Read in the file
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

    //Perform serial transpose
    char *transposed = new char[buffer.size()];
    //Validation function
    serial_transpose(buffer, 1024, transposed);

    //Allocate memory for kernel functions
    char* image_d;
    cudaMalloc(&image_d, buffer.size());
    //buffer.data() is the actual pointer
    cudaMemcpy(image_device, buffer.data(), buffer.size(), cudaMemcpyHostToDevice);

    char* forGlobal_d;
    cudaMalloc(forGlobal_d, buffer.size());

    char* forShared_d;
    cudaMalloc(&forShared_d, buffer.size());

    //Set up blocks
    dim3 DimGrid(ceil(n/256), 1, 1);
    dim3 DimBlock(256, 1, 1);
    //call kernel functions
    cuda_transpose_global<<<DimGrid, DimBlock>>>(image_d, forGlobal_d, 1024);
    cuda_transpose_shared<<<DimGrid, DimBlock>>>(image_d, forShared_d, 1024);

    //Copy results from kernel function back to host
    char* globalResult;
    cudaMemcpy(globalResult, forGlobal_d, buffer.size(), cudaMemcpyDeviceToHost);

    char* sharedResult;
    cudaMemcpy(sharedResult, forShared_d, buffer.size(), cudaMemcpyDeviceToHost);

    //verify that the kernel functions worked
    bool didGlobalRight = true;
    bool didSharedRight = true;
    for(int i = 0; i < buffer.size(); i++)
    {
        if(globalResult[i] != transposed[i])
            didGlobalRight = false;
        if(sharedResult[i] != transposed[i])
            didSharedRight = false;
    }

    //condition ? expression1 : expression2;
    didGlobalRight ? cout << "Did global right" << endl : cout << "Did global wrong" << endl;
    didSharedRight ? cout << "Did shared right" << endl : cout << "Did shared wrong" << endl;

    //write results to files
    write_image(transposed, buffer.size(), "serial.raw");
    write_image(globalResult, buffer.size(), "global.raw");
    write_image(sharedResult, buffer.size(), "shared.raw");

    return 0;
}


