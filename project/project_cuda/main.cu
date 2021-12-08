/**
 * compile: nvcc main.cu -o cuda_streams
 */


#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;




//These are separate structs to aid is differentiating points and vectors
struct Vector{
    float x_val;
    float y_val;
};

struct Point{
    float x_coord;
    float y_coord;
};

__device__
Vector const_vect_mult(float c, Vector v);
__device__
Vector get_v_from_field(int x_coord, int y_coord);
__device__
Vector get_v_from_field(float x_coord, float y_coord);
__device__
Vector get_v_from_field(Point p);
__device__
Vector add_vectors(Vector v1, Vector v2);
__device__
Point add_vector_point(Point p, Vector v);
__device__
Point rungeKutta(Point p, float time_step, Vector* vectors);
__device__
bool not_in_range(Point p);
__device__
Vector interpolate(Vector v1, Vector v2, int bigP, int smallP, float p);

const int data_cols = 1300;
const int data_rows = 600;
const int num_steps = 50;
const int stream_size = num_steps*data_rows*3;
const int num_vectors = data_rows * data_cols;
const int data_size = num_vectors*2;//2 floats per vector
//TODO: define tile size variables (whatever they are)

/**
 * Starting at the far left, calculate stream lines going to the right.
 * Each thread calculates one streamline. The row is determined by its id.
 * @param vectors the vector field
 * @param streams the output array - floats, a multiple of 3, in the order line_id, coordinate_x, coordinate_y
 */
__global__
void calculate_stream_lines(Vector* vectors, float* streams_d)//streams is the output
{
    //Each thread calculates one stream
    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;//TODO: make sure this is correct for 1D (this is the threadId)

    float time_step = .2;
    int lineId = thread_id;
    if(lineId >= data_rows) return; //passed the bottom row
    //initialize the starting point at the beginning of each new line
    Point current{};
    current.x_coord = 0;//Each streamline starts at the far left
    current.y_coord = lineId;
    int startPoint = 0;
    if(lineId > 0) //0 is an edge case in this calculation
        startPoint = lineId*num_steps*3 + 1;
    for(int step = 0; step < num_steps*3; step++)//each 'step' fills out 3 elements of the array
    {
        if(not_in_range(current)) break;//The streamline has left the known vector field. This thread is done
        streams_d[startPoint + step] = lineId;
        streams_d[startPoint + ++step] = current.x_coord;
        streams_d[startPoint + ++step] = current.y_coord;
        current = rungeKutta(current, time_step, vectors);
    }
}


int main() {
    Vector* vectors;
    std::ifstream inFile("cyl2d_1300x600_float32[2].raw", std::ios::binary);

    //Read in from file
    float f;
    int k = 0;
    float * data = new float[data_size];
    while (inFile.read(reinterpret_cast<char*>(&f), sizeof(float)))
    {
        data[k] = f;
        k++;
    }
    //set vector objects
    for(int i = 0; i < data_size; i++)
    {
        int index = i/2;//i will always be even at this point
        Vector thisVector{};
        thisVector.x_val = data[i];
        thisVector.y_val = data[++i];
        vectors[index] = thisVector;
    }

    //Allocate spaced on the GPU for vectors, then copy up
    Vector* vectors_d;
    cudaMalloc(&vectors_d, num_vectors);
    cudaMemcpy(vectors_d, vectors, num_vectors, cudaMemcpyHostToDevice);

    //initialize streams
    float* streams = new float[stream_size];
    for(int i = 0; i < stream_size; i++)
    {
        streams[i] = -1;
    }

    //allocate space for device results (GPU)
    float* streams_d;
    cudaMalloc(&streams_d, stream_size);
    cudaMemcpy(streams_d, streams, stream_size, cudaMemcpyHostToDevice);

    //I think I want 1D blocks, since each streamline starts in the first column. Not sure how to do that.
    //Set grid and block sizes
    dim3 DimGrid(1, 1, 1);// how many blocks
    dim3 DimBlock(600, 1, 1);// how many threads per block

    calculate_stream_lines<<<DimGrid, DimBlock>>>(vectors_d, streams_d);
    //copy results of calculating streams to host
    cudaMemcpy(streams, streams_d, stream_size, cudaMemcpyDeviceToHost);

    std::ofstream outFile("streamlines_cuda.csv", std::ios::app);
    //Parse results, groups of 3, line_id, coordinate_x, coordinate_y
    outFile << "line_id, coordinate_x, coordinate_y" << endl;
    //print local streams to file
    for(int j = 0; j < stream_size; j++)
    {
        if(streams[j] != -1)
            outFile << streams[j] << ", " << streams[++j] << ", " << streams[++j] << endl;
    }
    cudaFree(vectors);
    cudaFree(streams_d);
    delete[] streams;
    return 0;
}

/**
 * Get the value of the given vector field at the given point.
 * For integers, this can be directly retrieved from the vector field (no interpolation needed).
 * @param x_coord the x coordinate
 * @param y_coord the y coordinate
 * @param vectors the vector field
 * @return a Vector object
 */
__device__
Vector get_v_from_field(int x_coord, int y_coord, Vector* vectors)
{
    int index = y_coord*data_cols + x_coord;
    return vectors[index];
}

/**
 * Get the value of the given vector field at the given point.
 * This method uses Bilinear Interpolation.
 * @param x_coord the x coordinate
 * @param y_coord the y coordinate
 * @param vectors the vector field
 * @return a Vector object
 */
__device__
Vector get_v_from_field(float x_coord, float y_coord, Vector* vectors)
{
    //Bilinear Interpolation
    //Get integer points around the given x and y
    int floor_y = floor(y_coord);
    int floor_x = floor(x_coord);
    int ceil_y = ceil(y_coord);
    int ceil_x = ceil(x_coord);

    if(ceil_x == floor_x && ceil_y == floor_y)//both are integers, no interpolation
    {
        return get_v_from_field((int)x_coord, (int)y_coord, vectors);
    }

    Vector R1{}, R2{};
    //Linear interpolation
    if(ceil_x == floor_x) //x is an integer, y is not
    {
        R1 = get_v_from_field((int)x_coord, floor_y);
        R2 = get_v_from_field((int)x_coord, ceil_y);
        return interpolate(R1, R2, ceil_y, floor_y, y_coord);
    }

    if(ceil_y == floor_y)//y is an integer, x is not
    {
        R1 = get_v_from_field(floor_x, (int)y_coord);
        R2 = get_v_from_field(ceil_x, (int)y_coord);
        return interpolate(R1, R2, ceil_x, floor_x, x_coord);
    }

    //Neither are integers
    //bilinear interpolation
    //Q11 - bottom left; Q12 - top left; Q21 - bottom right; Q22 - top right
    Vector Q11 = get_v_from_field(floor_x, floor_y, vectors);
    Vector Q12 = get_v_from_field(floor_x, ceil_y, vectors);
    Vector Q21 = get_v_from_field(ceil_x, floor_y, vectors);
    Vector Q22 = get_v_from_field(ceil_x, ceil_y, vectors);

    //Calculate R10
    R1 = interpolate(Q11, Q21, ceil_x, floor_x, x_coord);

    //Calculate R2
    R2 = interpolate(Q12, Q22, ceil_x, floor_x, x_coord);

    //Calculate P
    return interpolate(R1, R2, ceil_y, floor_y, y_coord);
}
/*
 * How to calculate R1, R2, and P (from https://x-engineer.org/bilinear-interpolation/)
 * \[R_{1}(x, y) = Q_{11} \frac{x_{2}-x}{x_{2}-x_{1}} + Q_{21} \frac{x-x_{1}}{x_{2}-x_{1}} \tag{1}\]

   \[R_{2}(x, y) = Q_{12} \frac{x_{2}-x}{x_{2}-x_{1}} + Q_{22} \frac{x-x_{1}}{x_{2}-x_{1}} \tag{2}\]

   \[{P(x,y) = R_{1} \frac{y_{2}-y}{y_{2}-y_{1}} + R_{2} \frac{y-y_{1}}{y_{2}-y_{1}}} \tag{3}\]
 */

/**
 * This method performs linear interpolation.
 * @param v1 the vector corresponding to the larger point
 * @param v2 the vector corresponding to the smaller point
 * @param bigP the larger point
 * @param smallP the smaller point
 * @param p the desired point
 * @return the vector at the desired point
 */
__device__
Vector interpolate(Vector v1, Vector v2, int bigP, int smallP, float p)
{
    Vector temp1 = const_vect_mult((bigP - p) / (bigP - smallP), v1);
    Vector temp2 = const_vect_mult((p - smallP) / (bigP - smallP), v2);
    Vector returnVector = add_vectors(temp1, temp2);
    return returnVector;
}

/**
 * Wrapper function. Gets the vector associated with a point.
 * @param p the point
 * @param vectors the vector field
 * @return the desired vector
 */
__device__
Vector get_v_from_field(Point p, Vector* vectors)
{
    return get_v_from_field(p.x_coord, p.y_coord, vectors);
}

/**
 * Multiply a vector by a constant.
 * @param c the constant
 * @param v the vector
 * @return the new vector
 */
__device__
Vector const_vect_mult(float c, Vector v)
{
    Vector returnVector{};
    returnVector.x_val = c*v.x_val;
    returnVector.y_val = c*v.y_val;
    return returnVector;
}

/**
 * Add 2 vectors together
 * @param v1 the first vector
 * @param v2 the second vector
 * @return the sum of the two vectors
 */
__device__
Vector add_vectors(Vector v1, Vector v2)
{
    Vector returnVector{};
    returnVector.x_val = v1.x_val + v2.x_val;
    returnVector.y_val = v1.y_val + v2.y_val;
    return returnVector;
}

/**
 * Add a vector to a point to get a new point
 * @param p the starting point
 * @param v the vector
 * @return the new point
 */
__device__
Point add_vector_point(Point p, Vector v)
{
    Point returnPoint{};
    returnPoint.x_coord = p.x_coord + v.x_val;
    returnPoint.y_coord = p.y_coord + v.y_val;
    return returnPoint;
}

/**
 * Do the Runge-Kutta algorithm
 * @param p the starting point
 * @param time_step the time step
 * @param vectors the vector field
 * @return the next point
 */
__device__
Point rungeKutta(Point p, float time_step, Vector* vectors)
{
    Vector k1{}, k2{}, k3{}, k4{};
    Point failPoint{};
    failPoint.x_coord = -1;
    failPoint.y_coord = -1;

    // Apply Runge Kutta Formulas
    // to find next value of y
    k1 = const_vect_mult(time_step, get_v_from_field(p));
    Point p1 = add_vector_point(p, const_vect_mult(.5, k1));
    if(not_in_range(p1)) return failPoint;
    Vector v_1 = get_v_from_field(p1, vectors);

    k2 = const_vect_mult(time_step, v_1);
    Point p2 = add_vector_point(p, const_vect_mult(.5, k2));
    if(not_in_range(p2)) return failPoint;
    Vector v_2 = get_v_from_field(p2, vectors);

    k3 = const_vect_mult(time_step, v_2);
    Point p3 = add_vector_point(p, k3);
    if(not_in_range(p3)) return failPoint;
    Vector v_3 = get_v_from_field(p3, vectors);

    k4 = const_vect_mult(time_step, v_3);
    Vector tempSum = k1;
    tempSum = add_vectors(tempSum, const_vect_mult(2, k2));
    tempSum = add_vectors(tempSum, const_vect_mult(2, k3));
    tempSum = add_vectors(tempSum, k4);

    Vector temp = const_vect_mult(0.1667, tempSum);
    Point nextPoint = add_vector_point(p, temp);
    return nextPoint;
}
//Algorithm from: https://web.cs.ucdavis.edu/~ma/ECS177/papers/particle_tracing.pdf

/**
 * Check if a point is within the given vector field
 * @param p the point
 * @return whether the point is not in the vector field
 */
__device__
bool not_in_range(Point p)
{
    return p.x_coord < 0 || p.x_coord >= data_cols || p.y_coord < 0 || p.y_coord >= data_rows;
}

/**
 * Get input from the user, store, and broadcast
 * @param argc number of arguments
 * @param argv array containing the arguments
 */
void get_args(int argc, char* argv[])
{

}




