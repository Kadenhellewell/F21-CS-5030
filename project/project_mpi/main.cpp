#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
//#include <mpi.h>

using namespace std;

struct Point_Vector{
    float x_val;
    float y_val;
};

vector<Point_Vector> vectors;

void get_args(int argc, char* argv[]);
Point_Vector const_vect_mult(float c, Point_Vector v);
Point_Vector get_v_from_field(int x_coord, int y_coord);
Point_Vector get_v_from_field(float x_coord, float y_coord);
Point_Vector get_v_from_field(Point_Vector p);
Point_Vector add_vectors(Point_Vector v1, Point_Vector v2);

/**
 * Given a position of a particle, find its next position.
 * @param x0 Current particle x position
 * @param y0 Current particle y position
 * @param x New x position
 * @param x_step time step
 * @return New y position
 */
Point_Vector rungeKutta(Point_Vector pi, float x_step)
{
    Point_Vector k1{}, k2{}, k3{}, k4{};

    // Apply Runge Kutta Formulas
    // to find next value of y
    k1 = const_vect_mult(x_step, get_v_from_field(pi));
    Point_Vector v_1 = get_v_from_field(add_vectors(pi, const_vect_mult(.5, k1)));
    k2 = const_vect_mult(x_step, v_1);
    Point_Vector v_2 = get_v_from_field(add_vectors(pi, const_vect_mult(.5, k2)));
    k3 = const_vect_mult(x_step, v_2);
    Point_Vector v_3 = get_v_from_field(add_vectors(pi, k3));
    k4 = const_vect_mult(x_step, v_3);
    Point_Vector tempSum = k1;
    tempSum = add_vectors(tempSum, const_vect_mult(2, k2));
    tempSum = add_vectors(tempSum, const_vect_mult(2, k3));
    tempSum = add_vectors(tempSum, k4);

    return add_vectors(pi, const_vect_mult(0.1667, tempSum));
}
//Algorithm from: https://web.cs.ucdavis.edu/~ma/ECS177/papers/particle_tracing.pdf

int comm_sz; //Number of Process
int my_rank; //Rank of current process
//MPI_Comm comm;

int data_cols = 1300;
int data_rows = 600;
//for a total of 780,000 vectors

// description here of what order things are passed in
// 1. bin count
// 2. min
// 3. max
// 4. data count
int main(int argc, char* argv[]) {

//    MPI_Init(&argc, &argv);
//    comm = MPI_COMM_WORLD;
//    MPI_Comm_size(comm, &comm_sz);
//    MPI_Comm_rank(comm, &my_rank);

    //Read in file
    vector<float> buffer;
    float f;
    std::ifstream inFile("cyl2d_1300x600_float32[2].raw", std::ios::binary);
    //buffer contains all of the floats
    while (inFile.read(reinterpret_cast<char*>(&f), sizeof(float)))
        buffer.push_back(f);

    //Set up vector of vectors
    for(int i = 0; i < buffer.size(); i++)
    {
        Point_Vector thisVector{};
        thisVector.x_val = buffer[i];
        thisVector.y_val = buffer[++i];
        vectors.push_back(thisVector);
    }

    int initial_y = 300;
    float time_step = .1;
    int initial_x = 0;
    cout << "line_id, coordinate_x, coordinate_y" << endl;
    for(int row = 0; row < 10; row++)
    {
        int lineId = row;
        cout << lineId << ", ";
        for(int col = 0; col < 10; col++)
        {
            Point_Vector current = get_v_from_field(col, row);
//            current = rungeKutta(current, time_step);
            cout << current.x_val << ", " << current.y_val << endl;
        }
    }


    //MPI_Finalize();
    return 0;
}

Point_Vector get_v_from_field(int x_coord, int y_coord)
{
    int index = y_coord*data_cols + x_coord;
    return vectors[index];
}

Point_Vector get_v_from_field(float x_coord, float y_coord)
{
    //Get element index from coordinates
    int floor_y = floor(y_coord);
    int floor_x = floor(x_coord);
    int ceil_y = ceil(y_coord);
    int ceil_x = ceil(x_coord);

    //Q11 - bottom left; Q12 - top left; Q21 - bottom right; Q22 - top right
    //These are velocity vectors, not points
    Point_Vector Q11 = get_v_from_field(floor_x, floor_y);
    Point_Vector Q12 = get_v_from_field(floor_x, ceil_y);
    Point_Vector Q21 = get_v_from_field(ceil_x, floor_y);
    Point_Vector Q22 = get_v_from_field(ceil_x, ceil_y);

    //bilinear interpolation
    //Calculate R1
    Point_Vector temp1 = const_vect_mult((ceil_x - x_coord)/(ceil_x - floor_x), Q11);
    Point_Vector temp2 = const_vect_mult((x_coord - floor_x)/(ceil_x - floor_x), Q21);
    Point_Vector R1 = add_vectors(temp1, temp2);

    //Calculate R2
    temp1 = const_vect_mult((ceil_x - x_coord)/(ceil_x - floor_x), Q12);
    temp2 = const_vect_mult((x_coord - floor_x)/(ceil_x - floor_x), Q22);
    Point_Vector R2 = add_vectors(temp1, temp2);

    //Calculate P
    temp1 = const_vect_mult((ceil_y - y_coord)/(ceil_y - floor_y), R1);
    temp2 = const_vect_mult((y_coord - floor_y)/(ceil_y - floor_y), Q21);
    Point_Vector P = add_vectors(temp1, temp2);
    return P;
}

Point_Vector get_v_from_field(Point_Vector p)
{
    return get_v_from_field(p.x_val, p.y_val);
}

Point_Vector const_vect_mult(float c, Point_Vector v)
{
    Point_Vector returnVector{};
    returnVector.x_val = c*v.x_val;
    returnVector.y_val = c*v.y_val;
    return returnVector;
}

Point_Vector add_vectors(Point_Vector v1, Point_Vector v2)
{
    Point_Vector returnVector{};
    returnVector.x_val = v1.x_val + v2.x_val;
    returnVector.y_val = v1.y_val + v2.y_val;
    return returnVector;
}

/**
 * Get input from the user, store, and broadcast
 * @param argc number of arguments
 * @param argv array containing the arguments
 */
void get_args(int argc, char* argv[])
{

}



