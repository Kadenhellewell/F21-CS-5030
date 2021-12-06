#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <mpi.h>

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

void get_args(int argc, char* argv[]);

Vector const_vect_mult(float c, Vector v);
Vector get_v_from_field(int x_coord, int y_coord);
Vector get_v_from_field(float x_coord, float y_coord);
Vector get_v_from_field(Point p);
Vector add_vectors(Vector v1, Vector v2);
Point add_vector_point(Point p, Vector v);
Point rungeKutta(Point p, float time_step);
bool not_in_range(Point p);
Vector interpolate(Vector v1, Vector v2, int bigP, int smallP, float p);

MPI_Comm comm;
int comm_sz; //Number of Process
int my_rank; //Rank of current process

int num_steps = 50;
int data_cols = 1300;
int data_rows = 600;
//for a total of 780,000 vectors
Vector* vectors;

// description here of what order things are passed in
// 1. bin count
// 2. min
// 3. max
// 4. data count
int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);
    int lines_per_proc = data_rows / (comm_sz - 1);//the -1 is for the fact that 0 doesn't do this
    int data_size = data_rows*data_cols*2;
    vectors = new Vector[data_size];
    if(my_rank == 0)
    {
        std::ifstream inFile("cyl2d_1300x600_float32[2].raw", std::ios::binary);
        float * buffer;
        if (inFile) {
            // get length of file:
            inFile.seekg (0, inFile.end);
            int length = inFile.tellg();
            inFile.seekg (0, inFile.beg);
            buffer = new float[length / sizeof(float)];

            std::cout << "Reading " << length << " characters... " << endl;
            // read data as a block:
            inFile.read ((char*)buffer, length);
            inFile.close();

            //Set up vector of vectors
            for(unsigned int i = 0; i < length / sizeof(float); i++)
            {
                Vector thisVector{};
                thisVector.x_val = buffer[i];
                thisVector.y_val = buffer[++i];
                vectors[i] = thisVector;
            }
        }
    }
    //MPI_Bcast(bin_maxes, bin_count, MPI_FLOAT, 0, comm);
    MPI_Barrier(comm);
    MPI_Bcast(vectors, data_size, MPI_FLOAT, 0, comm);
    if(my_rank != 0)
    {
        int my_first = lines_per_proc*(my_rank - 1); //the -1 is for the fact that 0 doesn't do this
        int my_last = my_first + lines_per_proc; //inclusive
        float initial_x = 0;//all streamlines start at x=0
        float initial_y = my_first;
        float time_step = .2;
        //create string array; store output lines there
        string* my_lines = new string[num_steps];
        Point current{};
        current.x_coord = initial_x;
        current.y_coord = initial_y;
        int total_chars = 0;
        for(int lineId = my_first; lineId <= my_last; lineId++)
        {
            if(lineId >= data_cols) break; //passed the bottom row
            current.x_coord = 0;
            current.y_coord = lineId;
            for(int step = 0; step < num_steps; step++)
            {
                if(not_in_range(current)) break;//The streamline has left the known vector field. Go to the next line.
		string string1, string2, string3;
                string new_thing = itoa(lineId, string1, 10) + ", " + ftoa(current.x_coord, string2, 10) + ", " + ftoa(current.y_coord, string3, 10) + "\n";
                my_lines[step] = new_thing;
                total_chars += new_thing.length();
                current = rungeKutta(current, time_step);
            }
        }
        //send size before array
        MPI_Send(&total_chars, 1, MPI_INT, 0, 0, comm);
        MPI_Send(my_lines, total_chars, MPI_CHAR, 0, 0, comm);
    }

    if(my_rank == 0)
    {
        std::ofstream outFile("streamlines.csv", std::ios::app);
        for(int i = 1; i < comm_sz; i++)
        {
            int total_chars;
            string *incoming_lines = new string[num_steps];
            //Receive local streamlines from each process
            MPI_Recv(&total_chars, 1, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
            MPI_Recv(incoming_lines, total_chars, MPI_CHAR, i, 0, comm, MPI_STATUS_IGNORE);

            //print local streams to file
            for(int j = 0; j < num_steps; j++)
                outFile << incoming_lines[j];
            delete[] incoming_lines;
        }
    }

    MPI_Finalize();
    return 0;
}

Vector get_v_from_field(int x_coord, int y_coord)
{
    int index = y_coord*data_cols + x_coord;
    return vectors[index];
}

Vector get_v_from_field(float x_coord, float y_coord)
{
    //Bilinear Interpolation
    //Get integer points around the given x and y
    int floor_y = floor(y_coord);
    int floor_x = floor(x_coord);
    int ceil_y = ceil(y_coord);
    int ceil_x = ceil(x_coord);

    if(ceil_x == floor_x && ceil_y == floor_y)//both are integers, no interpolation
    {
        return get_v_from_field((int)x_coord, (int)y_coord);
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
    Vector Q11 = get_v_from_field(floor_x, floor_y);
    Vector Q12 = get_v_from_field(floor_x, ceil_y);
    Vector Q21 = get_v_from_field(ceil_x, floor_y);
    Vector Q22 = get_v_from_field(ceil_x, ceil_y);

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

Vector interpolate(Vector v1, Vector v2, int bigP, int smallP, float p)
{
    Vector temp1 = const_vect_mult((bigP - p) / (bigP - smallP), v1);
    Vector temp2 = const_vect_mult((p - smallP) / (bigP - smallP), v2);
    Vector returnVector = add_vectors(temp1, temp2);
    return returnVector;
}

Vector get_v_from_field(Point p)
{
    return get_v_from_field(p.x_coord, p.y_coord);
}

Vector const_vect_mult(float c, Vector v)
{
    Vector returnVector{};
    returnVector.x_val = c*v.x_val;
    returnVector.y_val = c*v.y_val;
    return returnVector;
}

Vector add_vectors(Vector v1, Vector v2)
{
    Vector returnVector{};
    returnVector.x_val = v1.x_val + v2.x_val;
    returnVector.y_val = v1.y_val + v2.y_val;
    return returnVector;
}

Point add_vector_point(Point p, Vector v)
{
    Point returnPoint{};
    returnPoint.x_coord = p.x_coord + v.x_val;
    returnPoint.y_coord = p.y_coord + v.y_val;
    return returnPoint;
}

Point rungeKutta(Point p, float time_step)
{
    Vector k1{}, k2{}, k3{}, k4{};

    // Apply Runge Kutta Formulas
    // to find next value of y
    k1 = const_vect_mult(time_step, get_v_from_field(p));
    Point p1 = add_vector_point(p, const_vect_mult(.5, k1));
    Vector v_1 = get_v_from_field(p1);

    k2 = const_vect_mult(time_step, v_1);
    Point p2 = add_vector_point(p, const_vect_mult(.5, k2));
    Vector v_2 = get_v_from_field(p2);

    k3 = const_vect_mult(time_step, v_2);
    Point p3 = add_vector_point(p, k3);
    Vector v_3 = get_v_from_field(p3);

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



