#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

float get_random(float max, float min);
void init_data(int count, float max, float min, float* to_populate);
void init_bins(float max, float min, int num_bins, float* bin_limits);
void Get_arg(int argc, char* argv[]);

int comm_sz; //Number of Process
int my_rank; //Rank of current process
MPI_Comm comm;
int bin_count;
int data_count;
float max_meas;
float min_meas;
float *bin_maxes;

// description here of what order things are passed in
// 1. bin count
// 2. min
// 3. max
// 4. data count
int main(int argc, char* argv[]) {

    float* local_data;

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    auto* data = new float[data_count];

    //Get_arg includes broadcasting
    Get_arg(argc, argv);

    if(my_rank == 0)
    {
        srand(100);
        init_data(data_count, max_meas, min_meas, data);
        MPI_Scatter(data, data_count, MPI_FLOAT, local_data, MPI_FLOAT, 0, comm);
        //TODO: receive data from other processes using MPI_Receive
    }
    else
    {
        MPI_Scatter(data, data_count, MPI_FLOAT, local_data, MPI_FLOAT, 0, comm);
        //TODO:
        //Do the sorting into bins here (local data into some local filled_bins).
        //Send the results back to process 0 using MPI_Send
    }

    MPI_Finalize(&argc, &argv);
    return 0;
}

/**
 * Get input from the user, store, and broadcast
 * @param argc number of arguments
 * @param argv array containing the arguments
 */
void Get_arg(int argc, char* argv[])
{
    //TODO: time permitting, add input validation
    bin_maxes = new float[bin_count];

    if(my_rank == 0)
    {
        std::string b_count_string(argv[1]);
        bin_count = std::stoi(b_count_string);

        std::string min_meas_string(argv[2]);
        min_meas = std::stof(min_meas_string);

        std::string  max_meas_string(argv[3]);
        max_meas = std::stof(max_meas_string);

        std::string d_count_string(argv[4]);
        data_count = std::stoi(d_count_string);

        init_bins(max_meas, min_meas, bin_count, bin_maxes);
    }
    MPI_Bcast(bin_count, 1, MPI_INT, 0, comm);
    MPI_Bcast(bin_maxes, bin_count, MPI_FLOAT, 0, comm);
}

/**
 * Get a random number between two values.
 * @param max the max value
 * @param min the min value
 * @return the random number
 */
float get_random(float max, float min)
{
    float random = ((float) rand()) / (float) RAND_MAX;
    float range = max - min;
    return (random * range) + min;
}

/**
 * Initialize the data array with count number of random numbers
 * @param count the number of elements in the array
 * @param max the max value
 * @param min the min value
 * @param to_populate the array in which to store the data
 */
void init_data(int count, float max, float min, float* to_populate)
{
    for(int i = 0; i < count; i++)
    {
        to_populate[i] = get_random(max, min);
    }
}

/**
 * Initialize the bin sizes
 * @param max the max value
 * @param min the min value
 * @param num_bins number of bins
 * @param bin_limits the top limits of the bins
 */
void init_bins(float max, float min, int num_bins, float* bin_limits)
{
    float size = (float)(max - min) / (float)num_bins;
    float current = size;
    for(int i = 0; i < num_bins; i++)
    {
        bin_limits[i] = current;
        current += size;
    }
}
