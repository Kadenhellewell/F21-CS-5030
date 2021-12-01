#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

float get_random(float max, float min);
void init_data(int count, float max, float min, float* to_populate);
void init_bins();
void Get_arg(int argc, char* argv[]);

int comm_sz; //Number of Process
int my_rank; //Rank of current process
MPI_Comm comm;
int bin_count;
int data_count;
float max_meas;
float min_meas;
float* bin_maxes;
int local_n;

// description here of what order things are passed in
// 1. bin count
// 2. min
// 3. max
// 4. data count
int main(int argc, char* argv[]) {


    float* local_data;

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);
    Get_arg(argc, argv);
    std::cout << "I am here " << my_rank << std:: endl;

    float* data = new float[data_count];
    float* filled_bins = new float[bin_count];
    //Get_arg includes broadcasting

    if(my_rank == 0)
    {

        srand(100);
        //init_data(data_count, max_meas, min_meas, data);
        /*
        MPI_Scatter(data, local_n, MPI_FLOAT, local_data, local_n, MPI_FLOAT, 0, comm);
        for(int i = 0; i < bin_count; i++)
        {
            filled_bins[i] = 0;
        }
        //TODO: receive data from other processes using MPI_Receive
        for(int i = 0; i < comm_sz; i++)
        {
            float *incoming_bins = new float[local_n];
            MPI_Recv(incoming_bins, local_n, MPI_FLOAT, i, 0, comm, MPI_STATUS_IGNORE);
            for(int j = 0; j < bin_count; j++)
                filled_bins[j] += incoming_bins[j];
	    delete[] incoming_bins;
        }
        std::cout << "The bins are: ";
        for(int i = 0; i < bin_count; i++)
        {
            std::cout << filled_bins[i] << ", ";
        }
        std::cout << std::endl;*/
    }
    else
    {
/*
        local_data = new float[local_n];
	MPI_Scatter(data, local_n, MPI_FLOAT, local_data, local_n, MPI_FLOAT, 0, comm);
        float *local_bins = new float[bin_count];
        //Initialize local bins to zero
        for(int i = 0; i < local_n; i++)
        {
            local_bins[i] = 0;
        }
        //TODO:
        //Do the sorting into bins here (local data into some local filled_bins).
        //Send the results back to process 0 using MPI_Send
        for(int i = 0; i < local_n; i++)
        {
            for(int j = 0; j < bin_count; j++)
            {
                if (local_data[i] <= bin_maxes[j])
                {
                    local_bins[j]++;
                    break;
                }
            }
        }
        MPI_Send(local_bins, local_n, MPI_FLOAT, 0, 0, comm);
	delete[] local_bins;
	delete[] local_data;*/
    }
    
    delete[] data;
    delete[] filled_bins;
    delete[] bin_maxes;
    MPI_Finalize();
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
    std::cout << "I get inside get-arg" << std::endl;
    if(my_rank == 0)
    {
        std::string b_count_string(argv[1]);
        bin_count = atoi(argv[1]);
	std::cout << "I set bin_count" << std::endl;
        std::string min_meas_string(argv[2]);
        min_meas = atof(argv[2]);
	std::cout << "I set min meas" << std::endl;
        std::string  max_meas_string(argv[3]);
        max_meas = atof(argv[3]);
	std::cout << "I set max_meas" << std::endl;
        std::string d_count_string(argv[4]);
        data_count = atoi(argv[4]);
	std::cout << "I set data_count" << std::endl;
        local_n = data_count / comm_sz;
        bin_maxes = new float[bin_count];
        init_bins();
    }
    std::cout << "I get to the barrier " << my_rank << std::endl;
    MPI_Barrier(comm);
    std::cout << "I get to the bcasts " << my_rank << std::endl;
    MPI_Bcast(&bin_count, 1, MPI_INT, 0, comm);
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
void init_bins()
{
    float size = (float)(max_meas - min_meas) / (float)bin_count;
    float current = size;
    for(int i = 0; i < bin_count; i++)
    {
	std::cout << "This is the bin max: " << current << " at " << i << std::endl;
        bin_maxes[i] = current;
        current += size;
    }
}
