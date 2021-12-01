/**
 * Report:
 * Compile: mpic++ -g -Wall -0 <output file name> <source file name>
 * Execute: mpiexec -n <num processes> ./<output file name> <bin_count> <min val> <max val> <data count>
 * I worked on notchpeak with modules gcc/6 and mpich.
 *
 * I was unable to get to the scaling part of the assignment.
 * I was also unable to get the algorithm quite right. I think the error is in my use of MPI_Scatter (processes have
 * nothing in their local_data).
 * Unlike the previous assignment, I was actually able to get this to compile and run to completion, though my
 * results were not correct.
 */


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
    //Get_arg includes broadcasting
    Get_arg(argc, argv);

    float* data = new float[data_count];
    float* bin_counts = new float[bin_count];
    local_data = new float[local_n];

    if(my_rank == 0)
    {
        srand(100);
        init_data(data_count, max_meas, min_meas, data);
        //Initialize bin counts to zero
        for(int i = 0; i < bin_count; i++)
        {
            bin_counts[i] = 0;
        }
        //Scatter array to other processes
        MPI_Scatter(data, local_n, MPI_FLOAT, local_data, local_n, MPI_FLOAT, 0, comm);
    }
    else
    {
        //Receive scattering from 0
	    MPI_Scatter(data, local_n, MPI_FLOAT, local_data, local_n, MPI_FLOAT, 0, comm);
        float *local_bins = new float[bin_count];
        //Initialize local bin counts to zero
        for(int i = 0; i < local_n; i++)
        {
            local_bins[i] = 0;
        }

        //Compute local histogram
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

        //Send local histogram back to process 0
        MPI_Send(local_bins, local_n, MPI_FLOAT, 0, 0, comm);
	    delete[] local_bins;
	    delete[] local_data;
    }

    if(my_rank == 0)
    {
        for(int i = 1; i < comm_sz; i++)
        {
            float *incoming_bins = new float[local_n];
            //Receive local histogram from each process
            MPI_Recv(incoming_bins, local_n, MPI_FLOAT, i, 0, comm, MPI_STATUS_IGNORE);
            //Add local histogram into global histogram (I would have used reduce here, but I didn't know how to do that with
            //an array)
            for(int j = 0; j < bin_count; j++)
                bin_counts[j] += incoming_bins[j];
            delete[] incoming_bins;
        }
        //Report final bin counts
        std::cout << "The bins counts are: ";
        for(int i = 0; i < bin_count; i++)
        {
            std::cout << bin_counts[i] << ", ";
        }
        std::cout << std::endl;
    }
    
    delete[] data;
    delete[] bin_counts;
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
    MPI_Barrier(comm);
    if(my_rank == 0)
    {
        std::string b_count_string(argv[1]);
        bin_count = atoi(argv[1]);
	
        std::string min_meas_string(argv[2]);
        min_meas = atof(argv[2]);

        std::string  max_meas_string(argv[3]);
        max_meas = atof(argv[3]);
	
        std::string d_count_string(argv[4]);
        data_count = atoi(argv[4]);
        
	    local_n = data_count / comm_sz;
    }
    MPI_Barrier(comm);
    MPI_Bcast(&bin_count, 1, MPI_INT, 0, comm);

    bin_maxes = new float[bin_count];
    if(my_rank == 0)
    {
	    init_bins();
    }
    
    MPI_Barrier(comm);
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
    std::cout << std::endl;
}

/**
 * Initialize the bin sizes
 */
void init_bins()
{
    float size = (float)(max_meas - min_meas) / (float)bin_count;
    float current = size;
    //Calculate and report bin maxes
    std::cout << "The bin maxes are: ";
    for(int i = 0; i < bin_count; i++)
    {
        bin_maxes[i] = current;
	    std::cout << current << ", ";
        current += size;
    }
    std::cout << std::endl;
}
