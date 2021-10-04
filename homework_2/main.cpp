#include <iostream>
#include <thread>
#include <mutex>
#include <random>

// Arguments passed in (in this order)
int thread_count;
int bin_count;
float min_meas;
float max_meas;
int data_count;

void sort_num(float num)
{

}

void threadFunction(float array[])
{
    for (int i = 0; i < data_count; i++)
    {
        sort_num(array[i]);
    }
}

float get_random()
{
    float random = ((float) rand()) / (float) RAND_MAX; // return a float between 0 and 1

    float range = max_meas - min_meas;
    return (random*range) + min_meas;
}

// description here of what order things are passed in
// 1. thread count
// 2. bin count
// 3. min
// 4. max
// 5. data count
int main(int argc, char* argv[])
{
    srand(100);
    thread_count = (int) argv[0];
    bin_count = (int) argv[1];
    std::string min_meas_string(argv[2]);
    min_meas = std::stof(min_meas_string);
    std::string  max_meas_string(argv[3]);
    max_meas = std::stof(max_meas_string);
    data_count = (int) argv[4];

    bin_size = (int) argv[0];
    data_count = (int) argv[1];
    float nums[data_count];
    for(float k : nums)
        k = get_random();

    std::thread threads[thread_count];
    for(int i = 0; i < thread_count; i++)
    {
        threads[i] = new std::thread(threadFunction, i);
    }
    for(int i = 0; i < thread_count; i++)
    {
        threads[i].join();
    }


    return 0;
}
