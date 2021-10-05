#include <iostream>
#include <thread>
#include <mutex>
#include <random>
#include <string>



class CreateHistogram{
public:
    CreateHistogram(int thread_count, int bin_count, float min_meas, float max_meas, int data_count)
    {
        m_thread_count = thread_count;
        m_bin_count = bin_count;
        m_min_meas = min_meas;
        m_max_meas = max_meas;
        m_data_count = data_count;
        m_data_array = new float[m_data_count];
        for(int i = 0; i < data_count; i++)
        {
            m_data_array[i] = get_random();
        }
        m_bin_maxes = new float[m_bin_count];
        m_bin_counts = new float[m_bin_count];
        init_bins();
    }

    void do_the_thing()
    {
        std::thread threads[m_thread_count];
        for(int i = 0; i < m_thread_count; i++)
        {
            threads[i] = std::thread([this, i]{
                int partition = (int) ceil( (float)m_data_count / m_thread_count );
                int first = partition * i;
                int last = first + partition - 1;
                for(int k = first; k <= last; k++)
                    sort_num(m_data_array[k]);
            });
        }
        for(int i = 0; i < m_thread_count; i++)
        {
            threads[i].join();
        }
        std::cout << "bin_maxes: ";
        for(int i = 0; i < m_bin_count; i++)
            std::cout << m_bin_maxes[i] << " ";
        std::cout << std::endl;
        std::cout << "bin_counts: ";
        for(int i = 0; i < m_bin_count; i++)
            std::cout << m_bin_counts[i] << " ";

    }


private:
    int m_thread_count;
    int m_bin_count;
    float m_min_meas;
    float m_max_meas;
    int m_data_count;
    float *m_data_array;
    float *m_bin_maxes;
    float *m_bin_counts;
    std::mutex mut;

    void init_bins()
    {
        float size = (float)(m_max_meas - m_min_meas) / (float)m_bin_count;
        float current = size;
        for(int i = 0; i < m_bin_count; i++)
        {
            m_bin_maxes[i] = current;
            m_bin_counts[i] = 0;
            current += size;
        }
    }

    void sort_num(float num)
    {
        for(int i = 0; i < m_bin_count; i++)
        {
            if (num <= m_bin_maxes[i])
            {
                mut.lock();
                m_bin_counts[i]++;
                mut.unlock();
                return;
            }
        }
    }
    // 14 11 8 17 9 15 8 12 3 7
    float get_random()
    {
        float random = ((float) rand()) / (float) RAND_MAX; // return a float between 0 and 1

        float range = m_max_meas - m_min_meas;
        return (random*range) + m_min_meas;
    }
};


// Issues arise from different threads updating the array of bin_maxes at the same time.
// I solved this issue by wrapping the updating in a mutex so that only could update at a time.



// description here of what order things are passed in
// 1. thread count
// 2. bin count
// 3. min
// 4. max
// 5. data count
int main(int argc, char* argv[])
{
    srand(100);

    //argv[0] is the actual command line, the first parameter is argv[1]
    std::string t_count_string(argv[1]);
    int thread_count = std::stoi(t_count_string);


    std::string b_count_string(argv[2]);
    int bin_count = std::stoi(b_count_string);

    std::string min_meas_string(argv[3]);
    float min_meas = std::stof(min_meas_string);

    std::string  max_meas_string(argv[4]);
    float max_meas = std::stof(max_meas_string);

    std::string d_count_string(argv[5]);
    int data_count = std::stoi(d_count_string);

    auto creation = new CreateHistogram(thread_count, bin_count, min_meas, max_meas, data_count);
    creation->do_the_thing();

    return 0;
}
