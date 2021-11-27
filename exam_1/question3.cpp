/* Generate the data */
Generate_data(min_meas, max_meas, data, data_count);

/* Create bins for storing counts */
Generate_bins(min_meas, max_meas, bin_maxes, bin_counts, bin_count);

/* Count number of values in each bin */
int i, j;
# pragma omp parallel num_threads(thread_count)
#ifdef _OPENMP
int my_rank = omp_get_thread_num();
#else
int my_rank = 0;
#endif
int my_offset = my_rank*bin_count;

//It's important note that for both for loops, the results of one loop don't affect another. We can therefore use OpenMP
#   pragma omp for private(bin, i) shared(data_count, loc_bin_counts, bin_maxes, bin_count, min_meas) 
for (i = 0; i < data_count; i++) {
  bin = What_bin(data[i], bin_maxes, bin_count, min_meas);
  loc_bin_counts[my_offset + bin]++;
}

//Here, each thread will loop the number of threads * number of elements per thread times. (I'm not sure how to use OpenMP
#   pragma omp for private(i, j) shared(data_count, bin_maxes, bin_count, min_meas)
for (i = 0; i < bin_count; i++)
  for (j = 0; j < thread_count; j++) {
	//Here, note that j*bin_count + 1 is the same as my_offset above and i is the bin (conceptually).
    bin_counts[i] += loc_bin_counts[j*bin_count + i];
  }