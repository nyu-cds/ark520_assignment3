import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# because I "don't" know the underlying distribution of the data,
# I'm choosing SIZE split points randomly from the data

from numpy.random import permutation

if rank==0:
    data = np.random.random(1000)
    # For a sanity check, I have also used these
    #  data = np.random.random(5)
    #  data = np.random.randint(0, 11, 10000)
    data = np.array(data, float)
        
    #for a sanity check, ensure these hashes match up at end
    orig_len_of_data = len(data)
    sum_data = np.sum(data)
    sum_data2 = np.sum(data**2)
    
    # choose an integer for eveness of buckets
    # The higher the integer, the more even the distribution 
    # over processes should be, but the longer rank 0 spends
    # before distributing
    eveness_parameter = 5

    splitters = np.random.choice(data, replace=True, size=(size-1)*eveness_parameter)
    splitters = list(np.reshape(np.sort(splitters), (size-1, eveness_parameter)).mean(axis=1))
    low_highs = list(zip([np.NINF] + splitters, splitters + [np.Inf]))
    
    def is_in_range(x, low_high):
        return low_high[0] <= x and x < low_high[1]
    def get_bucket_number(x, low_highs=low_highs):
        # this ensures each datapoint will only get sent to one bucket
        # even if there are ties
        for i, lh in enumerate(low_highs):
            if is_in_range(x, lh):
                return i
        return i
    bucket_numbers = np.array(list(map(get_bucket_number, data)))
    
    # since collective communication is faster, I'm going to first do a
    # rough reshuffling.
    # then use Scatterv and Gatherv
    len_per_bucket = []
    roughcut_data = data.copy() * np.nan
    for i in range(size):
        d = data[bucket_numbers==i]
        roughcut_data[ int(np.sum(len_per_bucket)): int(np.sum(len_per_bucket)) + len(d)] = d
        len_per_bucket.append(len(d))
    del data
    displacements = np.array(np.cumsum([0] + len_per_bucket)[0:size], dtype=int)
    len_per_bucket, displacements = tuple(len_per_bucket), tuple(displacements)
    
# broadcast size and displacements
if rank!=0:
    len_per_bucket, displacements = None, None
len_per_bucket, displacements = comm.bcast((len_per_bucket, displacements), root=0)

# prepare buffers of right size
data_local = np.full(len_per_bucket[rank],
                     np.nan,
                    dtype=float)

if rank!=0:
    roughcut_data = None
comm.Scatterv([roughcut_data,
               tuple(len_per_bucket),
               tuple(displacements),
              MPI.DOUBLE
              ],
             data_local)

# do the actual work
data_local = np.sort(data_local)

# gather
if rank==0:
    data = np.zeros(  int(sum(len_per_bucket)) )
else: 
    data = None
comm.Gatherv(data_local, 
             [data,
               tuple(len_per_bucket),
               tuple(displacements),
              MPI.DOUBLE
              ],
            )
    
if rank==0:
    # check data is sorted
    assert np.alltrue(np.sort(data)==data)
    
    # check data is the size we were expecting
    assert len(data)== orig_len_of_data
    
    # check data is probably exactly the same as what was put in
    # by checking first and second moment
    # we have to use np.isclose(), because, oddly, precision was 
    # being lost, probably because of float -> DOUBLE conversion
    assert np.isclose(np.sum(data), sum_data)
    assert np.isclose(np.sum(data**2), sum_data2)
    
    print(data)