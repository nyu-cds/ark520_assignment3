import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# because I "don't" know the underlying distribution of the data,
# I'm choosing SIZE ssplit points randomly from the data

if rank==0:
    data = np.random.random(100)
    # choose an integer for eveness of buckets
    # The higher the integer, the more even the distribution 
    # over processes should be, but the longer rank 0 spends
    # before distributing
    eveness_parameter = 2

    splitters = np.random.choice(data, replace=False, size=(size-1)*eveness_parameter)
    splitters = list(np.reshape(np.sort(splitters), (size-1, eveness_parameter)).mean(axis=1))
    low_highs = list(zip([np.NINF] + splitters, splitters + [np.Inf]))

    def is_in_range(x, low_high):
        return low_high[0] <= x and x < low_high[1]
    def get_bucket_number(x, low_highs=low_highs):
        for i, lh in enumerate(low_highs):
            if is_in_range(x, lh):
                return i
        return i
    bucket_numbers = np.array(list(map(get_bucket_number, data)))

    for i in range(size):
        # send bucket_numbers
        comm.isend( len(data[bucket_numbers==i]), i, tag=0 )
        comm.Isend(data[bucket_numbers==i], i, tag=1)

len_to_prep = comm.recv(source=0, tag=0)
data = np.full(len_to_prep, np.nan)
comm.Recv(data, source=0, tag=1)

print("rank ", rank, " got ", data)
data = np.sort(data)

# I am not sure if gather ensures order by node
# So I'm sending it with metadata
data = (rank, data)
ranks, datas = comm.gather(data, root=0)

if rank==0:
    print(datas)