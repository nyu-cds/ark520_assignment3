import numpy as np
from mpi4py import MPI

# learn about COMM environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# determine the parity of the rank
odd_rank = (rank % 2) == 1

# print
if odd_rank:
    print("Hello from process ", rank)
else: 
    print("Goodbye from process ", rank)