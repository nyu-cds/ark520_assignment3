from numpy import empty
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

current_integer = empty(1, dtype=int)

if size < 2:
    print("Program was expecting more ranks.  It will do it's best and just return your input.")

if rank == 0:
    
    # get the user input
    def is_integer_less_than_100(x):
        try:
            return float(x) == int(x) and int(x)<100
        except:
            return False

    user_integer = input('Enter an integer less than 100: ')
    while not is_integer_less_than_100(user_integer):
        print("Not a valid integer less than 100")
        user_integer = input('Enter an integer less than 100: ')
    current_integer[0] = int(user_integer)
    
    if size > 1:
        # send it to process 1
        comm.Send(current_integer, dest=1)

        # get the final answer and print it
        comm.Recv(current_integer, source=size-1)
    print(current_integer[0])
    
if rank >= 1:
    comm.Recv(current_integer, source=rank-1)
    current_integer *= rank
    if rank == size -1:
        comm.Isend(current_integer, dest=0)
    else: 
        comm.Isend(current_integer, dest=rank + 1)