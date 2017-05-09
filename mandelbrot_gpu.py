
# 
# A CUDA version to calculate the Mandelbrot set
#
from numba import cuda
import numpy as np
from pylab import imshow, show

@cuda.jit(device=True)
def mandel(x, y, max_iters):
    '''
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the 
    Mandelbrot set given a fixed number of iterations.
    '''
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@cuda.jit
def compute_mandel(min_x, max_x, min_y, max_y, image, iters):
    '''
    This function is called once per block.  Then I manually iterate threads.
    Because the blockdimy*griddimy != 1536, each block (and each thread within each block)
    will be responsible for more than one cell
    '''
    start_row, start_col = cuda.grid(2)
    
    # Thread id in a 1D block
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    # Block id in a 1D grid
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    # Block width, i.e. number of threads per block
    bw, bh = cuda.blockDim.x, cuda.blockDim.y
    # Grid width
    gw, gh = cuda.gridDim.x, cuda.gridDim.y

    # Compute flattened index inside the array    

    height = image.shape[0]
    width = image.shape[1]
    
    # Each row is divided into 32 blocks of 32 threads
    # Each column is divided into 16 blocks of 8 threads
    
    cols_per_thread = math.ceil( width  / (gw * bw) )
    rows_per_thread = math.ceil( height / (gh * bh) )
    
    min_col = (bx * bw + tx) * cols_per_thread
    min_row = (by * bh + ty) * rows_per_thread

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(min_col, min_col + cols_per_thread):
        real = min_x + x * pixel_size_x
        for y in range(min_row, min_row + rows_per_thread):
            imag = min_y + y * pixel_size_y
            if y < image.shape[0] and x < image.shape[1]:
                image[y, x] = mandel(real, imag, iters)

    ### YOUR CODE HERE
    
if __name__ == '__main__':
    image = np.zeros((1024, 1536), dtype = np.uint8)
    blockdim = (32, 8)
    griddim = (32, 16)

    image_global_mem = cuda.to_device(image)
    compute_mandel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
    image_global_mem.copy_to_host()
    imshow(image)
    show()