#%%cython

"""
A working version

"""

from itertools import combinations
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float_t DTYPE_FLOAT_t
ctypedef np.int_t DTYPE_INT_t

@cython.boundscheck(False)
cdef void to_do_simplified(int body1,
                     int body2,
                    np.ndarray[DTYPE_FLOAT_t, ndim=2] POSITIONS,
                    np.ndarray[DTYPE_FLOAT_t, ndim=2] VELOCITIES,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] MASSES,
                    float dt
                    ):
    
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] xyz1 = POSITIONS[body1]
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] xyz2 = POSITIONS[body2]

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] v1 = VELOCITIES[body1]
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] v2 = VELOCITIES[body2]

    cdef float m1 = MASSES[body1]
    cdef float m2 = MASSES[body2]
                
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] d = xyz1 - xyz2
    
    cdef float mag = dt * (np.inner(d, d) ** (-1.5))

    cdef body1_velocity_arg = m2 * mag
    VELOCITIES[body1] -= d * body1_velocity_arg

    cdef body2_velocity_arg = m1 * mag   
    VELOCITIES[body2] += d * body2_velocity_arg
    

@cython.boundscheck(False)    
cdef void update_rs_for_body(int body,
                       np.ndarray[DTYPE_FLOAT_t, ndim=2] POSITIONS,
                       np.ndarray[DTYPE_FLOAT_t, ndim=2] VELOCITIES,
                       np.ndarray[DTYPE_FLOAT_t, ndim=1] MASSES,
                       float dt):
    POSITIONS[body] += dt * VELOCITIES[body]

    
@cython.boundscheck(False)    
cdef float report_energy( np.ndarray[DTYPE_FLOAT_t, ndim=2] POSITIONS,
                   np.ndarray[DTYPE_FLOAT_t, ndim=2] VELOCITIES,
                   np.ndarray[DTYPE_FLOAT_t, ndim=1] MASSES,
                   np.ndarray[DTYPE_INT_t, ndim=1] KEYS,
                   np.ndarray[DTYPE_INT_t, ndim=2] VISIT_SCHEDULE):
    '''
        compute the energy and return it so that it can be printed
    '''
    cdef float e = 0.0
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] xyz1, xyz2, d
    cdef float m1, m2
    for (body1, body2) in VISIT_SCHEDULE:
        xyz1 = POSITIONS[body1]
        xyz2 = POSITIONS[body2]


        m1 = MASSES[body1]
        m2 = MASSES[body2]

        d = xyz1 - xyz2
        e -= (m1 * m2) / np.sqrt(np.inner(d, d))

        
    for body in KEYS:
        v = VELOCITIES[body]
        m = MASSES[body]   

        e += m * np.inner(v, v) / 2.0
    return e



@cython.boundscheck(False)
cdef void offset_momentum(int reference_index,
                   np.ndarray[DTYPE_FLOAT_t, ndim=2] POSITIONS,
                   np.ndarray[DTYPE_FLOAT_t, ndim=2] VELOCITIES,
                   np.ndarray[DTYPE_FLOAT_t, ndim=1] MASSES,
                   np.ndarray[DTYPE_INT_t, ndim=1] KEYS
                   ):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    cdef np.ndarray p = np.array([0.0, 0.0, 0.0])
    for body in KEYS:
        v = VELOCITIES[body]
        m = MASSES[body]

        p -= v * m

    m = MASSES[reference_index]    
    
    VELOCITIES[reference_index] = p / m



cdef nbody(int loops, str reference, int iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    # initialize stuff
    cdef float PI 
    PI = 3.14159265358979323

    cdef float SOLAR_MASS
    SOLAR_MASS = 4 * PI * PI

    cdef float DAYS_PER_YEAR
    DAYS_PER_YEAR = 365.24



    BODIES_NAMES = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']
    cdef np.ndarray[DTYPE_INT_t, ndim=1] KEYS = np.arange(5)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] POSITIONS = np.array([
        [0.0, 0.0, 0.0],
        [4.84143144246472090e+00,
                     -1.16032004402742839e+00,
                     -1.03622044471123109e-01],
        [8.34336671824457987e+00,
                    4.12479856412430479e+00,
                    -4.03523417114321381e-01],
        [1.28943695621391310e+01,
                    -1.51111514016986312e+01,
                    -2.23307578892655734e-01],
        [1.53796971148509165e+01,
                     -2.59193146099879641e+01,
                     1.79258772950371181e-01],
    ])

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] VELOCITIES = np.array([
        [0.0, 0.0, 0.0],
        [1.66007664274403694e-03 * DAYS_PER_YEAR,
                     7.69901118419740425e-03 * DAYS_PER_YEAR,
                     -6.90460016972063023e-05 * DAYS_PER_YEAR],
        [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                    4.99852801234917238e-03 * DAYS_PER_YEAR,
                    2.30417297573763929e-05 * DAYS_PER_YEAR],

         [2.96460137564761618e-03 * DAYS_PER_YEAR,
                    2.37847173959480950e-03 * DAYS_PER_YEAR,
                    -2.96589568540237556e-05 * DAYS_PER_YEAR],
        [2.68067772490389322e-03 * DAYS_PER_YEAR,
                     1.62824170038242295e-03 * DAYS_PER_YEAR,
                     -9.51592254519715870e-05 * DAYS_PER_YEAR],  
    ])

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] MASSES = np.array([
    SOLAR_MASS,
    9.54791938424326609e-04 * SOLAR_MASS,
    2.85885980666130812e-04 * SOLAR_MASS,
    4.36624404335156298e-05 * SOLAR_MASS,
    5.15138902046611451e-05 * SOLAR_MASS,
    ])


    cdef np.ndarray[DTYPE_INT_t, ndim=2] VISIT_SCHEDULE = np.array(list(combinations(KEYS, 2)))

    cdef float dt = 0.01
    # ================
    
    
    
    
    # Set up global state
    offset_momentum(BODIES_NAMES.index(reference), POSITIONS, VELOCITIES, MASSES, KEYS)

    for _ in range(loops):
        for _ in range(iterations):            
            for body1, body2 in VISIT_SCHEDULE:
                to_do_simplified(body1, body2, POSITIONS, VELOCITIES, MASSES, dt)   
            for b in KEYS:
                update_rs_for_body(b, POSITIONS, VELOCITIES, MASSES, dt)
        print(report_energy(POSITIONS, VELOCITIES, MASSES, KEYS, VISIT_SCHEDULE))
            

if __name__ == '__main__':
    nbody(100, 'sun', 20000)

nbody(100, 'sun', 20000)