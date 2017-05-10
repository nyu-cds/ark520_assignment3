import itertools
import numpy as np
from numba import jit, int32, float64, char, vectorize

@vectorize([float64(float64, float64)])
def vec_deltas(a, b):
    # fast computation of deltas, as per assignment
    return a - b

@jit("float64(float64[:])")
def l2_norm_len3(a):
    # computes the L2 norm very very quickly
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]

@jit("void(int32, int32, float64[:,:], float64[:,:], float64[:], float64)", nopython=True)
def to_do(body1, body2, POSITIONS, VELOCITIES, MASSES, dt):
    (xyz1, v1, m1) = POSITIONS[body1], VELOCITIES[body1], MASSES[body1]
    (xyz2, v2, m2) = POSITIONS[body2], VELOCITIES[body2], MASSES[body2]

    d = vec_deltas(xyz1, xyz2)
    
    mag = dt * (l2_norm_len3(d) ** (-1.5))

    

    body1_velocity_arg = m2 * mag
    v1[:] -= d * body1_velocity_arg


    body2_velocity_arg = m1 * mag       
    v2[:] += d * body1_velocity_arg

@jit("void(int32, float64[:,:], float64[:,:], float64[:], float64)", nopython=True)
def update_rs_for_body(body, POSITIONS, VELOCITIES, MASSES, dt):
    (r, vxyz, m) = POSITIONS[body], VELOCITIES[body], MASSES[body]
    r[:] += dt * vxyz

@jit("float64(float64[:,:], float64[:,:], float64[:], int32[:])", nopython=True)
def report_energy(POSITIONS, VELOCITIES, MASSES, BODIES_KEYS):
    '''
        compute the energy and return it so that it can be printed
    '''
    e = float(0.0)
    
    for body1 in BODIES_KEYS:
        for body2 in range(body1+1, len(BODIES_KEYS)):

            #(body1, body2) = VISIT_SCHEDULE[i, 0], VISIT_SCHEDULE[i, 1]
            (xyz1, v1, m1) = POSITIONS[body1], VELOCITIES[body1], MASSES[body1]
            (xyz2, v2, m2) = POSITIONS[body2], VELOCITIES[body2], MASSES[body2]

            d = vec_deltas(xyz1, xyz2)
            
            # Amazingly the code as written is faster than this code below.
            e -= (m1 * m2) / np.sqrt(l2_norm_len3(d))

        
    for body in BODIES_KEYS:
        (r, v, m) = POSITIONS[body], VELOCITIES[body], MASSES[body]
        e += m * (l2_norm_len3(v)) / 2.
        
    return e

@jit("void(int32, float64[:,:], float64[:,:], float64[:], int32[:])", nopython=True)
def offset_momentum(ref, POSITIONS, VELOCITIES, MASSES, BODIES_KEYS):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    p = np.zeros(3)
    for body in BODIES_KEYS:
        (r, v, m) = POSITIONS[body], VELOCITIES[body], MASSES[body]
        p -= v*m
        
    (r, v, m) = POSITIONS[ref], VELOCITIES[ref], MASSES[ref]
    
    # Necessary syntax to allow memory sharing
    v[:] = p / m

@jit("void(int32, char, int32)")
def nbody(loops, reference, iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    
    # initialize stuff
    # initialization done inside for the sake of jit and shared-memory issues
    # ====================================
    PI = 3.14159265358979323
    SOLAR_MASS = 4 * PI * PI
    DAYS_PER_YEAR = 365.24
    BODIES_NAMES = ['sun', 'jupiter', 'saturn', 'uranus', 'neptune']
    BODIES_KEYS = np.array( list(range(5)), dtype=np.int32)
    POSITIONS = np.array([
        [0.0, 0.0, 0.0],
        [4.84143144246472090e+00,                 -1.16032004402742839e+00,                 -1.03622044471123109e-01],
        [8.34336671824457987e+00,                4.12479856412430479e+00,                -4.03523417114321381e-01],
        [1.28943695621391310e+01,                -1.51111514016986312e+01,                -2.23307578892655734e-01],
        [1.53796971148509165e+01,                 -2.59193146099879641e+01,                 1.79258772950371181e-01],
    ], dtype=np.float64)

    VELOCITIES = np.array([
        [0.0, 0.0, 0.0],
        [1.66007664274403694e-03 * DAYS_PER_YEAR,                 7.69901118419740425e-03 * DAYS_PER_YEAR,                 -6.90460016972063023e-05 * DAYS_PER_YEAR],
        [-2.76742510726862411e-03 * DAYS_PER_YEAR,                4.99852801234917238e-03 * DAYS_PER_YEAR,                2.30417297573763929e-05 * DAYS_PER_YEAR],
        [2.96460137564761618e-03 * DAYS_PER_YEAR,                2.37847173959480950e-03 * DAYS_PER_YEAR,                -2.96589568540237556e-05 * DAYS_PER_YEAR],
        [2.68067772490389322e-03 * DAYS_PER_YEAR,                 1.62824170038242295e-03 * DAYS_PER_YEAR,                 -9.51592254519715870e-05 * DAYS_PER_YEAR],  
    ], dtype=np.float64)
    VISIT_SCHEDULE = np.array(list(itertools.combinations(BODIES_KEYS, 2)), dtype=np.int32)

    MASSES = np.array([ SOLAR_MASS, 9.54791938424326609e-04 * SOLAR_MASS, 2.85885980666130812e-04 * SOLAR_MASS, 4.36624404335156298e-05 * SOLAR_MASS, 5.15138902046611451e-05 * SOLAR_MASS,])
    dt = 0.01
    # ====================================
    
    
    
    
    
    # Set up global state
    offset_momentum(BODIES_NAMES.index(reference), POSITIONS, VELOCITIES, MASSES, BODIES_KEYS)

    total_runs = loops * iterations
    for i, (body1, body2) in enumerate(itertools.cycle(VISIT_SCHEDULE)):
        to_do(body1, body2, POSITIONS, VELOCITIES, MASSES, dt)
    
        
        if (i % iterations) == 0:
            for body in BODIES_KEYS:
                update_rs_for_body(body, POSITIONS, VELOCITIES, MASSES, dt)
                
        if (i % loops) == 0:
            print(report_energy(POSITIONS, VELOCITIES, MASSES, BODIES_KEYS))
                  


if __name__ == '__main__':
    nbody(100, 'sun', 20000)