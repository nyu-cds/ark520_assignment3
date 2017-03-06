"""
For assignment6, I got a speedup of 54%.
"""
 
""" 
    
    The relative speedups are: 
    {1: 2.5943132204285995, 2: 1.162570218357162, 3: 1.0218080537714478, 4: 0.99983733795216001, 'orig': 1.0, 'opt': 2.8114968869479546}
    
    The time it took to run is:
    {1: 30.994290845002979, 2: 69.164767191978171, 3: 78.692762500955723, 4: 80.421980100974906, 'orig': 80.408898497000337, 'opt': 28.600031132984441}
    
    I did break 30 seconds. :)
    
"""

"""
    By far, the biggest change was removing the duplicated function calls.
    
    I tried parallelization for the loops where it was safe to do them, but it was actually slower.     
     
    N-body simulation.
"""
import itertools
import numpy as np

PI = 3.14159265358979323
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24

BODIES = {
    'sun': (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), SOLAR_MASS),

    'jupiter': (np.array([4.84143144246472090e+00,
                 -1.16032004402742839e+00,
                 -1.03622044471123109e-01]),
                np.array([1.66007664274403694e-03 * DAYS_PER_YEAR,
                 7.69901118419740425e-03 * DAYS_PER_YEAR,
                 -6.90460016972063023e-05 * DAYS_PER_YEAR]),
                9.54791938424326609e-04 * SOLAR_MASS),

    'saturn': (np.array([8.34336671824457987e+00,
                4.12479856412430479e+00,
                -4.03523417114321381e-01]),
               np.array([-2.76742510726862411e-03 * DAYS_PER_YEAR,
                4.99852801234917238e-03 * DAYS_PER_YEAR,
                2.30417297573763929e-05 * DAYS_PER_YEAR]),
               2.85885980666130812e-04 * SOLAR_MASS),

    'uranus': (np.array([1.28943695621391310e+01,
                -1.51111514016986312e+01,
                -2.23307578892655734e-01]),
               np.array([2.96460137564761618e-03 * DAYS_PER_YEAR,
                2.37847173959480950e-03 * DAYS_PER_YEAR,
                -2.96589568540237556e-05 * DAYS_PER_YEAR]),
               4.36624404335156298e-05 * SOLAR_MASS),

    'neptune': (np.array([1.53796971148509165e+01,
                 -2.59193146099879641e+01,
                 1.79258772950371181e-01]),
                np.array([2.68067772490389322e-03 * DAYS_PER_YEAR,
                 1.62824170038242295e-03 * DAYS_PER_YEAR,
                 -9.51592254519715870e-05 * DAYS_PER_YEAR]),
                5.15138902046611451e-05 * SOLAR_MASS)}

VISIT_SCHEDULE = list(itertools.combinations(BODIES.keys(), 2))
BODIES1, BODIES2= zip(*VISIT_SCHEDULE)
BODIES_KEYS = list(BODIES.keys())
dt = 0.01

def to_do(body1, body2, BODIES=BODIES, dt=dt):
    (xyz1, v1, m1) = BODIES[body1]
    (xyz2, v2, m2) = BODIES[body2]
    d = xyz1-xyz2
    
    mag = dt * (np.inner(d,d) ** (-1.5))

    body1_velocity_arg = m2 * mag
    v1 = d * body1_velocity_arg


    body2_velocity_arg = m1 * mag       
    v2 = d * body1_velocity_arg


def update_rs_for_body(body, BODIES=BODIES, dt=dt):
    (r, vxyz, m) = BODIES[body]
    r = dt * vxyz

    
def report_energy(e=0.0, BODIES=BODIES, BODIES_KEYS=BODIES_KEYS):
    '''
        compute the energy and return it so that it can be printed
    '''
    for (body1, body2) in VISIT_SCHEDULE:
        (xyz1, v1, m1) = BODIES[body1]
        (xyz2, v2, m2) = BODIES[body2]

        d = xyz1 - xyz2

        # Amazingly the code as written is faster than this code below.
        #e -= (m1 * m2) / np.inner(d,d)**0.5
        e -= (m1 * m2) / (d[0]**2 + d[1]**2 + d[2]**2)**0.5

        
    for body in BODIES_KEYS:
        (r, v, m) = BODIES[body]
        e += m * np.inner(v,v) / 2.
        
    return e

def offset_momentum(ref, BODIES=BODIES, BODIES_KEYS=BODIES_KEYS):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    p = np.zeros(3)
    for body in BODIES_KEYS:
        (r, v, m) = BODIES[body]
        p -= v*m
        
    (r, v, m) = ref
    v = p / m


def nbody(loops, reference, iterations, VISIT_SCHEDULE=VISIT_SCHEDULE, BODIES=BODIES, BODIES_KEYS=BODIES_KEYS, BODIES1=BODIES1, BODIES2=BODIES2):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    # Set up global state
    offset_momentum(BODIES[reference])

    total_runs = loops * iterations
    for i, (body1, body2) in enumerate(itertools.cycle(VISIT_SCHEDULE)):
        to_do(body1, body2)
        
        if (i % iterations) == 0:
            for body in BODIES_KEYS:
                update_rs_for_body(body)
                
        if (i % loops) == 0:
            print(report_energy())
            
        if i >= total_runs - 1:
            break
      


if __name__ == '__main__':
    nbody(100, 'sun', 20000)
