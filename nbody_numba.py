"""

This was actually based off of nbody_iter.py, beacuse I didn't like my nbody_opt.py code


"""

from numba import jit
import numpy as np
import itertools

PI = 3.14159265358979323
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24

BODIES = {
    'sun': ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS),

    'jupiter': ([4.84143144246472090e+00,
                 -1.16032004402742839e+00,
                 -1.03622044471123109e-01],
                [1.66007664274403694e-03 * DAYS_PER_YEAR,
                 7.69901118419740425e-03 * DAYS_PER_YEAR,
                 -6.90460016972063023e-05 * DAYS_PER_YEAR],
                9.54791938424326609e-04 * SOLAR_MASS),

    'saturn': ([8.34336671824457987e+00,
                4.12479856412430479e+00,
                -4.03523417114321381e-01],
               [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                4.99852801234917238e-03 * DAYS_PER_YEAR,
                2.30417297573763929e-05 * DAYS_PER_YEAR],
               2.85885980666130812e-04 * SOLAR_MASS),

    'uranus': ([1.28943695621391310e+01,
                -1.51111514016986312e+01,
                -2.23307578892655734e-01],
               [2.96460137564761618e-03 * DAYS_PER_YEAR,
                2.37847173959480950e-03 * DAYS_PER_YEAR,
                -2.96589568540237556e-05 * DAYS_PER_YEAR],
               4.36624404335156298e-05 * SOLAR_MASS),

    'neptune': ([1.53796971148509165e+01,
                 -2.59193146099879641e+01,
                 1.79258772950371181e-01],
                [2.68067772490389322e-03 * DAYS_PER_YEAR,
                 1.62824170038242295e-03 * DAYS_PER_YEAR,
                 -9.51592254519715870e-05 * DAYS_PER_YEAR],
                5.15138902046611451e-05 * SOLAR_MASS)}

VISIT_SCHEDULE = list(itertools.combinations(BODIES.keys(), 2))
BODIES1, BODIES2= zip(*VISIT_SCHEDULE)
BODIES_KEYS = list(BODIES.keys())
dt = 0.01


# alter data types to arrays
# BODIES IS NOW SPLIT INTO BODIES1_XYZ, BODIES2_V, BODIES2_MASS
BODIES_2_XYZ = np.array([BODIES[k][0] for k in BODIES_KEYS])
BODIES_2_V = np.array([BODIES[k][1] for k in BODIES_KEYS])
BODIES_2_MASS = np.array([BODIES[k][2] for k in BODIES_KEYS])

BODIES_KEYS = np.array( [i for i, k in enumerate(BODIES_KEYS)])
VISIT_SCHEDULE = list(itertools.combinations(BODIES_KEYS, 2))
BODIES1, BODIES2= (np.array(it) for it in zip(*VISIT_SCHEDULE))

@jit("Tuple((float64[:], float64[:], float64)(int64, float64[:,:], float64[:,:], float64[:])")
def get_from_bodies(body, BODIES_2_XYZ=BODIES_2_XYZ, BODIES_2_V=BODIES_2_V, BODIES_2_MASS = BODIES_2_MASS):
    return (BODIES_2_XYZ[body], BODIES_2_V[body], BODIES_2_MASS[body])

@jit("void(int64, int64, float64)")
def to_do(body1, body2, dt=dt):
    (xyz1, v1, m1) = get_from_bodies(body1)
    (xyz2, v2, m2) = get_from_bodies(body2)
    d = xyz1 - xyz2
    
    mag = dt * (np.sum(d*d) ** (-1.5))

    body1_velocity_arg = m2 * mag
    v1 = v1 - d * body1_velocity_arg
    #v1[0] -= dx * body1_velocity_arg
    #v1[1] -= dy * body1_velocity_arg
    #v1[2] -= dz * body1_velocity_arg

    body2_velocity_arg = m1 * mag   
    v2 = v2 - d * body2_velocity_arg
    #v2[0] += dx * body2_velocity_arg
    #v2[1] += dy * body2_velocity_arg
    #v2[2] += dz * body2_velocity_arg

@jit("int64(int64, int64)")
def update_rs_for_body(body, dt=dt):
    (r, [vx, vy, vz], m) = get_from_bodies(body)
    r[0] += dt * vx
    r[1] += dt * vy
    r[2] += dt * vz 
    
@jit("int64(int64[:])")
def report_energy(BODIES_KEYS=BODIES_KEYS):
    '''
        compute the energy and return it so that it can be printed
    '''
    e = 0.0
    for (body1, body2) in VISIT_SCHEDULE:
        ((x1, y1, z1), v1, m1) = get_from_bodies(body1)
        ((x2, y2, z2), v2, m2) = get_from_bodies(body2)

        (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)

        e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
        
    for body in BODIES_KEYS:
        (r, v, m) = get_from_bodies(body)
        e += m * np.sum(v*v) / 2.
        
    return e

@jit("int64(int64, int64[:])")
def offset_momentum(ref, BODIES_KEYS=BODIES_KEYS):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    p = np.array([0,0,0], dtype='float64')
                         
    for body in BODIES_KEYS:
        (r, v, m) = get_from_bodies(body)
        p = p - v * m
        #px -= vx * m
        #py -= vy * m
        #pz -= vz * m
        
    (r, v, m) = ref
    v = p / m
    #v[0] = px / m
    #v[1] = py / m
    #v[2] = pz / m

def nbody(loops, reference, iterations, VISIT_SCHEDULE=VISIT_SCHEDULE,
          BODIES=BODIES, BODIES_KEYS=BODIES_KEYS, BODIES1=BODIES1, BODIES2=BODIES2):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    reference = 0
    # Set up global state
    offset_momentum(get_from_bodies(reference))

    total_runs = loops * iterations
    for i, (body1, body2) in enumerate(itertools.cycle(VISIT_SCHEDULE)):
        to_do(body1, body2)
        
        if (i % iterations) == 0:
            for body in BODIES_KEYS:
                update_rs_for_body(body)
                
        if (i % loops) == 0:
            print(report_energy())
            pass
            
        if i >= total_runs - 1:
            break

            
if __name__ == '__main__':
    nbody(10, 'sun', 2000)