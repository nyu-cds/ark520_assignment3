 

from itertools import combinations


cdef float PI 
PI = 3.14159265358979323

cdef float SOLAR_MASS
SOLAR_MASS = 4 * PI * PI

cdef float DAYS_PER_YEAR
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

VISIT_SCHEDULE = list(combinations(BODIES.keys(), 2))
BODIES1, BODIES2= zip(*VISIT_SCHEDULE)
BODIES_KEYS = list(BODIES.keys())
cdef float dt
dt = 0.01

def to_do(body1, body2, BODIES=BODIES, dt=dt):
    ([x1, y1, z1], v1, m1) = BODIES[body1]
    ([x2, y2, z2], v2, m2) = BODIES[body2]
    (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)
    
    mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))

    body1_velocity_arg = m2 * mag
    v1[0] -= dx * body1_velocity_arg
    v1[1] -= dy * body1_velocity_arg
    v1[2] -= dz * body1_velocity_arg

    body2_velocity_arg = m1 * mag            
    v2[0] += dx * body2_velocity_arg
    v2[1] += dy * body2_velocity_arg
    v2[2] += dz * body2_velocity_arg

def update_rs_for_body(body, BODIES=BODIES, dt=dt):
    (r, [vx, vy, vz], m) = BODIES[body]
    r[0] += dt * vx
    r[1] += dt * vy
    r[2] += dt * vz 
    
def advance(dt=dt, VISIT_SCHEDULE=VISIT_SCHEDULE, BODIES=BODIES, BODIES1=BODIES1, BODIES2=BODIES2):
    '''
        advance the system one timestep
    '''
    
    list(map(to_do, BODIES1, BODIES2))      
    list(map(update_rs_for_body, BODIES_KEYS))
    
    
def report_energy(e=0.0, BODIES=BODIES, BODIES_KEYS=BODIES_KEYS):
    '''
        compute the energy and return it so that it can be printed
    '''
    for (body1, body2) in VISIT_SCHEDULE:
        ((x1, y1, z1), v1, m1) = BODIES[body1]
        ((x2, y2, z2), v2, m2) = BODIES[body2]

        (dx, dy, dz) = (x1-x2, y1-y2, z1-z2)

        e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
        
    for body in BODIES_KEYS:
        (r, [vx, vy, vz], m) = BODIES[body]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e

def offset_momentum(ref, px=0.0, py=0.0, pz=0.0, BODIES=BODIES, BODIES_KEYS=BODIES_KEYS):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    for body in BODIES_KEYS:
        (r, [vx, vy, vz], m) = BODIES[body]
        px -= vx * m
        py -= vy * m
        pz -= vz * m
        
    (r, v, m) = ref
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m


def nbody(loops, reference, iterations, BODIES=BODIES, BODIES_KEYS=BODIES_KEYS, BODIES1=BODIES1, BODIES2=BODIES2):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    # Set up global state
    offset_momentum(BODIES[reference])

    print("HI3")
    for _ in range(loops):
        for _ in range(iterations):            
            for b1, b2 in zip(BODIES1, BODIES2):
                to_do(b1, b2)     
            for b in BODIES_KEYS:
                update_rs_for_body(b)
        print(report_energy())
            

if __name__ == '__main__':
    nbody(100, 'sun', 20000)