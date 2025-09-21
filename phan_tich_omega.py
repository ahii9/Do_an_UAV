import random
import math
from copy import deepcopy
from path_quality_utility import compute_sensor_SINR_dict_using_sensing, compute_link_sinr
from total_utility import compute_total_utility
from path_quality_utility import generate_candidate_paths, select_optimal_path
from sensing_utility import compute_sensing_radius, sensors_in_uav_area

# -----------------------------
# 3) Generate sensor data
# -----------------------------
def generate_sensor_clusters(seed, cluster_centers, std_dev):
    """
    Generate 4 clusters, each with 75 sensors => total 300 sensors
    (3 types: type1, type2, type3, each with 100 sensors)

    Parameters:
        seed: Random seed for reproducibility
        cluster_centers: List of cluster center coordinates
        std_dev: Standard deviation for sensor distribution

    Returns:
        tuple: (sensor_positions, sensor_types)
    """
    random.seed(seed)
    sensor_positions = []
    sensor_types = []

    for cx, cy in cluster_centers:
        # Each cluster: 75 sensors => 25 sensors per type
        for _ in range(25):
            x = random.gauss(cx, std_dev)
            y = random.gauss(cy, std_dev)
            sensor_positions.append((x, y))
            sensor_types.append("type1")
        for _ in range(25):
            x = random.gauss(cx, std_dev)
            y = random.gauss(cy, std_dev)
            sensor_positions.append((x, y))
            sensor_types.append("type2")
        for _ in range(25):
            x = random.gauss(cx, std_dev)
            y = random.gauss(cy, std_dev)
            sensor_positions.append((x, y))
            sensor_types.append("type3")
    return sensor_positions, sensor_types

# -----------------------------
# 4) Particle class for PSO
# -----------------------------
class Particle:
    def __init__(self, num_uavs, init_positions, x_min, x_max, y_min, y_max, v_min, v_max, h_fixed):
        """
        Initialize a particle with positions and velocities.
        If init_positions is provided, start near those positions.

        Parameters:
            num_uavs: Number of UAVs
            init_positions: Initial UAV positions
            x_min, x_max, y_min, y_max: Movement boundaries
            v_min, v_max: Velocity limits
            h_fixed: Fixed UAV height
        """
        self.num_uavs = num_uavs
        self.position = []
        self.velocity = []
        if init_positions is not None:
            for i in range(num_uavs):
                if i == 0:
                    # UAV0 = GBS (fixed)
                    self.position.append(init_positions[i])
                    self.velocity.append((0.0, 0.0))
                else:
                    x0, y0, z0 = init_positions[i]
                    self.position.append((x0, y0, z0))
                    vx = random.uniform(v_min, v_max)
                    vy = random.uniform(v_min, v_max)
                    self.velocity.append((vx, vy))
        else:
            for i in range(num_uavs):
                if i == 0:
                    self.position.append((0.0, 0.0, 0.0))
                    self.velocity.append((0.0, 0.0))
                else:
                    x = random.uniform(x_min, x_max)
                    y = random.uniform(y_min, y_max)
                    self.position.append((x, y, h_fixed))
                    vx = random.uniform(v_min, v_max)
                    vy = random.uniform(v_min, v_max)
                    self.velocity.append((vx, vy))

        self.best_position = deepcopy(self.position)
        self.best_fitness = -math.inf
        self.fitness = -math.inf

def evaluate_fitness(particle, sensor_positions, sensor_types, t_prev, current_time,
                     R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                     P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0):
    """
    Calculate the total fitness for a particle.

    Parameters:
        particle: Particle object to evaluate
        sensor_positions, sensor_types: Sensor positions and types
        t_prev: Last collection time for each sensor
        current_time: Current time
        Other parameters: Fitness calculation parameters

    Returns:
        float: total_fitness
    """
    t_prev_copy = deepcopy(t_prev)
    total_fitness, _, _ = compute_total_utility(
        particle.position, sensor_positions, sensor_types,
        current_time, t_prev_copy,
        R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
        P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
    )
    return total_fitness

def update_velocity_position(particle, global_best, omega, c1, c2, speed_factor, v_min, v_max, x_min, x_max, y_min, y_max):
    """
    Update particle velocity and position, applying SPEED_FACTOR for acceleration.

    Parameters:
        particle: Particle to update
        global_best: Global best position
        omega: Inertia weight
        c1, c2: Cognitive and social coefficients
        speed_factor: Acceleration factor
        v_min, v_max: Velocity limits
        x_min, x_max, y_min, y_max: Movement boundaries
    """
    for i in range(particle.num_uavs):
        if i == 0:
            particle.position[i] = particle.position[0]
            particle.velocity[i] = (0.0, 0.0)
            continue

        (x, y, z) = particle.position[i]
        (vx, vy) = particle.velocity[i]
        (lbx, lby, _) = particle.best_position[i]
        (gbx, gby, _) = global_best[i]

        r1 = random.random()
        r2 = random.random()

        new_vx = omega * vx + c1 * r1 * (lbx - x) + c2 * r2 * (gbx - x)
        new_vy = omega * vy + c1 * r1 * (lby - y) + c2 * r2 * (gby - y)

        new_vx *= speed_factor
        new_vy *= speed_factor

        new_vx = max(v_min, min(new_vx, v_max))
        new_vy = max(v_min, min(new_vy, v_max))

        nx = x + new_vx
        ny = y + new_vy
        nx = max(x_min, min(nx, x_max))
        ny = max(y_min, min(ny, y_max))

        particle.position[i] = (nx, ny, z)
        particle.velocity[i] = (new_vx, new_vy)

# -----------------------------
# 5) Run PSO for position optimization
# -----------------------------
def run_pso(uav_init_positions, sensor_positions, sensor_types, t_prev, current_time, current_fitness,
            num_particles, max_iterations, convergence_threshold,
            omega, c1, c2, speed_factor,
            x_min, x_max, y_min, y_max, v_min, v_max, h_fixed,
            R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
            P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0):
    """
    Run PSO to optimize UAV positions:
    - Iterate until max iterations or gbest improvement < epsilon
    - Maintain local and global best positions
    - Update velocities and positions

    Parameters:
        uav_init_positions: Initial UAV positions
        sensor_positions, sensor_types: Sensor positions and types
        t_prev: Last collection time for each sensor
        current_time: Current time
        current_fitness: Current fitness
        num_particles: Number of particles in swarm
        max_iterations: Maximum iterations
        convergence_threshold: Convergence threshold
        omega, c1, c2: Velocity update coefficients
        speed_factor: Acceleration factor
        x_min, x_max, y_min, y_max: Movement boundaries
        v_min, v_max: Velocity limits
        h_fixed: Fixed UAV height
        Other parameters: Fitness calculation parameters

    Returns:
        tuple: (global_best_fitness, global_best_position, better_than_current, iterations)
    """
    swarm = [Particle(num_uavs=5, init_positions=uav_init_positions,
                      x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                      v_min=v_min, v_max=v_max, h_fixed=h_fixed)
             for _ in range(num_particles)]
    global_best_fitness = -math.inf
    global_best_position = None
    prev_global_best = -math.inf
    iterations = 0

    for iteration in range(max_iterations):
        iterations += 1
        current_utility_iteration = -math.inf

        for p in swarm:
            fitness = evaluate_fitness(p, sensor_positions, sensor_types, t_prev, current_time,
                                      R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                                      P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0)
            p.fitness = fitness

            if fitness > p.best_fitness:
                p.best_fitness = fitness
                p.best_position = deepcopy(p.position)

            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = deepcopy(p.position)

            current_utility_iteration = max(current_utility_iteration, fitness)

        if global_best_position is not None:
            for p in swarm:
                update_velocity_position(p, global_best_position,
                                        omega, c1, c2, speed_factor,
                                        v_min, v_max, x_min, x_max, y_min, y_max)

        # Check for convergence based on gbest
        if iterations > 1 and abs(global_best_fitness - prev_global_best) < convergence_threshold:
            print(f"PSO converged after {iterations} iterations")
            break
        prev_global_best = global_best_fitness

    better_than_current = global_best_fitness > current_fitness
    if better_than_current:
        print(f"Found better position: {global_best_fitness:.4f} > {current_fitness:.4f}")
    else:
        print(f"No better position found: {global_best_fitness:.4f} <= {current_fitness:.4f}")

    return global_best_fitness, global_best_position, better_than_current, iterations

# -----------------------------
# 6) Reset VSI for collected sensors
# -----------------------------
def update_sensors_after_collection(uav_positions, sensor_positions, t_prev, current_time,
                                    G_St, G_Ur, lambda_S, P_st, P_U_min_r, h):
    """
    Reset t_prev[sensor_id] to current_time for sensors within UAV coverage,
    resetting VSI to Vmin.

    Parameters:
        uav_positions: UAV positions
        sensor_positions: Sensor positions
        t_prev: Last collection time for each sensor
        current_time: Current time
        G_St, G_Ur, lambda_S, P_st, P_U_min_r, h: Parameters for sensing radius

    Returns:
        tuple: (t_prev, collected_sensors)
    """
    sensing_r = compute_sensing_radius(G_St, G_Ur, lambda_S, P_st, P_U_min_r, h)
    uav_sensor_map, _ = sensors_in_uav_area(sensor_positions, uav_positions, sensing_r)

    collected_sensors = []
    for _, sensor_list in uav_sensor_map.items():
        for s_id in sensor_list:
            t_prev[s_id] = current_time
            collected_sensors.append(s_id)

    return t_prev, collected_sensors

# -----------------------------
# 8) Main - Analyze omega values with iteration tracking
# -----------------------------
def main():
    # 1) Common PSO parameters
    num_particles = 50
    max_pso_iterations = 200
    num_time_steps = 20
    convergence_threshold = 0.01
    c1 = 1
    c2 = 1
    speed_factor = 2.5
    omega_values = [0.4, 0.5, 0.6, 0.7, 0.8]  # Omega values to test

    x_min, x_max = 0.0, 2000.0
    y_min, y_max = 0.0, 2000.0
    h_fixed = 100.0
    v_min, v_max = -200.0, 200.0
    current_time = 0.0

    vsi_recovery_intervals = {
        "type1": 3,
        "type2": 5,
        "type3": 7
    }

    # 2) Parameters for total_utility and path_quality_utility
    R = [3, 5, 7]
    Vmax = [10, 15, 20]
    Vmin = [0.1, 0.15, 0.20]
    G_St = 1.0
    G_Ur = 1.0
    lambda_S = 3e8 / 2.4e9
    P_st = 0.006
    P_U_min_r = 10 ** ((-76 - 30) / 10)
    h = h_fixed
    P_tx_uav = 0.25
    G_tx_uav = 1.0
    G_rx_uav = 1.0
    N0 = 1e-11
    SINR_threshold = 4
    Q = 3
    N = 20
    cm = 0.5
    km = 0.5

    cluster_centers = [
        (400, 1600),
        (1200, 1700),
        (500, 400),
        (1500, 600)
    ]
    std_dev = 150.0

    # Initialize
    random.seed(123)
    sensor_positions, sensor_types = generate_sensor_clusters(
        seed=42,
        cluster_centers=cluster_centers,
        std_dev=std_dev
    )
    t_prev = [0.0] * len(sensor_positions)

    init_uav_positions = [
        (100.0, 100.0, 0.0),
        (500.0, 200.0, h_fixed),
        (600.0, 1000.0, h_fixed),
        (1500.0, 500.0, h_fixed),
        (1200.0, 1200.0, h_fixed)
    ]

    # File to store results
    with open("omega_analysis_results.txt", "w") as f:
        f.write("Omega Analysis Results\n")
        f.write("=====================\n")
        f.write("Time Step | Omega | Fitness | Sensors Collected | Iterations\n")

        # Test each omega value
        for omega in omega_values:
            print(f"\n===== Analysis with Omega = {omega} =====")
            current_time = 0.0
            t_prev = [0.0] * len(sensor_positions)
            current_uav_positions = deepcopy(init_uav_positions)

            # Calculate initial fitness
            current_fitness, _, _ = compute_total_utility(
                current_uav_positions, sensor_positions, sensor_types,
                current_time, t_prev,
                R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
            )
            f.write(f"0 | {omega} | {current_fitness:.4f} | 0 | 0\n")
            print(f"Initial fitness: {current_fitness:.4f}")

            # Time step loop
            for step in range(1, num_time_steps + 1):
                print(f"\n==== Time Step {step} (Tu = {step}) ====")
                current_time += 1

                current_fitness, _, _ = compute_total_utility(
                    current_uav_positions, sensor_positions, sensor_types,
                    current_time, t_prev,
                    R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                    P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
                )

                t_prev, collected_sensors = update_sensors_after_collection(
                    current_uav_positions, sensor_positions, t_prev, current_time,
                    G_St, G_Ur, lambda_S, P_st, P_U_min_r, h
                )

                print(f"Current fitness: {current_fitness:.4f}")

                print("Running PSO to find optimal position...")
                best_fitness, best_positions, is_better, iterations = run_pso(
                    current_uav_positions, sensor_positions, sensor_types,
                    t_prev, current_time, current_fitness,
                    num_particles, max_pso_iterations, convergence_threshold,
                    omega, c1, c2, speed_factor,
                    x_min, x_max, y_min, y_max, v_min, v_max, h_fixed,
                    R, Vmax, Vmin, G_St, G_Ur, lambda_S, P_st, P_U_min_r, h,
                    P_tx_uav, G_tx_uav, G_rx_uav, SINR_threshold, N, cm, km, Q, N0
                )

                if is_better and best_positions is not None:
                    print("UAV moving to new optimal position...")
                    current_uav_positions = deepcopy(best_positions)
                else:
                    print("Keeping current UAV position (no better position found)")

                print(f"Sensors collected: {len(collected_sensors)}")
                f.write(f"{step} | {omega} | {current_fitness:.4f} | {len(collected_sensors)} | {iterations}\n")

    print("\nAnalysis complete. Results saved to omega_analysis_results.txt.")

if __name__ == "__main__":
    main()